"""Chronological model selection with purged boundaries and explicit promotion."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, HORIZON, SCHEMA_VERSION, build_features, data_fingerprint


def chronological_split(features):
    """Split complete labels 60/20/20 by session, purging both earlier boundaries."""
    clean = features.dropna(subset=FEATURE_COLUMNS + ["target", "label_end"]).copy()
    dates = sorted(clean["session"].unique())
    if len(dates) < 150:
        raise ValueError("Training needs at least 150 fully featured, labelled sessions (about 215 input sessions). Fetch more history.")
    validation_start, test_start = dates[int(len(dates) * .6)], dates[int(len(dates) * .8)]
    train = clean.loc[(clean.session < validation_start) & (clean.label_end < validation_start)]
    validation = clean.loc[(clean.session >= validation_start) & (clean.session < test_start) & (clean.label_end < test_start)]
    test = clean.loc[clean.session >= test_start]
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Purging left an empty chronological partition; fetch more history.")
    return {"train": train, "validation": validation, "test": test}


def prediction_metrics(actual, predictions):
    actual, predictions = np.asarray(actual), np.asarray(predictions)
    correlation = None
    if np.std(actual) > 0 and np.std(predictions) > 0:
        correlation = float(np.corrcoef(actual, predictions)[0, 1])
    return {"rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
            "mae": float(mean_absolute_error(actual, predictions)),
            "direction_accuracy": float(np.mean((actual > 0) == (predictions > 0))),
            "correlation": correlation, "observations": len(actual)}


def _partition_metadata(frame):
    return {"start": str(frame.session.min()), "end": str(frame.session.max()),
            "label_end": str(frame.label_end.max()), "rows": len(frame),
            "sessions": sorted(frame.session.unique().tolist())}


def _write_json(path, value):
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def train(config, bars):
    expected_symbols = set(config.symbols) | {"SPY"}
    if "symbol" in bars and (missing := expected_symbols - set(bars.symbol)):
        raise ValueError(f"Training data is missing configured assets: {sorted(missing)}. Refresh the configured universe.")
    features = build_features(bars)
    features = features.loc[features.symbol.isin(config.symbols)]
    split = chronological_split(features)
    train_rows, validation = split["train"], split["validation"]
    candidates = {
        "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": Pipeline([("scale", StandardScaler()), ("model", RandomForestRegressor(
            n_estimators=64, max_depth=5, min_samples_leaf=20, max_features=1.0,
            random_state=config.seed, n_jobs=1))]),
    }
    results = {}
    for name, estimator in candidates.items():
        estimator.fit(train_rows[FEATURE_COLUMNS], train_rows.target)
        results[name] = {"validation": prediction_metrics(validation.target, estimator.predict(validation[FEATURE_COLUMNS]))}
    selected = min(results, key=lambda name: results[name]["validation"]["rmse"])
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = Path(config.models_dir) / version
    directory.mkdir(parents=True, exist_ok=False)
    # The snapshot makes evaluation reproducible even if adjusted provider history is revised later.
    bars.sort_values(["session", "symbol"]).to_csv(directory / "bars.csv.gz", index=False, compression="gzip", float_format="%.17g")
    joblib.dump({"candidates": candidates, "selected": selected}, directory / "model.joblib")
    artifact_sha256 = hashlib.sha256((directory / "model.joblib").read_bytes()).hexdigest()
    metadata = {
        "version": version, "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected, "selection_rule": "Lowest validation RMSE; test never used for selection; no refit after selection.",
        "feature_schema": FEATURE_COLUMNS, "feature_schema_version": SCHEMA_VERSION,
        "horizon_sessions": HORIZON, "target": "close[t+5]/close[t]-1 on provider split/dividend-adjusted daily bars",
        "seed": config.seed, "config": config.public_dict(), "symbols": list(config.symbols),
        "data_fingerprint": data_fingerprint(bars), "data_start": str(bars.session.min()), "data_end": str(bars.session.max()),
        "model_sha256": artifact_sha256, "sklearn_version": sklearn.__version__,
        "partitions": {name: _partition_metadata(frame) for name, frame in split.items()},
        "purge_rule": "Earlier partition label_end < next partition start (five signal sessions removed at each boundary).",
        "evaluation": results, "test_evaluated": False,
        "parameters": {"ridge": {"alpha": 1.0}, "random_forest": {"n_estimators": 64, "max_depth": 5, "min_samples_leaf": 20, "max_features": 1.0}},
        "preprocessing": "StandardScaler fitted on training rows only; complete feature rows required; no imputation.",
        "artifact_path": str(directory),
    }
    _write_json(directory / "metadata.json", metadata)
    return metadata


def load_version(config, version):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(version)):
        raise ValueError("Invalid model version.")
    directory = Path(config.models_dir) / version
    if not (directory / "metadata.json").exists():
        raise ValueError(f"Model version {version} does not exist; run train first.")
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["feature_schema"] != FEATURE_COLUMNS or metadata["feature_schema_version"] != SCHEMA_VERSION:
        raise ValueError("Model feature schema is incompatible. Train a new version.")
    if metadata["sklearn_version"] != sklearn.__version__:
        raise ValueError("Saved model sklearn version differs; restore locked dependencies or retrain.")
    if hashlib.sha256((directory / "model.joblib").read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Model artifact checksum mismatch; restore or retrain the model.")
    # Load only application-created local artifacts. Never accept downloaded pickle/joblib files.
    return joblib.load(directory / "model.joblib"), metadata


def promote(config, version):
    _, metadata = load_version(config, version)
    if tuple(metadata["symbols"]) != tuple(config.symbols) or metadata["config"]["feed"] != config.feed:
        raise ValueError("Model universe or data feed differs from current configuration; retrain first.")
    pointer = {"version": version, "promoted_at": datetime.now(timezone.utc).isoformat()}
    _write_json(Path(config.models_dir) / "active.json", pointer)
    return pointer


def load_active(config):
    pointer_path = Path(config.models_dir) / "active.json"
    if not pointer_path.exists():
        raise ValueError("No active model. Train, evaluate, then explicitly promote a saved version.")
    pointer = json.loads(pointer_path.read_text())
    artifact, metadata = load_version(config, pointer["version"])
    if tuple(metadata["symbols"]) != tuple(config.symbols) or metadata["config"]["feed"] != config.feed:
        raise ValueError("Active model universe or feed differs from configuration; train and promote a compatible model.")
    return artifact["candidates"][artifact["selected"]], metadata


def predict_active(config, bars, signal_session):
    model, metadata = load_active(config)
    if signal_session < metadata["partitions"]["validation"]["label_end"]:
        raise ValueError("The model was selected using labels after this signal session; historical inference would leak future information.")
    # Explicit cutoff also protects callers that accidentally pass future bars.
    history = bars.loc[bars.session <= signal_session].copy()
    rows = build_features(history)
    rows = rows.loc[(rows.session == signal_session) & rows.symbol.isin(config.symbols)]
    if set(rows.symbol) != set(config.symbols) or rows[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Missing complete features for the signal session; refresh sufficient real history.")
    predictions = dict(zip(rows.symbol, map(float, model.predict(rows[FEATURE_COLUMNS])), strict=True))
    reference = {"signal_session": signal_session, "history_fingerprint": data_fingerprint(history),
                 "feature_schema_version": SCHEMA_VERSION,
                 "features": {row["symbol"]: {column: float(row[column]) for column in FEATURE_COLUMNS} for row in rows.to_dict("records")}}
    return predictions, reference, metadata
