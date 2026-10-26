import json
from dataclasses import replace

import pytest

from paper_trader.config import Config, load_config
from paper_trader.store import Store


@pytest.mark.parametrize("name,value", [
    ("APCA_API_BASE_URL", "https://api.alpaca.markets"),
    ("ALPACA_BASE_URL", "https://paper-api.alpaca.markets.evil.example"),
    ("ALPACA_PAPER", "false"),
    ("TRADING_MODE", "live"),
])
def test_live_configuration_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="paper|live"):
        Config()


def test_unknown_config_and_credentials_in_toml_rejected(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('paper = false\n')
    with pytest.raises(ValueError, match="Unsupported"):
        load_config(path)
    path.write_text('api_key = "not-a-real-key"\n')
    with pytest.raises(ValueError, match="environment"):
        load_config(path)


def test_execution_default_and_public_config(monkeypatch, tmp_path):
    monkeypatch.delenv("PAPER_EXECUTION_ENABLED", raising=False)
    config = load_config(tmp_path / "config.toml")
    assert not config.paper_execution_enabled
    assert config.data_dir == tmp_path / "data"
    config = replace(config, api_key="fixture-key", secret_key="fixture-secret")
    assert "fixture-key" not in repr(config)
    assert "fixture-secret" not in json.dumps(config.public_dict())


@pytest.mark.parametrize("field,value", [("max_asset_weight", 1.5), ("max_total_weight", 2), ("max_drawdown", float("nan")), ("cash_buffer", -1), ("max_order_notional", 0)])
def test_risk_configuration_bounds(field, value):
    with pytest.raises(ValueError):
        replace(Config(), **{field: value})


def test_store_persistence_lock_and_redaction(tmp_path, monkeypatch):
    store = Store(tmp_path / "paper.sqlite")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-fixture-secret")
    store.set_control("paused", True)
    store.event("error", "provider echoed test-fixture-secret", {"api_key": "fixture"})
    restarted = Store(store.path)
    assert restarted.get_control("paused") is True
    assert "test-fixture-secret" not in json.dumps(restarted.events())
    with store.lock("trading"):
        with pytest.raises(RuntimeError, match="already running"):
            with restarted.lock("trading"):
                pass
    with restarted.lock("trading"):
        pass


def test_transaction_rolls_back(tmp_path):
    store = Store(tmp_path / "paper.sqlite")
    with pytest.raises(RuntimeError):
        with store.connection() as con:
            con.execute("INSERT INTO controls VALUES ('test', 'true')")
            raise RuntimeError("fixture interruption")
    assert store.get_control("test") is None
