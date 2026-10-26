"""Small, paper-only Alpaca adapter. No write request is automatically retried."""
from __future__ import annotations

from datetime import datetime, timezone
import math
import time
from typing import Any

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest
import requests

from .config import validate_paper_environment


PAPER_URL = "https://paper-api.alpaca.markets"
TERMINAL_STATUSES = frozenset({"filled", "canceled", "expired", "rejected", "replaced"})
ORDER_STATUSES = frozenset(status.value for status in OrderStatus)


class BrokerError(RuntimeError):
    """Sanitized, actionable broker failure (never includes request headers)."""


class SubmissionUncertain(BrokerError):
    """A POST may have arrived; reconciliation is required before any retry."""


class _PaperSession(requests.Session):
    def request(self, method: str, url: str, **kwargs: Any):
        if not url.startswith(PAPER_URL + "/v2/"):
            raise BrokerError("Only Alpaca's fixed paper-trading endpoint is permitted.")
        kwargs["timeout"] = (5, 20)
        kwargs["allow_redirects"] = False
        return super().request(method, url, **kwargs)


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise BrokerError(f"Broker returned invalid {name}; no new orders are safe.") from None
    if not math.isfinite(result):
        raise BrokerError(f"Broker returned nonfinite {name}; no new orders are safe.")
    return result


def _order(value: Any) -> dict:
    raw = _plain(value)
    return {
        "id": str(raw["id"]), "client_order_id": str(raw["client_order_id"]),
        "symbol": str(raw["symbol"]), "side": str(raw["side"]),
        "qty": _number(raw.get("qty"), "order quantity"),
        "filled_qty": _number(raw.get("filled_qty", 0), "filled quantity"),
        "status": str(raw["status"]), "type": raw.get("type"),
        "time_in_force": raw.get("time_in_force"),
        "limit_price": float(raw["limit_price"]) if raw.get("limit_price") else None,
        "filled_avg_price": float(raw["filled_avg_price"]) if raw.get("filled_avg_price") else None,
        "submitted_at": raw.get("submitted_at"), "updated_at": raw.get("updated_at"),
        "filled_at": raw.get("filled_at"), "canceled_at": raw.get("canceled_at"),
        "failed_at": raw.get("failed_at"), "legs": raw.get("legs"),
    }


class AlpacaBroker:
    """Use a dedicated paper account; outside orders are included in all checks."""

    def __init__(self, config):
        validate_paper_environment()
        if not config.api_key or not config.secret_key:
            raise BrokerError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY for an Alpaca paper account.")
        self.client = TradingClient(config.api_key, config.secret_key, paper=True)
        if getattr(self.client._base_url, "value", self.client._base_url) != PAPER_URL:
            raise BrokerError("The SDK did not select the required paper endpoint.")
        # alpaca-py's constructor ignores retry_attempts=0. Explicitly disable
        # its retry loop so an ambiguous POST is never replayed by the SDK.
        # The pinned SDK's transport interface is covered by adapter tests.
        self.client._retry = 0
        self.client._session.close()
        self.client._session = _PaperSession()

    def _read(self, method, *args, **kwargs):
        for attempt in range(3):
            try:
                return method(*args, **kwargs)
            except APIError as exc:
                code = exc.status_code
                if code == 404:
                    raise
                if code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise BrokerError(f"Alpaca read failed (HTTP {code}); check credentials, access and provider status.") from None
            except (requests.RequestException, TimeoutError, ConnectionError):
                if attempt == 2:
                    raise BrokerError("Alpaca read timed out or connection failed; retry after connectivity recovers.") from None
            time.sleep(0.5 * (2 ** attempt))

    def get_order(self, client_order_id: str) -> dict | None:
        try:
            result = self._read(self.client.get_order_by_client_id, client_order_id)
        except APIError as exc:
            if exc.status_code == 404:
                return None
            raise BrokerError("Could not reconcile the paper order by its stable identifier.") from None
        return _order(result)

    def snapshot(self) -> dict:
        """Reject snapshots crossing fills; do not trade on a torn account read."""
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True)
        first_orders = [_order(item) for item in self._read(self.client.get_orders, filter=request)]
        first_positions = [_plain(item) for item in self._read(self.client.get_all_positions)]
        account = _plain(self._read(self.client.get_account))
        recent = [_order(item) for item in self._read(
            self.client.get_orders, filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=100, nested=True)
        )]
        clock = _plain(self._read(self.client.get_clock))
        last_positions = [_plain(item) for item in self._read(self.client.get_all_positions)]
        last_orders = [_order(item) for item in self._read(self.client.get_orders, filter=request)]
        def order_signature(rows):
            return sorted((o["id"], o["qty"], o["filled_qty"], o["status"]) for o in rows)

        def position_signature(rows):
            return sorted((str(p["symbol"]), str(p["qty"])) for p in rows)
        if len(first_orders) >= 500 or len(last_orders) >= 500:
            raise BrokerError("Open-order query reached 500; full reconciliation is unavailable. Reduce open orders.")
        if order_signature(first_orders) != order_signature(last_orders) or position_signature(first_positions) != position_signature(last_positions):
            raise BrokerError("Account changed while being read (a fill or cancellation); rerun to reconcile before trading.")
        if account.get("trading_blocked") or account.get("account_blocked") or str(account.get("status")) != "ACTIVE":
            raise BrokerError("Alpaca paper account is not active for trading; review its account status.")
        positions = {
            str(p["symbol"]): {
                "qty": _number(p["qty"], "position quantity"),
                "market_value": _number(p["market_value"], "position value"),
                "current_price": _number(p["current_price"], "position price"),
            }
            for p in last_positions
        }
        return {
            "account_id": str(account["id"]),
            "cash": _number(account["cash"], "cash"),
            "buying_power": _number(account["buying_power"], "buying power"),
            "equity": _number(account["equity"], "equity"),
            "positions": positions, "open_orders": last_orders, "recent_orders": recent,
            "market_open": bool(clock["is_open"]), "broker_timestamp": clock["timestamp"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "paper": True,
        }

    def submit_order(self, order: dict) -> dict:
        quantity = _number(order.get("qty"), "proposed quantity")
        price = _number(order.get("limit_price"), "proposed limit price")
        if quantity <= 0 or not quantity.is_integer() or price <= 0 or order.get("side") not in {"buy", "sell"}:
            raise BrokerError("Only positive whole-share DAY limit paper orders are supported.")
        request = LimitOrderRequest(
            symbol=order["symbol"], qty=int(quantity), side=OrderSide(order["side"]),
            time_in_force=TimeInForce.DAY, limit_price=price,
            client_order_id=order["client_order_id"], extended_hours=False,
        )
        try:
            return _order(self.client.submit_order(order_data=request))
        except Exception:
            # Even 4xx may represent duplicate identifiers after a lost response.
            # Never infer rejection from the exception body and replay a POST.
            raise SubmissionUncertain("Paper order submission was not confirmed; reconcile its client order ID before any further submissions.") from None

    def cancel_open_orders(self) -> list[dict]:
        try:
            return [_plain(item) for item in self.client.cancel_orders()]
        except Exception:
            raise BrokerError("Paper cancellation was not confirmed. Refresh broker orders before retrying cancellation.") from None
