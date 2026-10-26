"""Verify the installed supported SDK adapter, with all networking mocked."""
import json
from unittest.mock import Mock

from alpaca.common.exceptions import APIError
from alpaca.trading.enums import TimeInForce
import pytest
import requests

from paper_trader.broker import AlpacaBroker, BrokerError, PAPER_URL, SubmissionUncertain, _PaperSession
from paper_trader.config import Config


@pytest.fixture
def broker():
    return AlpacaBroker(Config(api_key="fixture-key", secret_key="fixture-secret"))


def response_error(code):
    response = requests.Response()
    response.status_code = code
    response._content = b'{"code": 400, "message": "fixture error"}'
    return APIError(response.text, requests.HTTPError(response=response))


def raw_order():
    return {"id": "1", "client_order_id": "stable", "symbol": "SPY", "qty": "4",
            "filled_qty": "1", "side": "buy", "status": "partially_filled",
            "type": "limit", "limit_price": "100.10", "filled_avg_price": "100"}


def test_paper_endpoint_timeout_and_sdk_retries_disabled(broker):
    assert broker.client._base_url.value == PAPER_URL
    assert broker.client._retry == 0
    assert isinstance(broker.client._session, _PaperSession)


def test_transport_rejects_live_and_redirects(broker, monkeypatch):
    request = Mock(return_value=Mock())
    monkeypatch.setattr(requests.Session, "request", request)
    with pytest.raises(BrokerError, match="fixed paper"):
        broker.client._session.request("POST", "https://api.alpaca.markets/v2/orders")
    assert request.call_count == 0
    broker.client._session.request("GET", PAPER_URL + "/v2/account")
    assert request.call_args.kwargs["timeout"] == (5, 20)
    assert request.call_args.kwargs["allow_redirects"] is False


def test_constructor_rechecks_endpoint_environment(monkeypatch):
    config = Config(api_key="fixture-key", secret_key="fixture-secret")
    monkeypatch.setenv("APCA_API_BASE_URL", "https://api.alpaca.markets")
    with pytest.raises(ValueError, match="only the Alpaca paper"):
        AlpacaBroker(config)


def test_only_404_means_no_order(broker, monkeypatch):
    lookup = Mock(side_effect=response_error(404))
    monkeypatch.setattr(broker.client, "get_order_by_client_id", lookup)
    assert broker.get_order("stable") is None
    lookup.side_effect = response_error(403)
    with pytest.raises(BrokerError, match="HTTP 403"):
        broker.get_order("stable")


def test_transient_reads_retry_but_return_clear_failure(broker, monkeypatch):
    monkeypatch.setattr("paper_trader.broker.time.sleep", lambda _: None)
    read = Mock(side_effect=[response_error(429), requests.Timeout(), raw_order()])
    monkeypatch.setattr(broker.client, "get_order_by_client_id", read)
    assert broker.get_order("stable")["filled_qty"] == 1
    assert read.call_count == 3


def test_submit_uses_sdk_limit_order_and_never_retries(broker, monkeypatch):
    submit = Mock(return_value=raw_order())
    monkeypatch.setattr(broker.client, "submit_order", submit)
    intent = {"symbol": "SPY", "qty": 4, "side": "buy", "limit_price": 100.10, "client_order_id": "stable"}
    broker.submit_order(intent)
    request = submit.call_args.kwargs["order_data"]
    assert request.time_in_force == TimeInForce.DAY
    assert request.limit_price == 100.10
    assert request.extended_hours is False
    submit.reset_mock(side_effect=True)
    submit.side_effect = requests.Timeout("fixture secret should never leak")
    with pytest.raises(SubmissionUncertain) as error:
        broker.submit_order(intent)
    assert "fixture secret" not in str(error.value)
    assert submit.call_count == 1


def test_real_sdk_transport_does_not_retry_429_post(broker, monkeypatch):
    response = requests.Response()
    response.status_code = 429
    response._content = json.dumps({"code": 42910000, "message": "fixture rate limit"}).encode()
    http = Mock(return_value=response)
    monkeypatch.setattr(requests.Session, "request", http)
    with pytest.raises(SubmissionUncertain):
        broker.submit_order({"symbol": "SPY", "qty": 4, "side": "buy", "limit_price": 100.10, "client_order_id": "stable"})
    assert http.call_count == 1


def test_snapshot_rejects_a_fill_during_nonatomic_reads(broker, monkeypatch):
    monkeypatch.setattr(broker.client, "get_orders", Mock(side_effect=[[raw_order()], [raw_order()], [{**raw_order(), "filled_qty": 2}]]))
    monkeypatch.setattr(broker.client, "get_all_positions", Mock(return_value=[]))
    monkeypatch.setattr(broker.client, "get_account", Mock(return_value={"id": "paper", "status": "ACTIVE", "cash": 1000, "buying_power": 1000, "equity": 1000}))
    monkeypatch.setattr(broker.client, "get_clock", Mock(return_value={"is_open": True, "timestamp": "2026-10-13T14:00:00+00:00"}))
    with pytest.raises(BrokerError, match="changed while being read"):
        broker.snapshot()


def test_cancel_does_not_liquidate_positions(broker, monkeypatch):
    cancel = Mock(return_value=[{"id": "fixture", "status": 200}])
    close = Mock()
    monkeypatch.setattr(broker.client, "cancel_orders", cancel)
    monkeypatch.setattr(broker.client, "close_all_positions", close)
    assert broker.cancel_open_orders() == [{"id": "fixture", "status": 200}]
    assert cancel.call_count == 1 and close.call_count == 0
