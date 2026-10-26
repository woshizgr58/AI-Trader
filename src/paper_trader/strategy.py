"""Deterministic allocations and conservative sizing shared by live and backtest."""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import math


TERMINAL_STATUSES = {"filled", "canceled", "cancelled", "expired", "rejected", "replaced"}


def _number(value, name, *, positive=False):
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number <= 0):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number.")
    return number


def target_weights(predictions, config):
    if set(predictions) != set(config.symbols):
        raise ValueError("Predictions must contain every configured asset and no others.")
    if any(not math.isfinite(float(value)) for value in predictions.values()):
        raise ValueError("Predictions must be finite; no orders are safe with missing model outputs.")
    eligible = [s for s in config.symbols if float(predictions[s]) > config.threshold]
    weight = min(config.max_asset_weight, config.max_total_weight / len(eligible)) if eligible else 0.0
    return {s: weight if s in eligible else 0.0 for s in config.symbols}


def execution_price(price, side, config):
    """Cent-rounded limit price; buy planning reserves this worst-case price."""
    price = _number(price, "quote price", positive=True)
    adjusted = Decimal(str(price)) * (Decimal(1) + Decimal(str(config.slippage_bps)) / 10000 * (1 if side == "buy" else -1))
    adjusted = adjusted.quantize(Decimal("0.01"), rounding=ROUND_CEILING if side == "buy" else ROUND_FLOOR)
    return _number(adjusted, "execution price", positive=True)


def plan_orders(targets, snapshot, prices, config, pending_orders=None):
    """Reserve pending buys; never count sale proceeds before a fresh broker snapshot.

    Quantity differences include unfilled pending orders. Exposure headroom does
    not credit pending/new sells, because they may never fill. Existing limit
    breaches do not block risk-reducing sells; they prevent additional buys.
    """
    if set(targets) != set(config.symbols):
        raise ValueError("Targets must match the configured universe.")
    weights = {s: _number(v, f"{s} target") for s, v in targets.items()}
    if any(v > config.max_asset_weight + 1e-10 for v in weights.values()) or sum(weights.values()) > config.max_total_weight + 1e-10:
        raise ValueError("Targets exceed configured exposure limits.")
    equity = _number(snapshot["equity"], "equity", positive=True)
    cash = _number(snapshot["cash"], "cash")
    buying_power = _number(snapshot["buying_power"], "buying power")
    positions = snapshot.get("positions", {})
    unknown = set(positions).difference(config.symbols)
    if unknown:
        raise ValueError(f"Unmanaged positions require reconciliation: {sorted(unknown)}")
    quotes = {s: _number(prices[s]["price"] if isinstance(prices[s], dict) else prices[s], f"{s} quote", positive=True) for s in config.symbols}
    current = {s: _number(positions.get(s, {}).get("qty", 0), f"{s} quantity") for s in config.symbols}
    projected = current.copy()
    pending_sells = {s: 0.0 for s in config.symbols}
    # Use the larger account mark / current quote when reserving exposure.
    marks = {s: max(quotes[s], _number(positions.get(s, {}).get("current_price", quotes[s]), f"{s} account mark")) for s in config.symbols}
    asset_values = {s: max(current[s] * marks[s], _number(positions.get(s, {}).get("market_value", 0), f"{s} value")) for s in config.symbols}
    fee_rate = config.fee_bps / 10000
    reserved = 0.0
    pending = snapshot.get("open_orders", []) if pending_orders is None else pending_orders
    for order in pending:
        if str(order.get("status", "new")).lower() in TERMINAL_STATUSES:
            continue
        symbol = order["symbol"]
        if symbol not in quotes:
            raise ValueError("Unmanaged open orders require reconciliation.")
        qty = _number(order["qty"], "order quantity")
        filled = _number(order.get("filled_qty", 0), "filled quantity")
        if filled > qty:
            raise ValueError("Filled order quantity exceeds submitted quantity.")
        remaining = qty - filled
        side = str(order["side"]).lower()
        if side == "buy":
            price = max(execution_price(quotes[symbol], "buy", config), marks[symbol], _number(order.get("limit_price") or order.get("estimated_price") or quotes[symbol], "pending buy price", positive=True))
            projected[symbol] += remaining
            reserved += remaining * price * (1 + fee_rate)
            asset_values[symbol] += remaining * price
        elif side == "sell":
            projected[symbol] -= remaining
            pending_sells[symbol] += remaining
            if pending_sells[symbol] > current[symbol] + 1e-8:
                raise ValueError("Outstanding sells exceed held quantity.")
        else:
            raise ValueError("Unknown pending order side.")
    budget = max(0.0, min(cash, buying_power) - equity * config.cash_buffer - reserved)
    starting_budget = budget
    total_value = sum(asset_values.values())
    # Caps use an equity floor after worst-case pending/new transaction fees.
    # No sale value is credited to exposure headroom. Reserving all existing
    # position sale fees is conservative and avoids cross-order fee leakage.
    fee_equity_floor = equity - fee_rate * (
        reserved / (1 + fee_rate) + budget / (1 + fee_rate)
        + sum(current[s] * marks[s] for s in current)
    )
    if fee_equity_floor <= 0:
        raise ValueError("Transaction cost reserve would exhaust account equity.")
    orders = []
    target_quantities = {}
    for symbol in sorted(config.symbols):
        # Size target at the buy limit so execution slippage cannot over-allocate.
        buy_price = execution_price(quotes[symbol], "buy", config)
        desired = math.floor(equity * weights[symbol] / buy_price + 1e-12)
        target_quantities[symbol] = desired
        delta = desired - projected[symbol]
        if abs(delta) < 1:
            continue
        side = "buy" if delta > 0 else "sell"
        price = buy_price if side == "buy" else execution_price(quotes[symbol], "sell", config)
        qty = min(math.floor(abs(delta) + 1e-12), math.floor(config.max_order_notional / price + 1e-12))
        if side == "buy":
            asset_headroom = max(0.0, fee_equity_floor * config.max_asset_weight - asset_values[symbol])
            total_headroom = max(0.0, fee_equity_floor * config.max_total_weight - total_value)
            cost_per_share = price * (1 + fee_rate)
            exposure_price = max(price, marks[symbol])
            qty = min(qty, math.floor(budget / cost_per_share + 1e-12),
                      math.floor(asset_headroom / exposure_price + 1e-12),
                      math.floor(total_headroom / exposure_price + 1e-12))
            budget -= qty * cost_per_share
            asset_values[symbol] += qty * exposure_price
            total_value += qty * exposure_price
        else:
            # Unfilled buys are never sellable inventory.
            qty = min(qty, math.floor(current[symbol] - pending_sells[symbol] + 1e-12))
        if qty > 0:
            orders.append({"symbol": symbol, "side": side, "qty": int(qty),
                           "estimated_price": price, "estimated_notional": round(qty * price, 8)})
    # Sells first for observability only: purchase sizing above never used their proceeds.
    orders.sort(key=lambda order: (order["side"] != "sell", order["symbol"]))
    return {"orders": orders, "target_quantities": target_quantities,
            "projected_cash": cash - reserved - (starting_budget - budget),
            "projected_exposure": total_value / fee_equity_floor,
            "risk_checks": {"long_only": True, "whole_shares": True, "cash_budget": starting_budget,
                            "pending_buy_reserve": reserved, "sale_proceeds_assumed": False,
                            "fee_adjusted_equity_floor": fee_equity_floor,
                            "existing_exposure": sum(current[s] * marks[s] for s in current) / equity,
                            "order_notional_limit": config.max_order_notional}}
