"""Replenishment suggestion for one port and category.

The chandler orders today. Goods arrive after the lead time. The selected
horizon is the review cycle: the next time someone is assumed to look at
this stock. On-hand and inbound stock therefore have to cover the lead time
plus the review cycle.

    cover = lead_time + horizon
    expected = sum of daily forecasts over the cover window
    safety = z * inflation * sqrt(sum of daily residual variances)
    order = max(0, expected + safety - on_hand - inbound)

``inflation`` corrects the independence assumption. It is the ratio of the
observed standard deviation of 14-day total errors to the standard deviation
implied by adding the daily residual variances. Daily residual scales come
from the time-based backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from chandlery_forecast.config import UNIT_PRICES_USD, UNITS


@dataclass(frozen=True)
class OrderSuggestion:
    """A single suggested purchase."""

    expected_demand: float
    safety_stock: float
    on_hand: float
    inbound: float
    lead_time_days: int
    horizon_days: int
    cover_days: int
    service_level: float
    z: float
    sigma_cover: float
    suggested_qty: float
    unit: str
    unit_price_usd: float
    indicative_value_usd: float
    note: str


def suggest_order(
    daily_forecast: np.ndarray,
    sigma_by_step: np.ndarray,
    inflation: float,
    on_hand: float,
    inbound: float = 0.0,
    lead_time_days: int = 0,
    service_level: float = 0.95,
    unit: str = "units",
    unit_price_usd: float = 0.0,
) -> OrderSuggestion:
    """Return a base-stock order for the cover window.

    ``daily_forecast[0]`` is the next day (step 1). ``sigma_by_step`` is
    aligned with it. Both must be at least ``lead_time_days + len`` long
    enough to cover the review cycle: the forecast length is the horizon,
    and the cover window cannot exceed the forecast length. Pass a forecast
    at least as long as the cover window.
    """

    forecast = np.asarray(daily_forecast, dtype=float)
    sigma = np.asarray(sigma_by_step, dtype=float)
    if forecast.ndim != 1 or sigma.ndim != 1:
        raise ValueError("forecast and sigma must be one-dimensional")
    if len(forecast) == 0:
        raise ValueError("forecast is empty")
    if len(sigma) != len(forecast):
        raise ValueError("sigma_by_step must match the forecast length")
    if np.any(forecast < 0) or np.any(sigma < 0):
        raise ValueError("forecast and sigma must be non-negative")
    if not 0.5 < service_level < 1.0:
        raise ValueError("service_level must be between 0.5 and 1")
    if lead_time_days < 0:
        raise ValueError("lead_time_days must be >= 0")
    if on_hand < 0 or inbound < 0:
        raise ValueError("stock positions must be >= 0")

    horizon_days = len(forecast) - lead_time_days
    if horizon_days < 1:
        raise ValueError("forecast must extend at least one day past the lead time")
    cover = lead_time_days + horizon_days
    expected = float(np.sum(forecast[:cover]))
    variance = float(np.sum(np.square(sigma[:cover])))
    sigma_cover = float(max(inflation, 0.0) * np.sqrt(variance))
    z = float(norm.ppf(service_level))
    safety = z * sigma_cover
    suggested = max(0.0, expected + safety - on_hand - inbound)
    value = suggested * unit_price_usd
    note = (
        f"Cover {cover} days ({lead_time_days} days of lead time + "
        f"{horizon_days} days until the next review) at a {service_level:.0%} service level."
    )
    return OrderSuggestion(
        expected_demand=expected,
        safety_stock=safety,
        on_hand=float(on_hand),
        inbound=float(inbound),
        lead_time_days=int(lead_time_days),
        horizon_days=int(horizon_days),
        cover_days=int(cover),
        service_level=float(service_level),
        z=z,
        sigma_cover=sigma_cover,
        suggested_qty=suggested,
        unit=unit,
        unit_price_usd=float(unit_price_usd),
        indicative_value_usd=value,
        note=note,
    )


def sigma_path(sigma_by_horizon: dict[int, float], steps: int, fallback: float) -> np.ndarray:
    """Daily residual scale for steps 1..``steps``.

    Horizons the backtest never scored reuse the scale of the longest
    observed step, then ``fallback`` if the category has no scale at all.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    known = sorted(int(h) for h in sigma_by_horizon)
    longest = sigma_by_horizon[known[-1]] if known else fallback
    out = np.empty(steps, dtype=float)
    for step in range(1, steps + 1):
        out[step - 1] = float(sigma_by_horizon.get(step, longest if known else fallback))
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=fallback)
    return out


def price_for(category: str) -> tuple[str, float]:
    """Synthetic unit and USD price for a category label."""

    return UNITS[category], float(UNIT_PRICES_USD[category])
