"""Base-stock arithmetic for the suggested purchase."""

import numpy as np
import pytest

from chandlery_forecast.inventory import sigma_path, suggest_order


def _order(**overrides):
    forecast = overrides.pop("forecast", np.full(5, 20.0))
    sigma = overrides.pop("sigma", np.full(5, 4.0))
    params = dict(
        daily_forecast=forecast,
        sigma_by_step=sigma,
        inflation=1.0,
        on_hand=0.0,
        inbound=0.0,
        lead_time_days=0,
        service_level=0.95,
        unit="kg",
        unit_price_usd=2.0,
    )
    params.update(overrides)
    return suggest_order(**params)


def test_overstock_orders_nothing():
    suggestion = _order(on_hand=10_000)
    assert suggestion.suggested_qty == 0
    assert suggestion.cover_days == 5


def test_higher_service_level_does_not_shrink_the_order():
    low = _order(service_level=0.80, on_hand=50)
    high = _order(service_level=0.99, on_hand=50)
    assert high.suggested_qty > low.suggested_qty
    assert high.z > low.z


def test_lead_time_is_part_of_the_cover_window():
    forecast = np.arange(1, 11, dtype=float)
    suggestion = _order(forecast=forecast, sigma=np.ones(10), lead_time_days=3, on_hand=0)
    assert suggestion.cover_days == 10
    assert suggestion.horizon_days == 7
    assert suggestion.expected_demand == pytest.approx(55)


def test_zero_uncertainty_and_empty_stock_orders_the_forecast():
    suggestion = _order(sigma=np.zeros(5), service_level=0.95, on_hand=0, inbound=0)
    assert suggestion.safety_stock == pytest.approx(0)
    assert suggestion.suggested_qty == pytest.approx(100)
    assert suggestion.indicative_value_usd == pytest.approx(200)


def test_sigma_path_reuses_the_longest_known_step():
    path = sigma_path({1: 2.0, 3: 4.0}, steps=5, fallback=9.0)
    assert path.tolist() == [2.0, 4.0, 4.0, 4.0, 4.0]
    empty = sigma_path({}, steps=2, fallback=9.0)
    assert empty.tolist() == [9.0, 9.0]
