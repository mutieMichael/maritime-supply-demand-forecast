"""Model contracts that do not need the full backtest."""

import numpy as np
import pandas as pd

from chandlery_forecast.backtest import _error_metrics
from chandlery_forecast.features import build_supervised
from chandlery_forecast.generate import generate_dataset
from chandlery_forecast.models import FourierRidgeModel, GradientBoostModel, holt_winters_forecast, seasonal_naive_forecast


def test_seasonal_naive_repeats_the_last_week():
    history = np.arange(1, 8, dtype=float)
    forecast = seasonal_naive_forecast(history, horizon=8)
    assert forecast.tolist() == [1, 2, 3, 4, 5, 6, 7, 1]


def test_holt_winters_forecast_is_non_negative():
    rng = np.random.default_rng(0)
    history = 40 + 8 * np.sin(np.arange(70) * 2 * np.pi / 7) + rng.normal(0, 1, 70)
    forecast = holt_winters_forecast(history, horizon=7)
    assert forecast.shape == (7,)
    assert np.all(forecast >= 0)


def test_error_metrics_match_a_hand_calculation():
    metrics = _error_metrics(np.array([10.0, 10.0, 10.0]), np.array([8.0, 12.0, 10.0]))
    assert np.isclose(metrics["mae"], 4 / 3)
    assert np.isclose(metrics["wape"], 4 / 30)
    assert np.isclose(metrics["rmse"], np.sqrt(8 / 3))
    assert np.isclose(metrics["bias"], 0.0)


def test_global_models_fit_and_stay_non_negative():
    book = generate_dataset(start="2023-01-01", end="2023-04-30", ports=("Mombasa",), categories=("provisions",))
    frame = build_supervised(book.daily_demand, book.lineup, book.holidays, range(1, 4))
    train = frame.loc[frame["target_date"] <= "2023-04-01"]
    boost = GradientBoostModel(use_lineup=True, max_iter=15, max_leaf_nodes=15, min_samples_leaf=8).fit(train)
    ridge = FourierRidgeModel().fit(train)
    holdout = frame.loc[frame["target_date"] > "2023-04-01"]
    assert len(holdout) > 0
    assert np.all(boost.predict(holdout) >= 0)
    assert np.all(ridge.predict(holdout) >= 0)
    assert pd.Timestamp(train["target_date"].max()) <= pd.Timestamp("2023-04-01")
