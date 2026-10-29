"""The rolling origin does not train on the days it scores."""

import pandas as pd

from chandlery_forecast.backtest import backtest_origins, run_backtest
from chandlery_forecast.generate import generate_dataset


def test_origins_stop_before_the_horizon_falls_off_the_end():
    origins = backtest_origins("2024-01-01", "2024-03-01", step_days=14, max_horizon=14)
    assert origins[0] == pd.Timestamp("2024-01-01")
    assert origins[-1] + pd.Timedelta(days=14) <= pd.Timestamp("2024-03-01")
    assert origins[1] - origins[0] == pd.Timedelta(days=14)


def test_backtest_scores_only_dates_after_the_cutoff():
    book = generate_dataset(
        start="2023-01-01",
        end="2023-07-31",
        ports=("Mombasa",),
        categories=("provisions",),
    )
    result = run_backtest(
        book.daily_demand,
        book.lineup,
        book.holidays,
        cutoff="2023-06-01",
        max_horizon=7,
        train_horizon=7,
        step_days=14,
        score_horizons=(7,),
        include_holt_winters=False,
        hgb_params={"max_iter": 20, "max_leaf_nodes": 15, "min_samples_leaf": 10},
    )
    assert result.predictions["target_date"].min() > pd.Timestamp("2023-06-01")
    assert set(result.predictions["model"]) == {"seasonal_naive", "fourier_ridge", "hgb_history", "hgb_lineup"}
    assert (result.predictions["yhat"] >= 0).all()
    assert "hgb_lineup" in result.models
    scored = result.metrics.loc[result.metrics["level"] == "horizon_total", "wape"]
    assert scored.notna().all()
    assert (scored >= 0).all()
