"""Feature timing: lags end on the origin, lineup describes the target day."""

import numpy as np
import pandas as pd

from chandlery_forecast.features import build_forecast_frame, build_supervised
from chandlery_forecast.generate import generate_dataset


def _toy_book():
    dates = pd.date_range("2023-01-01", periods=80, freq="D")
    demand = pd.DataFrame(
        {
            "date": dates,
            "port": "Mombasa",
            "category": "provisions",
            "demand": np.arange(len(dates), dtype=float),
        }
    )
    lineup = pd.DataFrame({"date": dates, "port": "Mombasa", "n_calls": 0.0, "total_crew": 0.0, "total_dwt": 0.0, "total_passengers": 0.0, "n_container": 0.0, "n_bulk": 0.0, "n_tanker": 0.0, "n_general_cargo": 0.0, "n_roro": 0.0, "n_passenger": 0.0, "n_fishing": 0.0})
    holidays = pd.DataFrame(
        {
            "date": dates,
            "port": "Mombasa",
            "is_public_holiday": False,
            "is_ramadan": False,
            "is_pre_eid": False,
            "is_eid": False,
            "is_christmas_week": False,
            "is_closed": dates.day_of_week == 6,
            "days_until_holiday": 7,
        }
    )
    return demand, lineup, holidays


def test_lag_is_measured_from_the_origin():
    demand, lineup, holidays = _toy_book()
    frame = build_supervised(demand, lineup, holidays, [1, 7])
    step = frame.loc[frame["horizon"] == 1]
    # demand[t] = t index. lag_1 is demand on the origin, which is the previous day.
    assert np.allclose(step["lag_1"], step["demand"] - 1)
    week = frame.loc[frame["horizon"] == 7]
    assert np.allclose(week["lag_1"], week["demand"] - 7)


def test_forecast_frame_matches_supervised_features():
    book = generate_dataset(start="2023-01-01", end="2023-05-31", ports=("Mombasa",), categories=("provisions", "lubricants"))
    frame = build_supervised(book.daily_demand, book.lineup, book.holidays, range(1, 6))
    origin = pd.Timestamp("2023-04-20")
    forecast = build_forecast_frame(book.daily_demand, book.lineup, book.holidays, origin, 5, ports=["Mombasa"], categories=["lubricants"])
    supervised = frame.loc[(frame["port"] == "Mombasa") & (frame["category"] == "lubricants") & (frame["origin_date"] == origin)].sort_values("horizon")
    forecast = forecast.sort_values("horizon")
    columns = ["lag_1", "lag_7", "lag_28", "roll_mean_7", "roll_mean_28", "n_calls", "total_crew", "dow", "month", "is_closed"]
    for column in columns:
        assert np.allclose(supervised[column].to_numpy(dtype=float), forecast[column].to_numpy(dtype=float), equal_nan=True)


def test_future_spike_does_not_change_past_features():
    demand, lineup, holidays = _toy_book()
    original = build_supervised(demand, lineup, holidays, [1])
    spiked = demand.copy()
    spiked.loc[spiked["date"] == "2023-03-15", "demand"] = 10_000
    changed = build_supervised(spiked, lineup, holidays, [1])
    cutoff = pd.Timestamp("2023-03-01")
    left = original.loc[original["target_date"] <= cutoff].reset_index(drop=True)
    right = changed.loc[changed["target_date"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_lineup_feature_is_the_target_day():
    demand, lineup, holidays = _toy_book()
    target = pd.Timestamp("2023-03-10")
    lineup.loc[lineup["date"] == target, "n_calls"] = 7
    frame = build_supervised(demand, lineup, holidays, [1])
    row = frame.loc[frame["target_date"] == target].iloc[0]
    assert row["n_calls"] == 7
    assert row["origin_date"] == target - pd.Timedelta(days=1)
