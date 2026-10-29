"""Supervised rows for direct multi-horizon demand forecasts.

Every feature on a row dated ``target_date`` with a given ``horizon`` is
known on ``origin_date = target_date - horizon``. Demand lags and rolling
statistics end on the origin. The vessel lineup and the calendar describe
the target day, which is the information a chandler has once the port
lineup for that day is in hand.

``build_forecast_frame`` uses the same formulas as ``build_supervised``.
Tests check that the two agree on in-sample origins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY_FEATURES = (
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "roll_mean_7",
    "roll_mean_28",
    "roll_std_28",
    "seasonal_365",
)

LINEUP_FEATURES = (
    "n_calls",
    "total_crew",
    "total_dwt",
    "total_passengers",
    "n_container",
    "n_bulk",
    "n_tanker",
    "n_general_cargo",
    "n_roro",
    "n_passenger",
    "n_fishing",
)

CALENDAR_FEATURES = (
    "horizon",
    "dow",
    "month",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "is_public_holiday",
    "is_ramadan",
    "is_pre_eid",
    "is_eid",
    "is_christmas_week",
    "is_closed",
    "days_until_holiday",
)

KEY_COLUMNS = ("port", "category", "origin_date", "target_date", "horizon")
TARGET_COLUMN = "demand"


def build_supervised(
    demand: pd.DataFrame,
    lineup: pd.DataFrame,
    holidays: pd.DataFrame,
    horizons: range | tuple[int, ...] | list[int],
) -> pd.DataFrame:
    """Stack one block of rows per horizon.

    Rows without 28 days of history before the origin are dropped. The
    year-ago lag is left missing when the series is not long enough; tree
    models accept that, and the linear model imputes it.
    """

    blocks = [_history_block(demand, int(h)) for h in horizons]
    frame = pd.concat(blocks, ignore_index=True)
    frame = _attach_known_on_target(frame, lineup, holidays)
    frame = frame.dropna(subset=["lag_28", "roll_mean_28"]).reset_index(drop=True)
    return frame


def build_forecast_frame(
    demand: pd.DataFrame,
    lineup: pd.DataFrame,
    holidays: pd.DataFrame,
    origin: pd.Timestamp | str,
    horizon: int,
    ports: list[str] | tuple[str, ...] | None = None,
    categories: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Feature rows for days ``origin + 1`` ... ``origin + horizon``.

    ``demand`` must contain the history through ``origin`` for each series.
    Lineup and holiday rows must cover the target dates. Demand on the
    target dates is not read.
    """

    origin_ts = pd.Timestamp(origin)
    history = demand.loc[pd.to_datetime(demand["date"]) <= origin_ts].copy()
    pieces = []
    grouped = history.groupby(["port", "category"], sort=False)
    for (port, category), series in grouped:
        if ports is not None and port not in ports:
            continue
        if categories is not None and category not in categories:
            continue
        series = series.sort_values("date")
        if pd.Timestamp(series["date"].iloc[-1]) != origin_ts:
            raise ValueError(f"History for {port} / {category} does not end on {origin_ts.date()}")
        y = series["demand"].to_numpy(dtype=float)
        dates = pd.to_datetime(series["date"])
        pieces.append(_forecast_rows_for_series(y, dates, origin_ts, horizon, port, category))
    if not pieces:
        raise ValueError("No series matched the forecast request")
    frame = pd.concat(pieces, ignore_index=True)
    return _attach_known_on_target(frame, lineup, holidays)


def _history_block(demand: pd.DataFrame, horizon: int) -> pd.DataFrame:
    pieces = []
    for (port, category), series in demand.groupby(["port", "category"], sort=False):
        series = series.sort_values("date")
        y = series["demand"].to_numpy(dtype=float)
        dates = pd.to_datetime(series["date"])
        block = _features_for_horizon(y, dates, horizon)
        if block.empty:
            continue
        block["port"] = port
        block["category"] = category
        pieces.append(block)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def _features_for_horizon(y: np.ndarray, dates: pd.Series, horizon: int) -> pd.DataFrame:
    """Vectorised features for every target index that has a full lag-28 window."""

    n = len(y)
    # Target index i needs origin i - horizon >= 27.
    first = horizon + 27
    if first >= n:
        return pd.DataFrame()
    target_idx = np.arange(first, n)
    origin_idx = target_idx - horizon
    frame = _rows_from_indices(y, dates, target_idx, origin_idx, horizon)
    return frame


def _forecast_rows_for_series(
    y: np.ndarray,
    dates: pd.Series,
    origin: pd.Timestamp,
    horizon: int,
    port: str,
    category: str,
) -> pd.DataFrame:
    n = len(y)
    if n < 28:
        raise ValueError(f"Need 28 history points for {port} / {category}, found {n}")
    origin_idx = np.full(horizon, n - 1, dtype=int)
    # Virtual target indices sit past the end of history. Only seasonal_365
    # looks at the target index, and it is masked when it would read the future.
    target_idx = origin_idx + np.arange(1, horizon + 1)
    target_dates = pd.Series(origin + pd.to_timedelta(np.arange(1, horizon + 1), unit="D"))
    frame = _rows_from_indices(y, dates, target_idx, origin_idx, np.arange(1, horizon + 1), target_dates=target_dates)
    frame["port"] = port
    frame["category"] = category
    return frame


def _rows_from_indices(
    y: np.ndarray,
    dates: pd.Series,
    target_idx: np.ndarray,
    origin_idx: np.ndarray,
    horizon: int | np.ndarray,
    target_dates: pd.Series | None = None,
) -> pd.DataFrame:
    horizon_arr = np.full(len(target_idx), horizon, dtype=int) if np.isscalar(horizon) else np.asarray(horizon, dtype=int)

    def take(lookback: np.ndarray) -> np.ndarray:
        src = target_idx - lookback
        out = np.full(len(target_idx), np.nan)
        ok = (src >= 0) & (src < len(y))
        out[ok] = y[src[ok]]
        return out

    def rolling(window: int, std: bool = False) -> np.ndarray:
        end = origin_idx
        start = end - window + 1
        out = np.full(len(target_idx), np.nan)
        ok = start >= 0
        # Prefix sums on the raw history. Origin is always inside history.
        csum = np.cumsum(y)
        ends = end[ok]
        starts = start[ok]
        prev = np.where(starts > 0, csum[starts - 1], 0.0)
        total = csum[ends] - prev
        if not std:
            out[ok] = total / window
            return out
        csum2 = np.cumsum(y * y)
        prev2 = np.where(starts > 0, csum2[starts - 1], 0.0)
        second = csum2[ends] - prev2
        variance = np.maximum(second / window - (total / window) ** 2, 0.0)
        out[ok] = np.sqrt(variance)
        return out

    if target_dates is None:
        target_dates = pd.to_datetime(dates.iloc[target_idx].to_numpy())
    else:
        target_dates = pd.to_datetime(np.asarray(target_dates))
    origin_dates = target_dates - pd.to_timedelta(horizon_arr, unit="D")

    return pd.DataFrame(
        {
            "origin_date": origin_dates,
            "target_date": target_dates,
            "horizon": horizon_arr,
            # Lookback 0 reads the target day. Future targets (past the end of
            # history) stay missing; the backtest joins actuals afterwards.
            "demand": take(np.zeros(len(target_idx), dtype=int)),
            "lag_1": y[origin_idx],
            "lag_7": take(horizon_arr + 6),
            "lag_14": take(horizon_arr + 13),
            "lag_28": take(horizon_arr + 27),
            "roll_mean_7": rolling(7),
            "roll_mean_28": rolling(28),
            "roll_std_28": rolling(28, std=True),
            "seasonal_365": take(np.full(len(target_idx), 365)),
        }
    )


def _attach_known_on_target(frame: pd.DataFrame, lineup: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["target_date"] = pd.to_datetime(out["target_date"])
    out["origin_date"] = pd.to_datetime(out["origin_date"])
    line = lineup.copy()
    line["date"] = pd.to_datetime(line["date"])
    hol = holidays.copy()
    hol["date"] = pd.to_datetime(hol["date"])
    out = out.merge(line, left_on=["target_date", "port"], right_on=["date", "port"], how="left")
    out = out.drop(columns=["date"])
    holiday_cols = [
        "date",
        "port",
        "is_public_holiday",
        "is_ramadan",
        "is_pre_eid",
        "is_eid",
        "is_christmas_week",
        "is_closed",
        "days_until_holiday",
    ]
    out = out.merge(hol[holiday_cols], left_on=["target_date", "port"], right_on=["date", "port"], how="left")
    out = out.drop(columns=["date"])
    for column in LINEUP_FEATURES:
        out[column] = out[column].fillna(0.0)
    doy = out["target_date"].dt.day_of_year.to_numpy()
    dow = out["target_date"].dt.day_of_week.to_numpy()
    out["dow"] = dow
    out["month"] = out["target_date"].dt.month.to_numpy()
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for column in ("is_public_holiday", "is_ramadan", "is_pre_eid", "is_eid", "is_christmas_week", "is_closed"):
        out[column] = out[column].fillna(False).astype(int)
    out["days_until_holiday"] = out["days_until_holiday"].fillna(21).astype(int)
    return out


def model_feature_columns(use_lineup: bool) -> list[str]:
    """Column order passed to the linear and tree models."""

    columns = ["port", "category", *HISTORY_FEATURES, *CALENDAR_FEATURES]
    if use_lineup:
        columns.extend(LINEUP_FEATURES)
    return columns
