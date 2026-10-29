"""Rolling-origin backtest.

The supervised models are fit once, on rows whose target date is on or
before the cutoff. Each later origin forecasts the next days from the
information set available that morning: demand observed through the origin,
plus the lineup and calendar of the target days. Holt-Winters and the
seasonal naive are refit from the history of each series at each origin.

That is a fixed-model rolling origin for the regression models and a
refit-each-origin evaluation for the univariate smoothers. Both only score
dates after the cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from chandlery_forecast.config import PRIMARY_HORIZON, SEED, UNIT_PRICES_USD
from chandlery_forecast.features import build_supervised
from chandlery_forecast.models import (
    FourierRidgeModel,
    GradientBoostModel,
    holt_winters_forecast,
    seasonal_naive_forecast,
)


@dataclass
class BacktestResult:
    """Predictions, metrics, and the fitted global models."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    metrics_by_category: pd.DataFrame
    residual_std: dict[str, dict[int, float]]
    inflation: dict[str, float]
    origins: list[pd.Timestamp]
    cutoff: pd.Timestamp
    models: dict = field(default_factory=dict)


def backtest_origins(
    cutoff: pd.Timestamp | str,
    data_end: pd.Timestamp | str,
    step_days: int,
    max_horizon: int,
) -> list[pd.Timestamp]:
    """Origins on a fixed stride, starting at the cutoff.

    The origin is the last observed day. The longest scored target
    (``origin + max_horizon``) has to fall inside the extract.
    """

    if step_days < 1 or max_horizon < 1:
        raise ValueError("step_days and max_horizon must be positive")
    day = pd.Timestamp(cutoff)
    last = pd.Timestamp(data_end) - pd.Timedelta(days=max_horizon)
    origins: list[pd.Timestamp] = []
    while day <= last:
        origins.append(day)
        day += pd.Timedelta(days=step_days)
    if not origins:
        raise ValueError("No backtest origins: the series is shorter than the horizon after the cutoff")
    return origins


def run_backtest(
    demand: pd.DataFrame,
    lineup: pd.DataFrame,
    holidays: pd.DataFrame,
    cutoff: str,
    max_horizon: int = PRIMARY_HORIZON,
    train_horizon: int | None = None,
    step_days: int = 14,
    score_horizons: tuple[int, ...] = (7, 14, 28),
    include_holt_winters: bool = True,
    hgb_params: dict | None = None,
) -> BacktestResult:
    """Fit on ``target_date <= cutoff`` and score later rolling origins.

    ``train_horizon`` can be longer than ``max_horizon`` so the saved model
    knows how to forecast a lead time beyond the scored window. Evaluation
    itself never looks past ``max_horizon``.
    """

    cutoff_ts = pd.Timestamp(cutoff)
    data_end = pd.to_datetime(demand["date"]).max()
    fit_horizon = train_horizon or max_horizon
    if fit_horizon < max_horizon:
        raise ValueError("train_horizon must be at least max_horizon")

    frame = build_supervised(demand, lineup, holidays, range(1, fit_horizon + 1))
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    frame["origin_date"] = pd.to_datetime(frame["origin_date"])
    train = frame.loc[frame["target_date"] <= cutoff_ts].copy()
    if train.empty:
        raise ValueError("Training frame is empty; check the cutoff and history length")
    if train["target_date"].max() > cutoff_ts:
        raise AssertionError("Training rows leaked past the cutoff")

    hgb_kwargs = dict(hgb_params or {})
    hgb_kwargs.setdefault("random_state", SEED)
    models = {
        "fourier_ridge": FourierRidgeModel().fit(train),
        "hgb_history": GradientBoostModel(use_lineup=False, **hgb_kwargs).fit(train),
        "hgb_lineup": GradientBoostModel(use_lineup=True, **hgb_kwargs).fit(train),
    }

    origins = backtest_origins(cutoff_ts, data_end, step_days, max_horizon)
    origin_set = set(origins)
    test = frame.loc[frame["origin_date"].isin(origin_set) & (frame["horizon"] <= max_horizon)].copy()
    if test.empty:
        raise ValueError("No test rows for the chosen origins")

    pieces = []
    for name, model in models.items():
        piece = test.loc[:, ["port", "category", "origin_date", "target_date", "horizon", "demand"]].copy()
        piece = piece.rename(columns={"demand": "y"})
        piece["yhat"] = model.predict(test)
        piece["model"] = name
        pieces.append(piece)

    pieces.append(_univariate_predictions(demand, origins, max_horizon, "seasonal_naive", seasonal_naive_forecast))
    if include_holt_winters:
        pieces.append(_univariate_predictions(demand, origins, max_horizon, "holt_winters", holt_winters_forecast))

    predictions = pd.concat(pieces, ignore_index=True)
    predictions = predictions.dropna(subset=["y", "yhat"]).reset_index(drop=True)

    scales = _seasonal_scale(demand, cutoff_ts)
    scored = tuple(h for h in score_horizons if h <= max_horizon)
    metrics = _summarise(predictions, scored, scales)
    by_category = _summarise_by_category(predictions, scored)
    residual_std, inflation = _residual_scales(predictions, category_key="hgb_lineup")
    return BacktestResult(
        predictions=predictions,
        metrics=metrics,
        metrics_by_category=by_category,
        residual_std=residual_std,
        inflation=inflation,
        origins=origins,
        cutoff=cutoff_ts,
        models=models,
    )


def _univariate_predictions(demand, origins, horizon, name, forecaster) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = {
        (port, category): series.sort_values("date")
        for (port, category), series in demand.groupby(["port", "category"], sort=False)
    }
    for origin in origins:
        target_dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
        for (port, category), series in grouped.items():
            history = series.loc[series["date"] <= origin, "demand"].to_numpy(dtype=float)
            if len(history) < 28:
                continue
            forecast = forecaster(history, horizon)
            actual = series.set_index("date")["demand"]
            for step, day in enumerate(target_dates, start=1):
                if day not in actual.index:
                    continue
                rows.append(
                    {
                        "port": port,
                        "category": category,
                        "origin_date": origin,
                        "target_date": day,
                        "horizon": step,
                        "y": float(actual.loc[day]),
                        "yhat": float(forecast[step - 1]),
                        "model": name,
                    }
                )
    return pd.DataFrame(rows)


def _seasonal_scale(demand: pd.DataFrame, cutoff: pd.Timestamp, period: int = 7) -> dict[tuple[str, str], float]:
    """In-sample seasonal-naive MAE, the denominator of MASE (Hyndman)."""

    scales: dict[tuple[str, str], float] = {}
    for (port, category), series in demand.groupby(["port", "category"], sort=False):
        y = series.loc[series["date"] <= cutoff].sort_values("date")["demand"].to_numpy(dtype=float)
        if len(y) <= period:
            scales[(port, category)] = np.nan
            continue
        scales[(port, category)] = float(np.mean(np.abs(y[period:] - y[:-period])))
    return scales


def _summarise(predictions: pd.DataFrame, horizons: tuple[int, ...], scales: dict) -> pd.DataFrame:
    rows = []
    for model, model_df in predictions.groupby("model"):
        for horizon in horizons:
            part = model_df.loc[model_df["horizon"] <= horizon]
            if part.empty:
                continue
            # Errors are scored in synthetic USD so kilograms are not added to litres.
            daily = _error_metrics(*_valued(part))
            daily.update(
                {
                    "model": model,
                    "horizon": horizon,
                    "level": "daily",
                    "mase": _mase(part, scales),
                    "n": int(len(part)),
                }
            )
            rows.append(daily)
            totals = part.groupby(["port", "category", "origin_date"], as_index=False).agg(y=("y", "sum"), yhat=("yhat", "sum"))
            total_metrics = _error_metrics(*_valued(totals))
            total_metrics.update(
                {
                    "model": model,
                    "horizon": horizon,
                    "level": "horizon_total",
                    "mase": np.nan,
                    "n": int(len(totals)),
                }
            )
            rows.append(total_metrics)
    metrics = pd.DataFrame(rows)
    return metrics.sort_values(["level", "horizon", "wape"]).reset_index(drop=True)


def _summarise_by_category(predictions: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for (model, category), model_df in predictions.groupby(["model", "category"]):
        for horizon in horizons:
            part = model_df.loc[model_df["horizon"] <= horizon]
            if part.empty:
                continue
            metrics = _error_metrics(part["y"].to_numpy(), part["yhat"].to_numpy())
            totals = part.groupby(["port", "origin_date"], as_index=False).agg(y=("y", "sum"), yhat=("yhat", "sum"))
            total_metrics = _error_metrics(totals["y"].to_numpy(), totals["yhat"].to_numpy())
            rows.append(
                {
                    "model": model,
                    "category": category,
                    "horizon": horizon,
                    "daily_wape": metrics["wape"],
                    "daily_mae": metrics["mae"],
                    "total_wape": total_metrics["wape"],
                    "total_mae": total_metrics["mae"],
                    "n_days": int(len(part)),
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon", "category", "total_wape"]).reset_index(drop=True)


def _valued(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Scale quantities by the synthetic basket price for their category."""

    price = frame["category"].map(UNIT_PRICES_USD).to_numpy(dtype=float)
    return frame["y"].to_numpy(dtype=float) * price, frame["yhat"].to_numpy(dtype=float) * price


def _error_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    error = np.asarray(yhat, dtype=float) - np.asarray(y, dtype=float)
    abs_error = np.abs(error)
    denom = float(np.sum(np.abs(y)))
    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": float(np.sum(abs_error) / denom) if denom > 0 else float("nan"),
        "bias": float(np.mean(error)),
    }


def _mase(part: pd.DataFrame, scales: dict[tuple[str, str], float]) -> float:
    weighted = 0.0
    weight_sum = 0.0
    for (port, category), group in part.groupby(["port", "category"]):
        scale = scales.get((port, category), np.nan)
        if not np.isfinite(scale) or scale <= 0:
            continue
        mae = float(np.mean(np.abs(group["y"] - group["yhat"])))
        price = float(UNIT_PRICES_USD[category])
        weight = float(np.sum(np.abs(group["y"]))) * price
        weighted += weight * (mae / scale)
        weight_sum += weight
    if weight_sum == 0:
        return float("nan")
    return weighted / weight_sum


def _residual_scales(predictions: pd.DataFrame, category_key: str) -> tuple[dict[str, dict[int, float]], dict[str, float]]:
    """Out-of-sample residual std by category and step, plus a 14-day inflation factor."""

    part = predictions.loc[predictions["model"] == category_key].copy()
    residual_std: dict[str, dict[int, float]] = {}
    inflation: dict[str, float] = {}
    if part.empty:
        return residual_std, inflation
    for category, group in part.groupby("category"):
        by_step: dict[int, float] = {}
        for step, step_df in group.groupby("horizon"):
            by_step[int(step)] = float(np.std(step_df["y"].to_numpy() - step_df["yhat"].to_numpy(), ddof=1))
        residual_std[str(category)] = by_step
        window = group.loc[group["horizon"] <= 14]
        if window.empty or not by_step:
            inflation[str(category)] = 1.0
            continue
        totals = window.groupby(["port", "origin_date"], as_index=False).agg(y=("y", "sum"), yhat=("yhat", "sum"))
        empirical = float(np.std(totals["y"] - totals["yhat"], ddof=1))
        independent = float(np.sqrt(np.sum([by_step[h] ** 2 for h in by_step if h <= 14])))
        ratio = empirical / independent if independent > 0 else 1.0
        # Floor at 1 so the safety stock never pretends daily errors cancel.
        inflation[str(category)] = max(1.0, ratio)
    return residual_std, inflation
