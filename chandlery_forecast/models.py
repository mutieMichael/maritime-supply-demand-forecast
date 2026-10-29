"""Forecasting models for daily chandlery demand.

The seasonal naive and Holt-Winters models see only the demand history of
one series. The ridge and gradient-boosting models are global: one fit
covers every port and category, on the log1p of demand, with the horizon
as a feature (direct multi-step, not recursive).

The lineup model is the one the app orders against. The others exist to
show what that lineup is worth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from chandlery_forecast.config import HGB_PARAMS, RIDGE_ALPHA, SEED
from chandlery_forecast.features import LINEUP_FEATURES, model_feature_columns

_CATEGORICAL = ("port", "category")


def seasonal_naive_forecast(history: np.ndarray, horizon: int, period: int = 7) -> np.ndarray:
    """Repeat the last complete season, appending each prediction as it is made.

    ``history`` ends on the origin date. For a weekly period the first
    forecast day copies the value seven days before the target, which is
    inside the history. Later days copy the earlier predictions once the
    lookback would cross the origin.
    """

    values = [float(v) for v in np.asarray(history, dtype=float)]
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not values:
        return np.zeros(horizon)
    if len(values) < period:
        return np.full(horizon, max(values[-1], 0.0))
    out = np.empty(horizon, dtype=float)
    for step in range(horizon):
        nxt = values[-period]
        out[step] = nxt
        values.append(nxt)
    return np.maximum(out, 0.0)


def holt_winters_forecast(history: np.ndarray, horizon: int, period: int = 7) -> np.ndarray:
    """Additive weekly Holt-Winters, clipped at zero.

    Falls back to the seasonal naive if the series is too short, almost
    flat, or the smoother fails to converge. Imports statsmodels lazily so
    importing this module stays cheap in tests that never call it.
    """

    y = np.asarray(history, dtype=float)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if len(y) < period * 3 or float(np.nanstd(y)) < 1e-8:
        return seasonal_naive_forecast(y, horizon, period=period)
    try:
        import warnings

        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                y,
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=period,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True, remove_bias=True)
            forecast = np.asarray(fit.forecast(horizon), dtype=float)
    except Exception:
        return seasonal_naive_forecast(y, horizon, period=period)
    return np.maximum(forecast, 0.0)


class _LogDemandModel:
    """Shared fit/predict for models trained on log1p demand."""

    name = "base"
    use_lineup = False

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.feature_columns: list[str] = model_feature_columns(self.use_lineup)
        self.port_levels: list[str] = []
        self.category_levels: list[str] = []
        # Numeric columns with fewer than two observed values. Histogram
        # boosting cannot bin them, and a linear model cannot scale them.
        self.constant_columns_: list[str] = []
        self.smear_: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame) -> _LogDemandModel:
        """Fit on log demand, then rescale so predictions match the mean.

        Squared error on ``log1p`` tracks the median of a skewed order book
        and comes out low. A single multiplier per category, estimated on the
        last 90 days of the training frame (or in-sample, on a short frame),
        puts the order quantity back on the mean. The multiplier is applied
        after a refit on the whole frame.
        """

        frame = frame.reset_index(drop=True)
        self.smear_ = {}
        if "target_date" in frame.columns:
            dates = pd.to_datetime(frame["target_date"])
            cal_start = dates.max() - pd.Timedelta(days=90)
            early = frame.loc[dates < cal_start]
            late = frame.loc[dates >= cal_start]
            if len(early) >= 800 and len(late) >= 150:
                self._fit_estimator(early)
                self.smear_ = self._smear_factors(late)
        self._fit_estimator(frame)
        if not self.smear_:
            self.smear_ = self._smear_factors(frame)
        return self

    def _fit_estimator(self, frame: pd.DataFrame) -> None:
        self.port_levels = sorted(frame["port"].astype(str).unique())
        self.category_levels = sorted(frame["category"].astype(str).unique())
        self.constant_columns_ = []
        design = self.design_matrix(frame)
        self.constant_columns_ = [
            column
            for column in design.columns
            if column not in _CATEGORICAL and design[column].nunique(dropna=True) < 2
        ]
        if self.constant_columns_:
            design = design.drop(columns=self.constant_columns_)
        target = np.log1p(frame["demand"].to_numpy(dtype=float))
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(design, target)

    def _raw_predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name} has not been fit")
        raw = self.pipeline.predict(self.design_matrix(frame))
        return np.maximum(np.expm1(np.asarray(raw, dtype=float)), 0.0)

    def _smear_factors(self, frame: pd.DataFrame) -> dict[str, float]:
        predicted = self._raw_predict(frame)
        actual = frame["demand"].to_numpy(dtype=float)
        categories = frame["category"].astype(str).to_numpy()
        factors: dict[str, float] = {}
        for category in sorted(set(categories)):
            mask = categories == category
            predicted_sum = float(predicted[mask].sum())
            actual_sum = float(actual[mask].sum())
            factor = actual_sum / predicted_sum if predicted_sum > 0 else 1.0
            factors[category] = float(np.clip(factor, 0.75, 1.8))
        return factors

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self._raw_predict(frame)
        if not getattr(self, "smear_", None):
            return raw
        factors = frame["category"].astype(str).map(self.smear_).fillna(1.0).to_numpy(dtype=float)
        return raw * factors

    def design_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Feature frame is missing columns: {missing}")
        design = frame.loc[:, self.feature_columns].copy()
        design["port"] = pd.Categorical(design["port"].astype(str), categories=self.port_levels or None)
        design["category"] = pd.Categorical(design["category"].astype(str), categories=self.category_levels or None)
        if self.constant_columns_:
            design = design.drop(columns=[column for column in self.constant_columns_ if column in design.columns])
        return design

    def _build_pipeline(self) -> Pipeline:
        raise NotImplementedError


class FourierRidgeModel(_LogDemandModel):
    """Linear model on lags, rolling stats, and calendar Fourier terms.

    Port and category are one-hot encoded. Vessel-lineup columns are not
    included, so this is the classical baseline the lineup model has to beat.
    """

    name = "fourier_ridge"
    use_lineup = False

    def __init__(self, alpha: float = RIDGE_ALPHA) -> None:
        super().__init__()
        self.alpha = alpha

    def design_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        design = super().design_matrix(frame)
        # One-hot encoding wants plain strings; the booster keeps categoricals.
        design["port"] = design["port"].astype(str)
        design["category"] = design["category"].astype(str)
        return design

    def _build_pipeline(self) -> Pipeline:
        numeric = [
            column
            for column in self.feature_columns
            if column not in _CATEGORICAL and column not in self.constant_columns_
        ]
        transform = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    list(_CATEGORICAL),
                ),
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric,
                ),
            ]
        )
        return Pipeline(
            steps=[
                ("transform", transform),
                ("model", Ridge(alpha=self.alpha)),
            ]
        )


class GradientBoostModel(_LogDemandModel):
    """Histogram gradient boosting on log1p demand.

    Set ``use_lineup=False`` to measure the same model without the vessel
    particulars. Missing year-ago lags are left missing; the booster splits
    on missingness directly. Early stopping is off so the fit does not draw
    a random validation slice out of the future.
    """

    def __init__(self, use_lineup: bool = True, **hgb_params) -> None:
        self.use_lineup = use_lineup
        self.name = "hgb_lineup" if use_lineup else "hgb_history"
        super().__init__()
        params = dict(HGB_PARAMS)
        params.update(hgb_params)
        params["early_stopping"] = False
        self.hgb_params = params

    def _build_pipeline(self) -> Pipeline:
        model = HistGradientBoostingRegressor(**self.hgb_params)
        # A one-step pipeline keeps joblib payloads uniform with the ridge model.
        return Pipeline(steps=[("model", model)])

    def design_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        design = super().design_matrix(frame)
        # HistGradientBoosting treats pandas categoricals as categorical when
        # categorical_features="from_dtype" (the library default).
        return design


def lineup_feature_names() -> tuple[str, ...]:
    return LINEUP_FEATURES


def permutation_importance(
    model: _LogDemandModel,
    frame: pd.DataFrame,
    n_repeats: int = 4,
    max_rows: int = 2500,
    seed: int = SEED,
) -> pd.DataFrame:
    """Increase in MAE when each feature is shuffled.

    Shuffling is done on the modelling frame, then passed through the same
    design matrix as training, so port and category stay categorical.
    """

    if frame.empty:
        return pd.DataFrame(columns=["feature", "mae_increase"])
    sample = frame
    if len(sample) > max_rows:
        sample = sample.sample(max_rows, random_state=seed)
    y = sample["demand"].to_numpy(dtype=float)
    baseline = float(np.mean(np.abs(y - model.predict(sample))))
    rng = np.random.default_rng(seed)
    rows = []
    for column in model.feature_columns:
        increases = []
        for _ in range(n_repeats):
            shuffled = sample.copy()
            values = shuffled[column].to_numpy().copy()
            rng.shuffle(values)
            shuffled[column] = values
            bumped = float(np.mean(np.abs(y - model.predict(shuffled))))
            increases.append(bumped - baseline)
        rows.append({"feature": column, "mae_increase": float(np.mean(increases))})
    ranked = pd.DataFrame(rows).sort_values("mae_increase", ascending=False).reset_index(drop=True)
    return ranked
