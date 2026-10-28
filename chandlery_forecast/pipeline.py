"""Generate, score, and save the demo artefacts.

The entry point used by ``make train``. It refits every model, writes the
metrics the README quotes, and stores the lineup gradient-boosting model
the Streamlit app loads.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd

from chandlery_forecast import __version__
from chandlery_forecast.backtest import run_backtest
from chandlery_forecast.config import (
    BACKTEST_HORIZONS,
    BACKTEST_STEP_DAYS,
    END_DATE,
    GENERATOR_VERSION,
    MAX_HORIZON,
    MODEL_LABELS,
    PRIMARY_HORIZON,
    SEED,
    TRAIN_CUTOFF,
)
from chandlery_forecast.features import build_supervised
from chandlery_forecast.generate import generate_dataset, load_book, save_book
from chandlery_forecast.models import permutation_importance
from chandlery_forecast.paths import BUNDLE_PATH, DATA_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from chandlery_forecast.plotting import save_all


def generate(seed: int = SEED) -> None:
    """Write a fresh synthetic book. Overwrites ``data/synthetic``."""

    book = generate_dataset(seed=seed, start="2022-01-01", end=END_DATE)
    save_book(book, DATA_DIR)
    print(
        f"Wrote {book.metadata['n_calls']:,} calls and "
        f"{book.metadata['n_demand_rows']:,} demand rows to {DATA_DIR}"
    )


def train_and_evaluate() -> dict:
    """Backtest the book on disk and refresh reports, figures, and the model bundle."""

    book = load_book(DATA_DIR)
    print(
        f"Loaded synthetic book {book.metadata.get('start')} to {book.metadata.get('end')} "
        f"(seed {book.metadata.get('seed')}). Fitting through {TRAIN_CUTOFF}."
    )
    result = run_backtest(
        demand=book.daily_demand,
        lineup=book.lineup,
        holidays=book.holidays,
        cutoff=TRAIN_CUTOFF,
        max_horizon=max(BACKTEST_HORIZONS),
        train_horizon=MAX_HORIZON,
        step_days=BACKTEST_STEP_DAYS,
        score_horizons=BACKTEST_HORIZONS,
        include_holt_winters=True,
    )
    importance = _importance(result.models["hgb_lineup"], book, result.cutoff)
    headline = _headline(result.metrics)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(REPORTS_DIR / "predictions.csv", index=False)
    result.metrics.to_csv(REPORTS_DIR / "metrics.csv", index=False)
    result.metrics_by_category.to_csv(REPORTS_DIR / "metrics_by_category.csv", index=False)
    importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    (REPORTS_DIR / "headline.json").write_text(json.dumps(headline, indent=2) + "\n")

    save_all(
        daily=book.daily_demand,
        calls=book.vessel_calls,
        lineup=book.lineup,
        predictions=result.predictions,
        metrics=result.metrics,
        by_category=result.metrics_by_category,
        importance=importance,
        residual_std=result.residual_std,
        directory=FIGURES_DIR,
    )

    bundle = {
        "disclaimer": "Synthetic demonstration model. Not fit on client or port data.",
        "package_version": __version__,
        "generator_version": GENERATOR_VERSION,
        "seed": book.metadata.get("seed", SEED),
        "cutoff": str(result.cutoff.date()),
        "data_end": str(pd.to_datetime(book.daily_demand["date"]).max().date()),
        "model": result.models["hgb_lineup"],
        "residual_std": result.residual_std,
        "inflation": result.inflation,
        "headline": headline,
        "origins": [str(day.date()) for day in result.origins],
    }
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    _print_metrics(result.metrics)
    print(f"Saved model bundle to {BUNDLE_PATH}")
    return headline


def _importance(model, book, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = build_supervised(book.daily_demand, book.lineup, book.holidays, [PRIMARY_HORIZON])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    window_start = cutoff - pd.Timedelta(days=120)
    validation = frame.loc[(frame["target_date"] <= cutoff) & (frame["target_date"] >= window_start)]
    return permutation_importance(model, validation, n_repeats=4, max_rows=2000)


def _headline(metrics: pd.DataFrame) -> dict:
    part = metrics.loc[(metrics["horizon"] == PRIMARY_HORIZON)].copy()
    rows = []
    for _, row in part.iterrows():
        rows.append(
            {
                "model": row["model"],
                "label": MODEL_LABELS.get(row["model"], row["model"]),
                "level": row["level"],
                "mae": round(float(row["mae"]), 2),
                "rmse": round(float(row["rmse"]), 2),
                "wape": round(float(row["wape"]), 4),
                "mase": None if pd.isna(row["mase"]) else round(float(row["mase"]), 3),
                "bias": round(float(row["bias"]), 2),
                "n": int(row["n"]),
            }
        )
    return {
        "disclaimer": "Metrics are errors on the synthetic holdout, not a promise about a live port.",
        "cutoff": TRAIN_CUTOFF,
        "primary_horizon_days": PRIMARY_HORIZON,
        "rows": rows,
    }


def _print_metrics(metrics: pd.DataFrame) -> None:
    view = metrics.loc[metrics["horizon"] == PRIMARY_HORIZON, ["model", "level", "mae", "rmse", "wape", "mase", "n"]]
    print(view.to_string(index=False))
