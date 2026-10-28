"""Static figures for the README and the notebook.

Charts are drawn from the synthetic order book and the backtest. Titles
describe the picture; they do not claim a real port behaved this way.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chandlery_forecast.config import CATEGORY_LABELS, MODEL_LABELS

NAVY = "#0B3A4A"
TEAL = "#1F7A8C"
SAND = "#C4A36A"
CORAL = "#D4644A"
INK = "#1C2830"
SLATE = "#5C6B73"
GREEN = "#2E7D5B"
FOG = "#F4F7F8"

MODEL_COLORS = {
    "seasonal_naive": "#8B979F",
    "holt_winters": "#C4A36A",
    "fourier_ridge": "#2A9D8F",
    "hgb_history": "#3D5A80",
    "hgb_lineup": NAVY,
}

CATEGORY_COLORS = {
    "provisions": NAVY,
    "fresh_water": TEAL,
    "lubricants": SAND,
    "deck_engine_stores": "#3D5A80",
    "safety_equipment": CORAL,
}


def save_all(daily, calls, lineup, predictions, metrics, by_category, importance, residual_std, directory: Path) -> list[Path]:
    """Write the figure set and return the paths."""

    directory.mkdir(parents=True, exist_ok=True)
    writers = [
        ("demand_timeline.png", lambda path: plot_demand_timeline(daily, path)),
        ("weekday_booking_shift.png", lambda path: plot_weekday_shift(calls, path)),
        ("monthly_profile.png", lambda path: plot_monthly_profile(daily, path)),
        ("lineup_vs_provisions.png", lambda path: plot_lineup_scatter(daily, lineup, path)),
        ("backtest_overlay.png", lambda path: plot_backtest_overlay(daily, predictions, path)),
        ("model_comparison.png", lambda path: plot_model_comparison(metrics, path)),
        ("wape_by_category.png", lambda path: plot_wape_by_category(by_category, path)),
        ("feature_importance.png", lambda path: plot_importance(importance, path)),
        ("replenishment_example.png", lambda path: plot_replenishment_example(daily, predictions, residual_std, path)),
    ]
    paths = []
    for name, writer in writers:
        path = directory / name
        writer(path)
        paths.append(path)
        plt.close("all")
    return paths


def plot_demand_timeline(daily: pd.DataFrame, path: Path) -> None:
    series = _series(daily, "Mombasa", "provisions")
    end = series["date"].max()
    window = series.loc[series["date"] >= end - pd.Timedelta(days=540)].copy()
    window["roll"] = window["demand"].rolling(14, min_periods=7).mean()
    fig, ax = plt.subplots(figsize=(11.2, 4.4))
    ax.plot(window["date"], window["demand"], color="#B7C4C9", lw=0.8, label="Daily demand")
    ax.plot(window["date"], window["roll"], color=NAVY, lw=2.0, label="14-day mean")
    _finish(ax, "Mombasa provisions, synthetic book", "A weekly pulse with a slower seasonal swing", "Kilograms")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, path)


def plot_weekday_shift(calls: pd.DataFrame, path: Path) -> None:
    frame = calls.loc[calls["port"] == "Mombasa"].copy()
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    arrivals = frame["arrival_date"].dt.day_of_week.value_counts().reindex(range(7), fill_value=0)
    bookings = frame["order_date"].dt.day_of_week.value_counts().reindex(range(7), fill_value=0)
    # Bookings that spill past the extract are absent; compare shares.
    arrival_share = arrivals / arrivals.sum()
    booking_share = bookings / max(bookings.sum(), 1)
    x = np.arange(7)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(x - 0.18, arrival_share, width=0.36, color=SAND, label="Arrivals")
    ax.bar(x + 0.18, booking_share, width=0.36, color=NAVY, label="Booked onto the desk")
    ax.set_xticks(x, order)
    _finish(ax, "Sunday arrivals are booked on Monday", "Mombasa synthetic calls, share of the week", "Share of calls")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_monthly_profile(daily: pd.DataFrame, path: Path) -> None:
    frame = daily.copy()
    frame["month"] = frame["date"].dt.month
    means = frame.groupby(["category", "month"], as_index=False)["demand"].mean()
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for category, group in means.groupby("category"):
        ordered = group.sort_values("month")
        level = float(ordered["demand"].mean())
        ax.plot(
            ordered["month"],
            ordered["demand"] / level,
            marker="o",
            ms=4,
            lw=1.8,
            color=CATEGORY_COLORS[category],
            label=CATEGORY_LABELS[category],
        )
    ax.axhline(1.0, color=SLATE, lw=0.8, ls="--")
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    _finish(ax, "Each category has its own season", "Mean demand by month, indexed to 1, all synthetic ports", "Index")
    ax.legend(frameon=False, ncol=2)
    _save(fig, path)


def plot_lineup_scatter(daily: pd.DataFrame, lineup: pd.DataFrame, path: Path) -> None:
    demand = _series(daily, "Mombasa", "provisions")
    line = lineup.loc[lineup["port"] == "Mombasa", ["date", "total_crew", "n_calls"]]
    merged = demand.merge(line, on="date", how="inner")
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.scatter(merged["total_crew"], merged["demand"], s=12, alpha=0.35, c=TEAL, linewidths=0)
    corr = merged["total_crew"].corr(merged["demand"])
    _finish(ax, "Provisions move with the crew on the lineup", f"Mombasa synthetic days, correlation {corr:.2f}", "Kilograms")
    ax.set_xlabel("Total crew booked that day")
    _save(fig, path)


def plot_backtest_overlay(daily: pd.DataFrame, predictions: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.6), sharex=True)
    panels = (("Mombasa", "provisions", "Kilograms"), ("Djibouti", "lubricants", "Litres"))
    for ax, (port, category, ylab) in zip(axes, panels):
        actual, naive, model = _stitched(daily, predictions, port, category, horizon=14)
        ax.plot(actual["date"], actual["demand"], color="#C5D0D4", lw=1.0, label="Actual")
        ax.plot(naive["target_date"], naive["yhat"], color=MODEL_COLORS["seasonal_naive"], lw=1.3, label="Seasonal naive")
        ax.plot(model["target_date"], model["yhat"], color=NAVY, lw=1.6, label="Boosting + lineup")
        ax.set_ylabel(ylab)
        ax.set_title(f"{port} {CATEGORY_LABELS[category].lower()}", loc="left", color=INK, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    fig.suptitle("Holdout forecasts, 14-day blocks after March 2026", x=0.02, ha="left", color=INK, fontsize=14)
    fig.text(0.02, 0.01, "Synthetic series. Blocks do not overlap. Models were fit only on targets through 31 Mar 2026.", color=SLATE, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    _save(fig, path, tight=False)


def plot_model_comparison(metrics: pd.DataFrame, path: Path) -> None:
    part = metrics.loc[(metrics["level"] == "horizon_total") & (metrics["horizon"] == 14)].copy()
    part = part.sort_values("wape", ascending=True)
    labels = [MODEL_LABELS.get(model, model) for model in part["model"]]
    colors = [MODEL_COLORS.get(model, TEAL) for model in part["model"]]
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    y = np.arange(len(part))
    ax.barh(y, part["wape"] * 100, color=colors)
    ax.set_yticks(y, labels)
    for i, value in enumerate(part["wape"] * 100):
        ax.text(value + 0.4, i, f"{value:.1f}%", va="center", color=INK, fontsize=9)
    _finish(ax, "14-day order error on the synthetic holdout", "Value-weighted absolute percentage error of the horizon total", "WAPE (%)")
    ax.set_xlim(0, max(part["wape"] * 100) * 1.18)
    _save(fig, path)


def plot_wape_by_category(by_category: pd.DataFrame, path: Path) -> None:
    part = by_category.loc[(by_category["horizon"] == 14) & (by_category["model"].isin(["seasonal_naive", "hgb_lineup"]))].copy()
    categories = [c for c in CATEGORY_LABELS if c in set(part["category"])]
    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    width = 0.36
    for shift, model, label in ((-width / 2, "seasonal_naive", "Seasonal naive"), (width / 2, "hgb_lineup", "Boosting + lineup")):
        values = []
        for category in categories:
            row = part.loc[(part["category"] == category) & (part["model"] == model), "total_wape"]
            values.append(float(row.iloc[0]) * 100 if len(row) else np.nan)
        ax.bar(x + shift, values, width=width, color=MODEL_COLORS[model], label=label)
    ax.set_xticks(x, [CATEGORY_LABELS[c] for c in categories], rotation=15, ha="right")
    _finish(ax, "Where the vessel lineup earns its keep", "14-day total WAPE by category, synthetic holdout", "WAPE (%)")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_importance(importance: pd.DataFrame, path: Path) -> None:
    top = importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    colors = [NAVY if feature in {"n_calls", "total_crew", "total_dwt", "total_passengers", "n_container", "n_bulk", "n_tanker", "n_passenger", "n_general_cargo", "n_roro", "n_fishing"} else TEAL for feature in top["feature"]]
    ax.barh(top["feature"], top["mae_increase"], color=colors)
    _finish(ax, "Shuffling these features hurts the most", "Mean absolute error increase on a late training slice, horizon 14", "MAE increase")
    _save(fig, path)


def plot_replenishment_example(daily, predictions, residual_std, path: Path) -> None:
    """One Mombasa provisions origin: history, 14-day forecast, and an 90% band."""

    port, category = "Mombasa", "provisions"
    model = predictions.loc[(predictions["model"] == "hgb_lineup") & (predictions["port"] == port) & (predictions["category"] == category) & (predictions["horizon"] <= 14)]
    if model.empty:
        fig, ax = plt.subplots()
        ax.set_axis_off()
        _save(fig, path)
        return
    preferred = pd.Timestamp("2026-06-23")
    origins = pd.to_datetime(model["origin_date"].unique())
    origin = preferred if preferred in set(origins) else pd.Timestamp(sorted(origins)[len(origins) // 2])
    path_df = model.loc[model["origin_date"] == origin].sort_values("horizon")
    history = _series(daily, port, category)
    history = history.loc[(history["date"] <= origin) & (history["date"] >= origin - pd.Timedelta(days=56))]
    sigma = [residual_std.get(category, {}).get(int(step), 0.0) for step in path_df["horizon"]]
    z = 1.64485
    lower = np.maximum(path_df["yhat"].to_numpy() - z * np.asarray(sigma), 0)
    upper = path_df["yhat"].to_numpy() + z * np.asarray(sigma)
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    ax.plot(history["date"], history["demand"], color=SLATE, lw=1.3, label="History")
    ax.plot(path_df["target_date"], path_df["y"], color=CORAL, lw=1.3, label="Actual, held out")
    ax.plot(path_df["target_date"], path_df["yhat"], color=NAVY, lw=2.0, label="Forecast")
    ax.fill_between(path_df["target_date"], lower, upper, color=TEAL, alpha=0.18, label="90% band")
    ax.axvline(origin, color=SAND, lw=1.0, ls="--")
    _finish(ax, f"Example replenishment window from {origin.date()}", "Mombasa provisions, synthetic holdout", "Kilograms")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    _save(fig, path)


def _stitched(daily, predictions, port, category, horizon: int):
    actual = _series(daily, port, category)
    model_names = {"seasonal_naive": None, "hgb_lineup": None}
    frames = {}
    for name in model_names:
        part = predictions.loc[
            (predictions["model"] == name)
            & (predictions["port"] == port)
            & (predictions["category"] == category)
            & (predictions["horizon"] <= horizon)
        ].sort_values(["origin_date", "horizon"])
        frames[name] = part
    start = frames["hgb_lineup"]["target_date"].min()
    end = frames["hgb_lineup"]["target_date"].max()
    actual = actual.loc[actual["date"].between(start, end)]
    return actual, frames["seasonal_naive"], frames["hgb_lineup"]


def _series(daily: pd.DataFrame, port: str, category: str) -> pd.DataFrame:
    return daily.loc[(daily["port"] == port) & (daily["category"] == category)].sort_values("date")


def _finish(ax, title: str, subtitle: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=13, pad=16)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, color=SLATE, fontsize=8, va="bottom")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK)
    ax.yaxis.label.set_color(INK)
    ax.set_facecolor("white")


def _save(fig, path: Path, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
