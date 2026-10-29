"""Streamlit entry point for the chandlery demand forecast.

Michael Mutie (@mutieMichael), Silent Ramihel Limited.

Run locally with ``streamlit run app.py``. On Streamlit Community Cloud,
point the app at this file. It reads the synthetic book and the model
bundle produced by ``make train``; it does not need any secrets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chandlery_forecast.config import (
    CATEGORIES,
    CATEGORY_LABELS,
    DEFAULT_LEAD_TIME_DAYS,
    MODEL_LABELS,
    PORTS,
    SEED,
    TRAIN_CUTOFF,
    UNITS,
    UNIT_PRICES_USD,
)
from chandlery_forecast.features import build_forecast_frame
from chandlery_forecast.generate import generate_dataset, load_book
from chandlery_forecast.inventory import sigma_path, suggest_order
from chandlery_forecast.paths import BUNDLE_PATH, FIGURES_DIR, REPORTS_DIR

st.set_page_config(
    page_title="Chandlery demand forecast",
    page_icon="⚓",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; max-width: 1180px; }
      .banner {
        background: #0B3A4A; color: #F4F7F8; border-radius: 10px;
        padding: 0.9rem 1.1rem; margin-bottom: 0.8rem;
      }
      .banner h1 { color: #F4F7F8; font-size: 1.55rem; margin: 0 0 0.2rem 0; font-weight: 560; }
      .banner p { margin: 0; color: #D5E2E6; font-size: 0.95rem; }
      .note {
        background: #F4F7F8; border-left: 3px solid #C4A36A; padding: 0.55rem 0.8rem;
        color: #1C2830; border-radius: 4px; font-size: 0.92rem;
      }
      div[data-testid="stMetric"] {
        background: #F7FAFB; border: 1px solid #E3EBEE; padding: 0.55rem 0.7rem; border-radius: 8px;
      }
      .footer { color: #5C6B73; font-size: 0.85rem; margin-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle():
    import joblib

    if not BUNDLE_PATH.exists():
        return None
    return joblib.load(BUNDLE_PATH)


@st.cache_data
def load_tables():
    book = load_book()
    metrics = pd.read_csv(REPORTS_DIR / "metrics.csv") if (REPORTS_DIR / "metrics.csv").exists() else None
    by_category = (
        pd.read_csv(REPORTS_DIR / "metrics_by_category.csv")
        if (REPORTS_DIR / "metrics_by_category.csv").exists()
        else None
    )
    return book, metrics, by_category


@st.cache_data
def forward_book(port: str, start: str, end: str):
    """Illustrative calls after the extract. A different seed from the history."""

    return generate_dataset(seed=SEED + 17, start=start, end=end, ports=(port,), categories=("provisions",))


def extend_support(book, port: str, origin: pd.Timestamp, steps: int):
    data_end = pd.to_datetime(book.daily_demand["date"]).max()
    target_end = origin + pd.Timedelta(days=steps)
    if target_end <= data_end:
        return book.lineup, book.holidays, False
    future_start = data_end + pd.Timedelta(days=1)
    forward = forward_book(port, str(future_start.date()), str(target_end.date()))
    lineup = pd.concat([book.lineup, forward.lineup], ignore_index=True)
    holidays = pd.concat([book.holidays, forward.holidays], ignore_index=True)
    return lineup, holidays, True


def format_qty(value: float, unit: str) -> str:
    if unit == "kg" and abs(value) >= 1000:
        return f"{value / 1000:,.1f} t"
    if abs(value) >= 100:
        return f"{value:,.0f} {unit}"
    return f"{value:,.1f} {unit}"


def render_banner() -> None:
    st.markdown(
        """
        <div class="banner">
          <h1>Chandlery demand forecast</h1>
          <p>Michael Mutie · Silent Ramihel Limited · synthetic order book for provisions, water, lubricants, stores, and safety gear.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="note">Every vessel, quantity, price, and error in this app is <b>synthetic</b>. '
        "It comes from a fixed seed so the same book can be reproduced. It is not a record of Mombasa, "
        "Dar es Salaam, Djibouti, Durban, or any client.</div>",
        unsafe_allow_html=True,
    )


def page_forecast(book, bundle, by_category) -> None:
    model = bundle["model"]
    data_end = pd.to_datetime(book.daily_demand["date"]).max()
    data_start = pd.to_datetime(book.daily_demand["date"]).min()

    with st.sidebar:
        st.header("Plan a purchase")
        port = st.selectbox("Port", list(PORTS), index=0)
        category = st.selectbox("Category", list(CATEGORIES), format_func=lambda key: CATEGORY_LABELS[key])
        horizon = st.select_slider("Review cycle (days)", options=[7, 14, 28], value=14)
        lead_default = int(DEFAULT_LEAD_TIME_DAYS[category])
        lead_cap = max(0, 42 - int(horizon))
        lead = st.number_input(
            "Lead time (days)",
            min_value=0,
            max_value=lead_cap,
            value=min(lead_default, lead_cap),
            step=1,
            key=f"lead-{category}-{horizon}",
        )
        service = st.slider("Service level", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
        origin = st.date_input(
            "As-of date",
            value=pd.Timestamp("2026-09-01").date(),
            min_value=(data_start + pd.Timedelta(days=40)).date(),
            max_value=data_end.date(),
        )
        origin_ts = pd.Timestamp(origin)
        history = book.daily_demand.loc[
            (book.daily_demand["port"] == port)
            & (book.daily_demand["category"] == category)
            & (book.daily_demand["date"] <= origin_ts)
        ].sort_values("date")
        trailing = float(history.tail(14)["demand"].sum())
        on_hand = st.number_input("On hand", min_value=0.0, value=round(trailing * 0.45, 1), step=1.0, key=f"oh-{port}-{category}")
        inbound = st.number_input("Inbound already ordered", min_value=0.0, value=0.0, step=1.0)

    cover = int(horizon + lead)
    lineup, holidays, used_forward = extend_support(book, port, origin_ts, cover)
    frame = build_forecast_frame(
        book.daily_demand,
        lineup,
        holidays,
        origin_ts,
        cover,
        ports=[port],
        categories=[category],
    ).sort_values("horizon")
    forecast = model.predict(frame)
    sigma = sigma_path(bundle["residual_std"].get(category, {}), cover, fallback=float(np.std(forecast) or 1.0))
    unit = UNITS[category]
    price = float(UNIT_PRICES_USD[category])
    suggestion = suggest_order(
        daily_forecast=forecast,
        sigma_by_step=sigma,
        inflation=float(bundle["inflation"].get(category, 1.0)),
        on_hand=on_hand,
        inbound=inbound,
        lead_time_days=int(lead),
        service_level=float(service),
        unit=unit,
        unit_price_usd=price,
    )

    actual = book.daily_demand.loc[
        (book.daily_demand["port"] == port) & (book.daily_demand["category"] == category),
        ["date", "demand"],
    ]
    view = frame[["target_date", "horizon", "n_calls", "total_crew", "total_dwt"]].copy()
    view["forecast"] = forecast
    view = view.merge(actual, left_on="target_date", right_on="date", how="left")
    z90 = 1.64485
    view["lower"] = np.maximum(view["forecast"] - z90 * sigma, 0)
    view["upper"] = view["forecast"] + z90 * sigma

    st.subheader(f"{port} · {CATEGORY_LABELS[category]}")
    if origin_ts >= pd.Timestamp(TRAIN_CUTOFF):
        st.caption(f"Out of sample. The model was fit on targets through {pd.Timestamp(TRAIN_CUTOFF).date()}.")
    else:
        st.caption("This as-of date sits inside the training window, so the line is an illustration, not a fair test.")
    if used_forward:
        st.caption(
            f"Days after {data_end.date()} use an illustrative forward lineup "
            f"(generator seed {SEED + 17}), not a live ETA feed."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected demand to cover", format_qty(suggestion.expected_demand, unit))
    c2.metric("Suggested order", format_qty(suggestion.suggested_qty, unit))
    c3.metric("Indicative value", f"USD {suggestion.indicative_value_usd:,.0f}")
    category_wape = _category_wape(by_category, category, horizon)
    c4.metric("14-day holdout WAPE" if horizon != 14 else "Holdout WAPE, this category", category_wape)

    chart = _forecast_chart(history.tail(60), view, origin_ts, int(lead), unit)
    st.plotly_chart(chart, use_container_width=True)

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Suggested stock order**")
        st.write(suggestion.note)
        order_table = pd.DataFrame(
            {
                "Component": [
                    "Expected demand over lead time + review cycle",
                    "Safety stock",
                    "On hand",
                    "Inbound",
                    "Suggested order",
                ],
                "Quantity": [
                    suggestion.expected_demand,
                    suggestion.safety_stock,
                    suggestion.on_hand,
                    suggestion.inbound,
                    suggestion.suggested_qty,
                ],
            }
        )
        order_table["Quantity"] = order_table["Quantity"].map(lambda value: format_qty(value, unit))
        st.dataframe(order_table, hide_index=True, use_container_width=True)
        st.caption(
            f"Safety stock uses the category's backtest residual, inflated for day-to-day correlation, "
            f"at z = {suggestion.z:.2f}. The USD figure multiplies the order by a synthetic basket price "
            f"of {price:.2f} per {unit}. It is not a supplier quote."
        )
        if cover > 28:
            st.caption("Days beyond 28 reuse the 28-day residual scale. The backtest itself scores 7, 14, and 28 days.")
    with right:
        st.markdown("**Daily forecast**")
        table = view[["target_date", "horizon", "forecast", "demand", "n_calls", "total_crew"]].copy()
        table["target_date"] = table["target_date"].dt.date
        table = table.rename(
            columns={
                "target_date": "Date",
                "horizon": "Day",
                "forecast": "Forecast",
                "demand": "Actual, if in the extract",
                "n_calls": "Calls on the lineup",
                "total_crew": "Crew",
            }
        )
        st.dataframe(table, hide_index=True, use_container_width=True, height=360)


def _category_wape(by_category, category: str, horizon: int) -> str:
    if by_category is None or by_category.empty:
        return "—"
    # The selector is the review cycle. Accuracy quoted here is the 14-day
    # total, which is the decision the backtest was built around, except when
    # the user picked 7 or 28 and we have that row.
    use = horizon if horizon in set(by_category["horizon"]) else 14
    row = by_category.loc[
        (by_category["category"] == category)
        & (by_category["model"] == "hgb_lineup")
        & (by_category["horizon"] == use)
    ]
    if row.empty:
        return "—"
    return f"{float(row.iloc[0]['total_wape']) * 100:.1f}%"


def _forecast_chart(history, view, origin, lead: int, unit: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["date"], y=history["demand"], name="History", line=dict(color="#5C6B73", width=1.4)))
    fig.add_trace(
        go.Scatter(x=view["target_date"], y=view["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=view["target_date"],
            y=view["lower"],
            fill="tonexty",
            fillcolor="rgba(31,122,140,0.18)",
            line=dict(width=0),
            name="90% band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(go.Scatter(x=view["target_date"], y=view["forecast"], name="Forecast", line=dict(color="#0B3A4A", width=2.4)))
    observed = view.dropna(subset=["demand"])
    if not observed.empty:
        fig.add_trace(
            go.Scatter(
                x=observed["target_date"],
                y=observed["demand"],
                name="Actual in the extract",
                mode="lines",
                line=dict(color="#D4644A", width=1.5, dash="dot"),
            )
        )
    delivery = origin + pd.Timedelta(days=lead)
    fig.add_vrect(x0=origin, x1=delivery, fillcolor="#C4A36A", opacity=0.13, line_width=0, annotation_text="Lead time" if lead else None, annotation_position="top left")
    fig.add_vline(x=origin, line_dash="dash", line_color="#C4A36A", line_width=1)
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=420,
        legend=dict(orientation="h", y=1.12),
        yaxis_title=unit,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E6EEF0")
    return fig


def page_backtest(metrics, by_category) -> None:
    st.subheader("Holdout, after 31 March 2026")
    st.write(
        "The regression models are fit once, on demand targets on or before the cutoff. "
        "Each later origin forecasts the next days from demand already observed and from "
        "the vessel lineup of the days being forecast. Holt-Winters and the seasonal naive "
        "are refit on each series at each origin. The score below is on synthetic USD, so "
        "kilograms are not added to litres."
    )
    if metrics is None:
        st.info("Metrics appear after `make train`.")
        return
    primary = metrics.loc[metrics["horizon"] == 14].copy()
    primary["Model"] = primary["model"].map(MODEL_LABELS)
    primary["Slice"] = primary["level"].map({"daily": "Each day", "horizon_total": "14-day total"})
    primary["WAPE"] = (primary["wape"] * 100).map(lambda value: f"{value:.1f}%")
    primary["MAE (USD)"] = primary["mae"].map(lambda value: f"{value:,.0f}")
    primary["RMSE (USD)"] = primary["rmse"].map(lambda value: f"{value:,.0f}")
    primary["MASE"] = primary["mase"].map(lambda value: "—" if pd.isna(value) else f"{value:.2f}")
    st.dataframe(
        primary[["Model", "Slice", "WAPE", "MAE (USD)", "RMSE (USD)", "MASE"]],
        hide_index=True,
        use_container_width=True,
    )
    st.caption("WAPE is the sum of absolute errors divided by the sum of actual synthetic order value. MASE is 1.0 for a seasonal naive fit in-sample; lower is better.")

    if by_category is not None:
        st.markdown("**14-day total WAPE by category** (native units, within the category)")
        wide = by_category.loc[by_category["horizon"] == 14, ["model", "category", "total_wape"]].copy()
        wide["total_wape"] = (wide["total_wape"] * 100).round(1)
        table = wide.pivot(index="category", columns="model", values="total_wape")
        table = table.rename(index=CATEGORY_LABELS, columns=MODEL_LABELS)
        st.dataframe(table, use_container_width=True)

    figures = [
        ("model_comparison.png", "Value-weighted 14-day error"),
        ("wape_by_category.png", "Error by category"),
        ("backtest_overlay.png", "Holdout paths"),
        ("feature_importance.png", "What the lineup model uses"),
    ]
    for name, caption in figures:
        path = FIGURES_DIR / name
        if path.exists():
            st.image(str(path), caption=caption)


def page_method() -> None:
    st.subheader("What the generator is doing")
    st.write(
        "There is no client extract. A seeded process builds vessel calls for four ports, "
        "gives each call a type, a size, a crew, and a voyage length, then turns some of "
        "those calls into orders. Provisions scale with people and voyage length. Lubricants "
        "scale with size. Safety gear is sparse and ticks up in a stylised March and September "
        "survey season. Sunday and public-holiday arrivals are booked on the next open desk day, "
        "which is why Monday is heavy."
    )
    st.write(
        "The forecast the order uses is a histogram gradient-boosting model on the log of "
        "demand. Features known on the morning of the origin are lags and rolling statistics. "
        "Features known about the target day are the calendar and the vessel lineup (calls, "
        "crew, deadweight, type mix). The lineup does not include the voyage length or the "
        "fact of whether the master will actually buy, so those stay in the residual."
    )
    st.write(
        "Seasonal naive (repeat last week), weekly Holt-Winters, a Fourier ridge regression, "
        "and the same booster without the lineup are the comparisons. They are there to show "
        "what the lineup is worth, not to crowd the purchase screen."
    )
    st.markdown("**A real engagement would replace the generator**")
    st.write(
        "Pre-arrival notices, the chandler's own invoices, and stock on hand would take the "
        "place of the synthetic book. The evaluation design — a cutoff, rolling origins, and "
        "a value-weighted error on the order quantity — would stay."
    )


def main() -> None:
    render_banner()
    bundle = load_bundle()
    try:
        book, metrics, by_category = load_tables()
    except FileNotFoundError:
        book = metrics = by_category = None
    if bundle is None or book is None:
        st.error("The synthetic book or the model bundle is missing. From the repo root run `make data` and `make train`.")
        st.stop()

    page = st.sidebar.radio("View", ["Forecast and order", "Backtest", "Method"], index=0)
    if page == "Forecast and order":
        page_forecast(book, bundle, by_category)
    elif page == "Backtest":
        page_backtest(metrics, by_category)
    else:
        page_method()

    st.markdown(
        '<p class="footer">Michael Mutie (@mutieMichael), Silent Ramihel Limited · silentramihellimited@gmail.com</p>',
        unsafe_allow_html=True,
    )


main()
