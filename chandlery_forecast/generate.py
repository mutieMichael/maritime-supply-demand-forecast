"""Reproducible synthetic chandlery order book.

The generator builds a daily demand series from invented vessel calls. The
process is deterministic for a given seed.

Demand identity, for each port, category, and booking date:

    demand = (sum of orders booked that day + walk-in baseline) * port-day shock

Orders are placed by individual calls. A call books on its arrival date, or
on the next open desk day when the arrival falls on Sunday or a public
holiday. Quantity equations live in ``_order_quantity`` and are the only
place a vessel is turned into kilograms, cubic metres, litres, or units.

Nothing in the output is a real port statistic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from chandlery_forecast.config import (
    AR_INNOVATION_SIGMA,
    AR_PHI,
    BASELINE_OPEN_DAY,
    CATEGORIES,
    CHRISTMAS_PROVISIONS_FACTOR,
    CHRISTMAS_WATER_FACTOR,
    CLOSED_DAY_BASELINE_FRACTION,
    CLOSURE_WEEKDAY,
    DISRUPTIONS,
    DOW_ARRIVAL_INDEX,
    EASTER,
    EID_AL_ADHA,
    EID_AL_FITR,
    END_DATE,
    FIXED_HOLIDAYS,
    GENERATOR_VERSION,
    LUBE_LITRES_PER_THOUSAND_DWT,
    MONTH_INDEX,
    ORDER_NOISE_SIGMA,
    ORDER_PROBABILITY,
    PORTS,
    PRE_EID_PROVISIONS_FACTOR,
    PRE_EID_WATER_FACTOR,
    RAMADAN,
    RAMADAN_PROVISIONS_FACTOR,
    SAFETY_SURVEY_MONTHS,
    SAFETY_SURVEY_PROB_MULTIPLIER,
    SEED,
    START_DATE,
    UNITS,
    VESSEL_PROFILES,
)
from chandlery_forecast.paths import DATA_DIR

VESSEL_TYPES = tuple(VESSEL_PROFILES)


@dataclass
class SyntheticBook:
    """Tables that make up one generated order book."""

    vessel_calls: pd.DataFrame
    orders: pd.DataFrame
    daily_demand: pd.DataFrame
    lineup: pd.DataFrame
    holidays: pd.DataFrame
    disruptions: pd.DataFrame
    metadata: dict


def generate_dataset(
    seed: int = SEED,
    start: str = START_DATE,
    end: str = END_DATE,
    ports: tuple[str, ...] | list[str] | None = None,
    categories: tuple[str, ...] | list[str] | None = None,
) -> SyntheticBook:
    """Generate vessel calls, orders, and a complete daily demand panel.

    ``ports`` and ``categories`` default to the full demo set. Tests use a
    shorter date range; the seed still fixes every draw.
    """

    port_names = tuple(ports) if ports is not None else tuple(PORTS)
    category_names = tuple(categories) if categories is not None else CATEGORIES
    unknown = set(port_names) - set(PORTS)
    if unknown:
        raise ValueError(f"Unknown ports: {sorted(unknown)}")
    unknown_cat = set(category_names) - set(CATEGORIES)
    if unknown_cat:
        raise ValueError(f"Unknown categories: {sorted(unknown_cat)}")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts < start_ts:
        raise ValueError("end date is before start date")

    # A few days past the extract so a closure can push a booking forward
    # without falling off the calendar arrays.
    calendar_end = end_ts + pd.Timedelta(days=10)
    dates = pd.date_range(start_ts, calendar_end, freq="D")

    holidays = build_holiday_calendar(dates, port_names)
    disruptions = disruptions_frame(port_names)
    rng = np.random.default_rng(seed)

    calls = _generate_calls(rng, dates, port_names, holidays, disruptions)
    orders = _generate_orders(rng, calls, holidays, category_names)
    daily, lineup = _aggregate(
        dates=pd.date_range(start_ts, end_ts, freq="D"),
        calls=calls,
        orders=orders,
        holidays=holidays,
        ports=port_names,
        categories=category_names,
        rng=rng,
    )

    in_window = (calls["arrival_date"] >= start_ts) & (calls["arrival_date"] <= end_ts)
    calls = calls.loc[in_window].reset_index(drop=True)
    orders = orders.loc[orders["order_date"].between(start_ts, end_ts)].reset_index(drop=True)
    holidays = holidays.loc[holidays["date"].between(start_ts, end_ts)].reset_index(drop=True)

    metadata = {
        "disclaimer": (
            "Synthetic data generated for demonstration. Not measurements "
            "from any port, vessel, or client."
        ),
        "generator_version": GENERATOR_VERSION,
        "seed": int(seed),
        "start": str(start_ts.date()),
        "end": str(end_ts.date()),
        "ports": list(port_names),
        "categories": list(category_names),
        "n_calls": int(len(calls)),
        "n_orders": int(len(orders)),
        "n_demand_rows": int(len(daily)),
    }
    return SyntheticBook(
        vessel_calls=calls,
        orders=orders,
        daily_demand=daily,
        lineup=lineup,
        holidays=holidays,
        disruptions=disruptions,
        metadata=metadata,
    )


def save_book(book: SyntheticBook, directory=DATA_DIR) -> None:
    """Write the book to CSV/JSON under ``directory``."""

    directory.mkdir(parents=True, exist_ok=True)
    book.vessel_calls.to_csv(directory / "vessel_calls.csv", index=False)
    book.orders.to_csv(directory / "orders.csv", index=False)
    book.daily_demand.to_csv(directory / "daily_demand.csv", index=False)
    book.lineup.to_csv(directory / "lineup.csv", index=False)
    book.holidays.to_csv(directory / "holidays.csv", index=False)
    book.disruptions.to_csv(directory / "disruptions.csv", index=False)
    (directory / "metadata.json").write_text(json.dumps(book.metadata, indent=2) + "\n")


def load_book(directory=DATA_DIR) -> SyntheticBook:
    """Load a book written by :func:`save_book`."""

    def _csv(name: str, dates: list[str]) -> pd.DataFrame:
        frame = pd.read_csv(directory / name, parse_dates=dates)
        return frame

    meta_path = directory / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return SyntheticBook(
        vessel_calls=_csv("vessel_calls.csv", ["arrival_date", "order_date"]),
        orders=_csv("orders.csv", ["order_date", "arrival_date"]),
        daily_demand=_csv("daily_demand.csv", ["date"]),
        lineup=_csv("lineup.csv", ["date"]),
        holidays=_csv("holidays.csv", ["date"]),
        disruptions=_csv("disruptions.csv", ["start", "end"]),
        metadata=metadata,
    )


def build_holiday_calendar(dates: pd.DatetimeIndex, ports: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """One row per port and date, with the flags the generator and models share."""

    ramadan_days = _inclusive_ranges(RAMADAN)
    eid_fitr = _eid_days(EID_AL_FITR)
    eid_adha = _eid_days(EID_AL_ADHA)
    eid_days = eid_fitr | eid_adha
    pre_eid = set()
    for first in list(EID_AL_FITR) + list(EID_AL_ADHA):
        first_ts = pd.Timestamp(first)
        for offset in (1, 2, 3):
            pre_eid.add(first_ts - pd.Timedelta(days=offset))

    easter_days = {}
    for good_friday, easter_monday in EASTER:
        easter_days[pd.Timestamp(good_friday)] = "Good Friday"
        easter_days[pd.Timestamp(easter_monday)] = "Easter Monday"

    rows: list[dict] = []
    for port in ports:
        fixed = {}
        for month, day, name in FIXED_HOLIDAYS[port]:
            for year in sorted({d.year for d in dates}):
                try:
                    fixed[pd.Timestamp(year=year, month=month, day=day)] = name
                except ValueError:
                    continue
        holiday_dates = sorted(set(fixed) | set(easter_days) | eid_days)
        holiday_dates = [d for d in holiday_dates if dates[0] <= d <= dates[-1]]
        next_holiday = _days_until(dates, holiday_dates)

        for i, day in enumerate(dates):
            names: list[str] = []
            if day in fixed:
                names.append(fixed[day])
            if day in easter_days:
                names.append(easter_days[day])
            if day in eid_fitr:
                names.append("Eid al-Fitr")
            if day in eid_adha:
                names.append("Eid al-Adha")
            is_public = bool(names)
            rows.append(
                {
                    "date": day,
                    "port": port,
                    "is_public_holiday": is_public,
                    "is_ramadan": day in ramadan_days and day not in eid_days,
                    "is_pre_eid": day in pre_eid,
                    "is_eid": day in eid_days,
                    "is_christmas_week": bool(day.month == 12 and day.day >= 18),
                    "days_until_holiday": int(next_holiday[i]),
                    "holiday_name": "; ".join(names),
                    "is_closed": bool(is_public or int(day.day_of_week) == CLOSURE_WEEKDAY),
                }
            )
    calendar = pd.DataFrame(rows)
    calendar["date"] = pd.to_datetime(calendar["date"])
    return calendar


def disruptions_frame(ports: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """Invented disruption windows that fall on the selected ports."""

    rows = []
    for item in DISRUPTIONS:
        if item["port"] not in ports:
            continue
        rows.append(
            {
                "port": item["port"],
                "start": pd.Timestamp(item["start"]),
                "end": pd.Timestamp(item["end"]),
                "name": item["name"],
                "multiplier": item["multiplier"],
                "rebound": item["rebound"],
                "rebound_days": item["rebound_days"],
            }
        )
    return pd.DataFrame(rows)


def booking_date(arrival: pd.Timestamp, closed: set[pd.Timestamp]) -> pd.Timestamp:
    """Move a requisition off a closed desk day onto the next open day."""

    day = pd.Timestamp(arrival)
    for _ in range(10):
        if day not in closed:
            return day
        day += pd.Timedelta(days=1)
    return day


def _generate_calls(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    ports: tuple[str, ...],
    holidays: pd.DataFrame,
    disruptions: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for port in ports:
        spec = PORTS[port]
        n_days = len(dates)
        month_ix = np.array(spec["month_index"], dtype=float)[dates.month - 1]
        dow_ix = np.array(DOW_ARRIVAL_INDEX, dtype=float)[dates.day_of_week]
        years = (dates - dates[0]).days / 365.25
        trend = (1.0 + spec["annual_growth"]) ** years
        disruption = _disruption_multiplier(port, dates, disruptions)
        lam = spec["mean_daily_arrivals"] * month_ix * dow_ix * trend * disruption
        counts = rng.poisson(lam)
        total = int(counts.sum())
        if total == 0:
            continue
        day_index = np.repeat(np.arange(n_days), counts)
        arrival = pd.to_datetime(dates[day_index])
        types = _sample_types(rng, total, spec["type_mix"])
        profile = _sample_profiles(rng, types)
        closed = set(holidays.loc[(holidays["port"] == port) & holidays["is_closed"], "date"])
        # Map every arrival day once, then look up. Closures are few.
        unique_arrivals = pd.to_datetime(pd.Index(arrival).unique())
        booking_lookup = {day: booking_date(day, closed) for day in unique_arrivals}
        order_dates = arrival.map(booking_lookup)
        frame = pd.DataFrame(
            {
                "port": port,
                "country": spec["country"],
                "arrival_date": arrival,
                "order_date": pd.to_datetime(order_dates),
                "vessel_type": types,
                "dwt": np.round(profile["dwt"], 0),
                "crew": profile["crew"],
                "passengers": profile["passengers"],
                "stay_days": profile["stay_days"],
                "voyage_days": profile["voyage_days"],
            }
        )
        frames.append(frame)

    calls = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    calls = calls.sort_values(["arrival_date", "port", "vessel_type"]).reset_index(drop=True)
    calls.insert(0, "call_id", [f"C{i:06d}" for i in range(1, len(calls) + 1)])
    return calls


def _generate_orders(
    rng: np.random.Generator,
    calls: pd.DataFrame,
    holidays: pd.DataFrame,
    categories: tuple[str, ...],
) -> pd.DataFrame:
    if calls.empty:
        return pd.DataFrame(columns=["order_id", "call_id", "port", "order_date", "arrival_date", "vessel_type", "category", "quantity"])

    flags = holidays[
        ["date", "port", "is_ramadan", "is_pre_eid", "is_eid", "is_christmas_week"]
    ].rename(columns={"date": "arrival_date"})
    enriched = calls.merge(flags, on=["arrival_date", "port"], how="left")
    for column in ("is_ramadan", "is_pre_eid", "is_eid", "is_christmas_week"):
        enriched[column] = enriched[column].fillna(False).astype(bool)

    pieces = []
    for category in categories:
        quantity, ordered = _order_quantity(rng, enriched, category)
        keep = ordered & (quantity > 0)
        if not keep.any():
            continue
        piece = enriched.loc[keep, ["call_id", "port", "order_date", "arrival_date", "vessel_type"]].copy()
        piece["category"] = category
        piece["quantity"] = np.round(quantity[keep], 1)
        pieces.append(piece)

    if not pieces:
        return pd.DataFrame(columns=["order_id", "call_id", "port", "order_date", "arrival_date", "vessel_type", "category", "quantity"])
    orders = pd.concat(pieces, ignore_index=True)
    orders.insert(0, "order_id", [f"O{i:07d}" for i in range(1, len(orders) + 1)])
    return orders


def _order_quantity(
    rng: np.random.Generator,
    calls: pd.DataFrame,
    category: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(quantity, ordered)`` arrays aligned to ``calls``.

    Random draws happen in a fixed order for every category: one uniform for
    the buy/no-buy decision, one lognormal multiplier, and for safety gear one
    extra uniform for the basket size. Rows that are not ordered still consume
    a noise draw so a change in one probability does not reshuffle later noise.
    """

    n = len(calls)
    vtype = calls["vessel_type"].to_numpy()
    month = calls["arrival_date"].dt.month.to_numpy()
    month_ix = np.asarray(MONTH_INDEX[category], dtype=float)[month - 1]
    crew = calls["crew"].to_numpy(dtype=float)
    pax = calls["passengers"].to_numpy(dtype=float)
    dwt = calls["dwt"].to_numpy(dtype=float)
    voyage = calls["voyage_days"].to_numpy(dtype=float)
    stay = calls["stay_days"].to_numpy(dtype=float)
    people = crew + np.where(vtype == "passenger", 0.30 * pax, 0.0)

    if category == "provisions":
        raw = people * voyage * 2.10 * month_ix
        raw = raw * np.where(calls["is_ramadan"].to_numpy(), RAMADAN_PROVISIONS_FACTOR, 1.0)
        raw = raw * np.where(calls["is_pre_eid"].to_numpy(), PRE_EID_PROVISIONS_FACTOR, 1.0)
        raw = raw * np.where(calls["is_christmas_week"].to_numpy(), CHRISTMAS_PROVISIONS_FACTOR, 1.0)
    elif category == "fresh_water":
        raw = (people * voyage * 0.12 + dwt / 20_000.0 * 8.0) * month_ix
        raw = raw * np.where(calls["is_pre_eid"].to_numpy(), PRE_EID_WATER_FACTOR, 1.0)
        raw = raw * np.where(calls["is_christmas_week"].to_numpy(), CHRISTMAS_WATER_FACTOR, 1.0)
    elif category == "lubricants":
        per = np.array([LUBE_LITRES_PER_THOUSAND_DWT[t] for t in vtype], dtype=float)
        raw = (dwt / 1_000.0) * per * month_ix
    elif category == "deck_engine_stores":
        # A longer stay is a stand-in for a repair call, which lifts stores.
        raw = (6.0 + dwt / 12_000.0) * np.where(stay >= 4.0, 1.35, 1.0) * month_ix
    elif category == "safety_equipment":
        raw = rng.uniform(2.0, 10.0, size=n) * month_ix
    else:
        raise ValueError(category)

    probability = np.array([ORDER_PROBABILITY[category][t] for t in vtype], dtype=float)
    if category == "safety_equipment":
        probability = np.where(np.isin(month, SAFETY_SURVEY_MONTHS), probability * SAFETY_SURVEY_PROB_MULTIPLIER, probability)
    probability = np.clip(probability, 0.0, 0.98)

    ordered = rng.random(n) < probability
    sigma = ORDER_NOISE_SIGMA[category]
    noise = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=n)
    quantity = np.where(ordered, raw * noise, 0.0)
    return quantity, ordered


def _aggregate(
    dates: pd.DatetimeIndex,
    calls: pd.DataFrame,
    orders: pd.DataFrame,
    holidays: pd.DataFrame,
    ports: tuple[str, ...],
    categories: tuple[str, ...],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lineup = _lineup_from_calls(calls, dates, ports)
    grid = pd.MultiIndex.from_product([dates, ports, categories], names=["date", "port", "category"]).to_frame(index=False)

    if orders.empty:
        summed = grid.copy()
        summed["vessel_qty"] = 0.0
        summed["n_vessel_orders"] = 0
    else:
        grouped = (
            orders.loc[orders["order_date"].between(dates[0], dates[-1])]
            .groupby(["order_date", "port", "category"], as_index=False)
            .agg(vessel_qty=("quantity", "sum"), n_vessel_orders=("quantity", "size"))
            .rename(columns={"order_date": "date"})
        )
        summed = grid.merge(grouped, on=["date", "port", "category"], how="left")
        summed["vessel_qty"] = summed["vessel_qty"].fillna(0.0)
        summed["n_vessel_orders"] = summed["n_vessel_orders"].fillna(0).astype(int)

    summed = summed.merge(
        holidays[["date", "port", "is_closed"]],
        on=["date", "port"],
        how="left",
    )
    summed["is_closed"] = summed["is_closed"].fillna(False)

    scale = summed["port"].map(lambda port: PORTS[port]["mean_daily_arrivals"] / 6.0)
    base_open = summed["category"].map(BASELINE_OPEN_DAY).astype(float) * scale
    baseline = np.where(summed["is_closed"].to_numpy(), base_open * CLOSED_DAY_BASELINE_FRACTION, base_open)
    shock = _port_day_shock(rng, dates, ports)
    summed = summed.merge(shock, on=["date", "port"], how="left")
    demand = (summed["vessel_qty"].to_numpy() + baseline) * summed["shock"].to_numpy()
    summed["baseline"] = np.round(baseline, 2)
    summed["demand"] = np.round(np.clip(demand, 0.0, None), 1)
    summed["unit"] = summed["category"].map(UNITS)
    daily = summed[
        ["date", "port", "category", "unit", "demand", "vessel_qty", "baseline", "n_vessel_orders"]
    ].sort_values(["port", "category", "date"]).reset_index(drop=True)
    daily["vessel_qty"] = np.round(daily["vessel_qty"], 1)
    return daily, lineup


def _lineup_from_calls(calls: pd.DataFrame, dates: pd.DatetimeIndex, ports: tuple[str, ...]) -> pd.DataFrame:
    """Vessel particulars booked onto each open (or closed) date.

    The lineup is what a chandler can see from pre-arrival notices: how many
    ships, how large, what type, and a rough crew. It deliberately excludes
    the voyage length and the realised order, which are not known until the
    master sends the requisition.
    """

    index = pd.MultiIndex.from_product([dates, ports], names=["date", "port"])
    base = index.to_frame(index=False)
    if calls.empty:
        for column in (
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
        ):
            base[column] = 0.0
        return base

    window = calls.loc[calls["order_date"].between(dates[0], dates[-1])].copy()
    window["n_calls"] = 1
    for vessel_type in VESSEL_TYPES:
        window[f"n_{vessel_type}"] = (window["vessel_type"] == vessel_type).astype(int)
    grouped = window.groupby(["order_date", "port"], as_index=False).agg(
        n_calls=("n_calls", "sum"),
        total_crew=("crew", "sum"),
        total_dwt=("dwt", "sum"),
        total_passengers=("passengers", "sum"),
        n_container=("n_container", "sum"),
        n_bulk=("n_bulk", "sum"),
        n_tanker=("n_tanker", "sum"),
        n_general_cargo=("n_general_cargo", "sum"),
        n_roro=("n_roro", "sum"),
        n_passenger=("n_passenger", "sum"),
        n_fishing=("n_fishing", "sum"),
    ).rename(columns={"order_date": "date"})
    lineup = base.merge(grouped, on=["date", "port"], how="left")
    for column in grouped.columns:
        if column in ("date", "port"):
            continue
        lineup[column] = lineup[column].fillna(0.0)
    return lineup.sort_values(["port", "date"]).reset_index(drop=True)


def _port_day_shock(rng: np.random.Generator, dates: pd.DatetimeIndex, ports: tuple[str, ...]) -> pd.DataFrame:
    """AR(1) multiplicative shock, mean 1, shared by every category in a port-day."""

    phi = AR_PHI
    sigma = AR_INNOVATION_SIGMA
    variance = sigma**2 / (1.0 - phi**2)
    frames = []
    for port in ports:
        eps = np.zeros(len(dates))
        eps[0] = rng.normal(0.0, sigma / np.sqrt(1.0 - phi**2))
        innov = rng.normal(0.0, sigma, size=len(dates) - 1)
        for t in range(1, len(dates)):
            eps[t] = phi * eps[t - 1] + innov[t - 1]
        shock = np.exp(eps - 0.5 * variance)
        frames.append(pd.DataFrame({"date": dates, "port": port, "shock": shock}))
    return pd.concat(frames, ignore_index=True)


def _sample_types(rng: np.random.Generator, n: int, mix: dict[str, float]) -> np.ndarray:
    types = np.array(list(mix))
    probs = np.array(list(mix.values()), dtype=float)
    probs = probs / probs.sum()
    return rng.choice(types, size=n, p=probs)


def _sample_profiles(rng: np.random.Generator, types: np.ndarray) -> dict[str, np.ndarray]:
    n = len(types)
    dwt = np.empty(n)
    crew = np.empty(n)
    passengers = np.zeros(n)
    stay = np.empty(n)
    voyage = np.empty(n)
    for vessel_type, profile in VESSEL_PROFILES.items():
        mask = types == vessel_type
        count = int(mask.sum())
        if count == 0:
            continue
        dwt[mask] = np.clip(rng.lognormal(profile["dwt_mu"], profile["dwt_sigma"], size=count), profile["dwt_min"], profile["dwt_max"])
        stay[mask] = np.clip(np.round(rng.lognormal(np.log(profile["stay_mu"]), 0.30, size=count)), 1, 14)
        voyage[mask] = np.clip(np.round(rng.lognormal(np.log(profile["voyage_mu"]), 0.35, size=count)), 4, 60)
        if vessel_type == "passenger":
            pax = rng.lognormal(profile["pax_mu"], profile["pax_sigma"], size=count)
            passengers[mask] = np.clip(np.round(pax), 40, 4_000)
            crew[mask] = np.clip(np.round(rng.normal(profile["crew_mu"], profile["crew_sigma"], size=count) + 0.07 * pax), 20, 450)
        else:
            crew[mask] = np.clip(np.round(rng.normal(profile["crew_mu"], profile["crew_sigma"], size=count)), 4, 40)
    return {
        "dwt": dwt,
        "crew": crew.astype(int),
        "passengers": passengers.astype(int),
        "stay_days": stay.astype(int),
        "voyage_days": voyage.astype(int),
    }


def _disruption_multiplier(port: str, dates: pd.DatetimeIndex, disruptions: pd.DataFrame) -> np.ndarray:
    factor = np.ones(len(dates), dtype=float)
    if disruptions.empty:
        return factor
    for row in disruptions.itertuples(index=False):
        if row.port != port:
            continue
        during = (dates >= row.start) & (dates <= row.end)
        rebound_end = row.end + pd.Timedelta(days=int(row.rebound_days))
        rebound = (dates > row.end) & (dates <= rebound_end)
        factor[during] *= float(row.multiplier)
        factor[rebound] *= float(row.rebound)
    return factor


def _inclusive_ranges(ranges: tuple[tuple[str, str], ...]) -> set[pd.Timestamp]:
    days: set[pd.Timestamp] = set()
    for start, end in ranges:
        days.update(pd.date_range(start, end, freq="D"))
    return days


def _eid_days(first_days: tuple[str, ...]) -> set[pd.Timestamp]:
    days: set[pd.Timestamp] = set()
    for first in first_days:
        start = pd.Timestamp(first)
        days.add(start)
        days.add(start + pd.Timedelta(days=1))
    return days


def _days_until(dates: pd.DatetimeIndex, holiday_dates: list[pd.Timestamp], cap: int = 21) -> np.ndarray:
    if not holiday_dates:
        return np.full(len(dates), cap, dtype=int)
    holidays = np.array(sorted(holiday_dates), dtype="datetime64[ns]")
    values = dates.to_numpy(dtype="datetime64[ns]")
    index = np.searchsorted(holidays, values, side="left")
    out = np.full(len(dates), cap, dtype=int)
    valid = index < len(holidays)
    delta = (holidays[index[valid]] - values[valid]).astype("timedelta64[D]").astype(int)
    out[valid] = np.minimum(delta, cap)
    return out
