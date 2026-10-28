"""Synthetic assumptions for the chandlery demand generator.

These constants are modelling choices for a fictional order book. They are
not estimates of real port traffic, crew levels, or prices. The narrative
uses four well-known ports so the demo is easy to talk about; the numbers
attached to them are invented.
"""

from __future__ import annotations

SEED = 42
GENERATOR_VERSION = "1.0.0"

# Synthetic extract. The held-out backtest starts the day after TRAIN_CUTOFF.
START_DATE = "2022-01-01"
END_DATE = "2026-09-30"
TRAIN_CUTOFF = "2026-03-31"

# Direct models are trained for every lead from 1 through this horizon so the
# app can cover a review cycle plus a replenishment lead time.
MAX_HORIZON = 42
BACKTEST_HORIZONS = (7, 14, 28)
BACKTEST_STEP_DAYS = 14
PRIMARY_HORIZON = 14

CATEGORIES = (
    "provisions",
    "fresh_water",
    "lubricants",
    "deck_engine_stores",
    "safety_equipment",
)

CATEGORY_LABELS = {
    "provisions": "Provisions",
    "fresh_water": "Fresh water",
    "lubricants": "Lubricants",
    "deck_engine_stores": "Deck & engine stores",
    "safety_equipment": "Safety equipment",
}

# Units of the generated demand series.
UNITS = {
    "provisions": "kg",
    "fresh_water": "m3",
    "lubricants": "L",
    "deck_engine_stores": "units",
    "safety_equipment": "units",
}

# Illustrative basket prices in USD. Used only to show an order value in the
# demo. They are round synthetic figures, not supplier quotes.
UNIT_PRICES_USD = {
    "provisions": 2.35,
    "fresh_water": 12.00,
    "lubricants": 3.40,
    "deck_engine_stores": 55.00,
    "safety_equipment": 95.00,
}

# Default replenishment lead time, in days, by category. Local fresh food is
# short; imported safety gear is long. Stylised, not a purchasing contract.
DEFAULT_LEAD_TIME_DAYS = {
    "provisions": 2,
    "fresh_water": 1,
    "lubricants": 5,
    "deck_engine_stores": 10,
    "safety_equipment": 14,
}

# Monday=0 ... Sunday=6. One closure day (Sunday) for every port. This is a
# desk-calendar choice in the generator, not a claim about real port weekends.
CLOSURE_WEEKDAY = 6

# Stylised mean arrivals per day before month, weekday, trend, and disruptions.
PORTS = {
    "Mombasa": {
        "country": "Kenya",
        "mean_daily_arrivals": 6.0,
        "annual_growth": 0.025,
        "type_mix": {
            "container": 0.40,
            "bulk": 0.16,
            "tanker": 0.12,
            "general_cargo": 0.16,
            "roro": 0.07,
            "passenger": 0.05,
            "fishing": 0.04,
        },
        # Jan..Dec multiplier on the arrival rate. December is busier, the
        # middle of the year is quieter. Stylised shape, not a tide table.
        "month_index": [1.04, 1.02, 0.98, 0.95, 0.90, 0.88, 0.90, 0.93, 1.00, 1.06, 1.10, 1.14],
    },
    "Dar es Salaam": {
        "country": "Tanzania",
        "mean_daily_arrivals": 3.4,
        "annual_growth": 0.030,
        "type_mix": {
            "container": 0.34,
            "bulk": 0.22,
            "tanker": 0.14,
            "general_cargo": 0.18,
            "roro": 0.04,
            "passenger": 0.03,
            "fishing": 0.05,
        },
        "month_index": [1.02, 1.00, 0.97, 0.94, 0.92, 0.90, 0.93, 0.96, 1.02, 1.06, 1.08, 1.10],
    },
    "Djibouti": {
        "country": "Djibouti",
        "mean_daily_arrivals": 4.4,
        "annual_growth": 0.020,
        "type_mix": {
            "container": 0.36,
            "bulk": 0.10,
            "tanker": 0.28,
            "general_cargo": 0.12,
            "roro": 0.06,
            "passenger": 0.02,
            "fishing": 0.06,
        },
        "month_index": [1.06, 1.04, 1.00, 0.96, 0.92, 0.90, 0.92, 0.95, 1.00, 1.04, 1.08, 1.10],
    },
    "Durban": {
        "country": "South Africa",
        "mean_daily_arrivals": 5.2,
        "annual_growth": 0.015,
        "type_mix": {
            "container": 0.42,
            "bulk": 0.18,
            "tanker": 0.14,
            "general_cargo": 0.12,
            "roro": 0.08,
            "passenger": 0.03,
            "fishing": 0.03,
        },
        "month_index": [1.08, 1.06, 1.02, 0.98, 0.96, 0.94, 0.95, 0.97, 1.00, 1.02, 1.04, 1.08],
    },
}

# Weekday multiplier on the arrival rate (Mon..Sun).
DOW_ARRIVAL_INDEX = [1.02, 1.10, 1.14, 1.12, 1.05, 0.84, 0.72]

# lognormal parameters are the mean and sd of log(DWT). Medians are exp(dwt_mu).
# "dwt" is a size index. For passenger ships it is only a bulk figure so the
# same column can describe every call.
VESSEL_PROFILES = {
    "container": dict(dwt_mu=10.45, dwt_sigma=0.35, dwt_min=4_000, dwt_max=200_000, crew_mu=22, crew_sigma=3.0, stay_mu=1.8, voyage_mu=22),
    "bulk": dict(dwt_mu=10.85, dwt_sigma=0.40, dwt_min=8_000, dwt_max=250_000, crew_mu=21, crew_sigma=3.0, stay_mu=3.2, voyage_mu=24),
    "tanker": dict(dwt_mu=11.05, dwt_sigma=0.45, dwt_min=5_000, dwt_max=320_000, crew_mu=24, crew_sigma=3.0, stay_mu=2.0, voyage_mu=28),
    "general_cargo": dict(dwt_mu=9.30, dwt_sigma=0.45, dwt_min=1_500, dwt_max=40_000, crew_mu=16, crew_sigma=3.0, stay_mu=2.6, voyage_mu=18),
    "roro": dict(dwt_mu=9.70, dwt_sigma=0.30, dwt_min=3_000, dwt_max=50_000, crew_mu=20, crew_sigma=3.0, stay_mu=1.3, voyage_mu=14),
    "passenger": dict(dwt_mu=9.10, dwt_sigma=0.55, dwt_min=2_000, dwt_max=80_000, crew_mu=55, crew_sigma=12.0, stay_mu=1.1, voyage_mu=7, pax_mu=6.20, pax_sigma=0.55),
    "fishing": dict(dwt_mu=6.90, dwt_sigma=0.55, dwt_min=80, dwt_max=5_000, crew_mu=14, crew_sigma=4.0, stay_mu=1.8, voyage_mu=16),
}

# Probability that a call places an order with this chandler, by category and
# vessel type. The complement is "stored elsewhere or not required".
ORDER_PROBABILITY = {
    "provisions": {"container": 0.70, "bulk": 0.68, "tanker": 0.66, "general_cargo": 0.74, "roro": 0.69, "passenger": 0.90, "fishing": 0.78},
    "fresh_water": {"container": 0.52, "bulk": 0.50, "tanker": 0.48, "general_cargo": 0.55, "roro": 0.50, "passenger": 0.84, "fishing": 0.60},
    "lubricants": {"container": 0.34, "bulk": 0.42, "tanker": 0.50, "general_cargo": 0.30, "roro": 0.32, "passenger": 0.28, "fishing": 0.22},
    "deck_engine_stores": {"container": 0.24, "bulk": 0.30, "tanker": 0.26, "general_cargo": 0.28, "roro": 0.22, "passenger": 0.18, "fishing": 0.16},
    "safety_equipment": {"container": 0.10, "bulk": 0.11, "tanker": 0.12, "general_cargo": 0.11, "roro": 0.10, "passenger": 0.16, "fishing": 0.09},
}

# Litres of lubricant per thousand DWT if the vessel orders.
LUBE_LITRES_PER_THOUSAND_DWT = {
    "container": 42.0,
    "bulk": 55.0,
    "tanker": 70.0,
    "general_cargo": 36.0,
    "roro": 40.0,
    "passenger": 30.0,
    "fishing": 22.0,
}

# Jan..Dec quantity multiplier, applied inside the order equation.
MONTH_INDEX = {
    "provisions": [1.06, 1.03, 1.00, 0.98, 0.96, 0.95, 0.97, 0.99, 1.02, 1.05, 1.08, 1.14],
    "fresh_water": [1.16, 1.14, 1.10, 1.04, 0.96, 0.90, 0.88, 0.92, 1.00, 1.08, 1.12, 1.15],
    "lubricants": [1.02, 1.08, 1.12, 1.10, 1.00, 0.96, 0.95, 0.97, 1.02, 1.04, 1.00, 0.98],
    "deck_engine_stores": [1.04, 1.10, 1.14, 1.08, 1.00, 0.96, 0.95, 0.98, 1.04, 1.06, 1.02, 1.00],
    "safety_equipment": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
}

# Log-space noise on an order that is placed. Mean of the multiplier is 1.
ORDER_NOISE_SIGMA = {
    "provisions": 0.16,
    "fresh_water": 0.18,
    "lubricants": 0.26,
    "deck_engine_stores": 0.32,
    "safety_equipment": 0.38,
}

# Harbour-craft / walk-in floor on an open day, scaled by port size relative
# to a 6-call/day reference. Closed days keep a small fraction of this floor.
BASELINE_OPEN_DAY = {
    "provisions": 280.0,
    "fresh_water": 9.0,
    "lubricants": 140.0,
    "deck_engine_stores": 1.6,
    "safety_equipment": 0.35,
}
CLOSED_DAY_BASELINE_FRACTION = 0.15

# Shared port-day AR(1) shock on total demand, after orders are summed.
AR_PHI = 0.60
AR_INNOVATION_SIGMA = 0.045

# Months in which more vessels buy safety gear (a stylised survey season).
SAFETY_SURVEY_MONTHS = (3, 9)
SAFETY_SURVEY_PROB_MULTIPLIER = 2.1

# Provisions and water respond to a few calendar events on the arrival date.
RAMADAN_PROVISIONS_FACTOR = 0.90
PRE_EID_PROVISIONS_FACTOR = 1.28
PRE_EID_WATER_FACTOR = 1.10
CHRISTMAS_PROVISIONS_FACTOR = 1.18
CHRISTMAS_WATER_FACTOR = 1.06

# Named, invented disruptions. They are not historical incidents.
DISRUPTIONS = (
    {"port": "Mombasa", "start": "2022-08-15", "end": "2022-08-17", "name": "Synthetic crane outage", "multiplier": 0.35, "rebound": 1.25, "rebound_days": 3},
    {"port": "Dar es Salaam", "start": "2023-03-06", "end": "2023-03-09", "name": "Synthetic channel restriction", "multiplier": 0.40, "rebound": 1.22, "rebound_days": 3},
    {"port": "Durban", "start": "2023-10-02", "end": "2023-10-04", "name": "Synthetic weather closure", "multiplier": 0.30, "rebound": 1.30, "rebound_days": 3},
    {"port": "Djibouti", "start": "2024-06-11", "end": "2024-06-12", "name": "Synthetic berth congestion", "multiplier": 0.45, "rebound": 1.20, "rebound_days": 3},
    {"port": "Mombasa", "start": "2025-01-20", "end": "2025-01-22", "name": "Synthetic labour stoppage", "multiplier": 0.32, "rebound": 1.28, "rebound_days": 3},
    {"port": "Durban", "start": "2026-05-04", "end": "2026-05-05", "name": "Synthetic pilotage delay", "multiplier": 0.40, "rebound": 1.22, "rebound_days": 3},
)

# Conventional civil dates used as synthetic features. Islamic festival dates
# move with lunar sightings; the dates below are fixed inputs to the generator,
# not announcements.
EID_AL_FITR = ("2022-05-02", "2023-04-21", "2024-04-10", "2025-03-30", "2026-03-20")
EID_AL_ADHA = ("2022-07-09", "2023-06-28", "2024-06-16", "2025-06-06", "2026-05-27")
RAMADAN = (
    ("2022-04-02", "2022-05-01"),
    ("2023-03-23", "2023-04-20"),
    ("2024-03-11", "2024-04-09"),
    ("2025-03-01", "2025-03-29"),
    ("2026-02-18", "2026-03-19"),
)

# Good Friday and Easter Monday.
EASTER = (
    ("2022-04-15", "2022-04-18"),
    ("2023-04-07", "2023-04-10"),
    ("2024-03-29", "2024-04-01"),
    ("2025-04-18", "2025-04-21"),
    ("2026-04-03", "2026-04-06"),
)

# Fixed month-day public holidays by port. Eid and Easter are added separately.
FIXED_HOLIDAYS = {
    "Mombasa": (
        (1, 1, "New Year"),
        (5, 1, "Labour Day"),
        (6, 1, "Madaraka Day"),
        (10, 20, "Mashujaa Day"),
        (12, 12, "Jamhuri Day"),
        (12, 25, "Christmas"),
        (12, 26, "Boxing Day"),
    ),
    "Dar es Salaam": (
        (1, 1, "New Year"),
        (1, 12, "Zanzibar Revolution Day"),
        (4, 26, "Union Day"),
        (5, 1, "Labour Day"),
        (7, 7, "Saba Saba"),
        (8, 8, "Nane Nane"),
        (12, 9, "Independence Day"),
        (12, 25, "Christmas"),
        (12, 26, "Boxing Day"),
    ),
    "Djibouti": (
        (1, 1, "New Year"),
        (5, 1, "Labour Day"),
        (6, 27, "Independence Day"),
        (12, 25, "Christmas"),
    ),
    "Durban": (
        (1, 1, "New Year"),
        (3, 21, "Human Rights Day"),
        (4, 27, "Freedom Day"),
        (5, 1, "Workers' Day"),
        (6, 16, "Youth Day"),
        (8, 9, "Women's Day"),
        (9, 24, "Heritage Day"),
        (12, 16, "Day of Reconciliation"),
        (12, 25, "Christmas"),
        (12, 26, "Day of Goodwill"),
    ),
}

# Gradient boosting hyperparameters for the full training run.
HGB_PARAMS = dict(
    loss="squared_error",
    learning_rate=0.08,
    max_iter=220,
    max_leaf_nodes=63,
    min_samples_leaf=40,
    l2_regularization=0.1,
    early_stopping=False,
    random_state=SEED,
)

RIDGE_ALPHA = 6.0

MODEL_LABELS = {
    "seasonal_naive": "Seasonal naive (7-day)",
    "holt_winters": "Holt-Winters (weekly)",
    "fourier_ridge": "Fourier ridge",
    "hgb_history": "Gradient boosting",
    "hgb_lineup": "Gradient boosting + lineup",
}

LINEUP_COLUMNS = (
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
