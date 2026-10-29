"""Generator invariants. The series is synthetic and seed-stable."""

import pandas as pd

from chandlery_forecast.generate import booking_date, generate_dataset


def test_january_2022_mombasa_provisions_checksum():
    # Locks the generator. Update this only when the demand equations change on purpose.
    book = generate_dataset(start="2022-01-01", end="2022-01-31", ports=("Mombasa",), categories=("provisions",))
    assert round(float(book.daily_demand["demand"].sum()), 1) == 163133.4
    assert len(book.vessel_calls) == 182


def test_repeat_runs_match():
    first = generate_dataset(start="2022-01-01", end="2022-02-28", ports=("Mombasa",), categories=("provisions", "fresh_water"))
    second = generate_dataset(start="2022-01-01", end="2022-02-28", ports=("Mombasa",), categories=("provisions", "fresh_water"))
    pd.testing.assert_frame_equal(first.daily_demand, second.daily_demand)
    pd.testing.assert_frame_equal(first.vessel_calls, second.vessel_calls)


def test_demand_panel_is_complete_and_non_negative():
    book = generate_dataset(start="2022-03-01", end="2022-04-15", ports=("Durban", "Djibouti"), categories=("lubricants", "safety_equipment"))
    expected = 46 * 2 * 2
    assert len(book.daily_demand) == expected
    assert book.daily_demand["demand"].min() >= 0
    assert set(book.daily_demand["port"]) == {"Durban", "Djibouti"}
    assert book.metadata["disclaimer"].lower().startswith("synthetic")


def test_orders_are_not_booked_on_a_closed_desk():
    book = generate_dataset(start="2022-01-01", end="2022-03-31", ports=("Mombasa", "Dar es Salaam"), categories=("provisions",))
    closed = book.holidays.loc[book.holidays["is_closed"], ["date", "port"]]
    booked = book.orders.merge(closed, left_on=["order_date", "port"], right_on=["date", "port"], how="left", indicator=True)
    assert (booked["_merge"] == "left_only").all()
    assert (book.orders["order_date"].dt.day_of_week != 6).all()


def test_booking_date_skips_sunday_and_a_holiday():
    # 1 Jan 2022 was a Saturday and a public holiday; 2 Jan was a Sunday.
    closed = set(pd.to_datetime(["2022-01-01", "2022-01-02"]))
    assert booking_date(pd.Timestamp("2022-01-01"), closed) == pd.Timestamp("2022-01-03")
    assert booking_date(pd.Timestamp("2022-01-04"), closed) == pd.Timestamp("2022-01-04")


def test_christmas_week_flag_is_limited_to_late_december():
    book = generate_dataset(start="2022-12-15", end="2022-12-31", ports=("Mombasa",), categories=("provisions",))
    flags = book.holidays.drop_duplicates("date").set_index("date")["is_christmas_week"]
    assert bool(flags.loc[pd.Timestamp("2022-12-17")]) is False
    assert bool(flags.loc[pd.Timestamp("2022-12-18")]) is True
    assert bool(flags.loc[pd.Timestamp("2022-12-25")]) is True
