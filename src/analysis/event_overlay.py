"""Daftar event temporal untuk overlay di plot time-series."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Event:
    name: str
    date: pd.Timestamp
    kind: str  # 'patch', 'ti', 'pandemic_start', 'pandemic_end', 'dpc_era'


def major_patches() -> list[Event]:
    """Patch major Dota 2 yang relevan untuk dekade 2016-2026."""
    rows = [
        ("7.00", "2016-12-12"),
        ("7.06", "2017-05-15"),
        ("7.20", "2018-11-19"),
        ("7.22", "2019-05-02"),
        ("7.23", "2019-11-26"),
        ("7.27", "2020-08-18"),
        ("7.29", "2021-04-09"),
        ("7.30", "2021-08-18"),
        ("7.31", "2022-02-23"),
        ("7.32", "2022-08-24"),
        ("7.33", "2023-04-20"),
        ("7.34", "2023-08-08"),
        ("7.35", "2023-12-14"),
        ("7.36", "2024-05-22"),
        ("7.37", "2024-08-01"),
    ]
    return [Event(name=n, date=pd.Timestamp(d), kind="patch") for n, d in rows]


def the_internationals() -> list[Event]:
    """The International tahunan."""
    rows = [
        ("TI6", "2016-08-13"),
        ("TI7", "2017-08-12"),
        ("TI8", "2018-08-25"),
        ("TI9", "2019-08-25"),
        # TI10 ditunda ke 2021 karena pandemi
        ("TI10", "2021-10-17"),
        ("TI11", "2022-10-30"),
        ("TI12", "2023-10-29"),
        ("TI13", "2024-09-15"),
    ]
    return [Event(name=n, date=pd.Timestamp(d), kind="ti") for n, d in rows]


def pandemic_band() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Band era pandemi COVID-19 daring (terapan ke turnamen Dota 2)."""
    return pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31")


def dpc_era() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Era DPC formal: 2017 sd 2022 (Valve mengakhiri DPC setelah TI11)."""
    return pd.Timestamp("2017-01-01"), pd.Timestamp("2022-12-31")
