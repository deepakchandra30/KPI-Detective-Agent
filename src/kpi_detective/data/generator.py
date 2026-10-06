"""Synthetic daily sales data for a fictional electronics company.

Grain: one row per day x country x channel x product category x customer
segment (11 x 3 x 4 x 3 = 396 cells per day).

Realism built in (PRD section 5.1):

* weekly seasonality: weekend lift for Consumer, weekend dip for B2B
* yearly seasonality from :mod:`kpi_detective.data.calendar` (Black Friday,
  holiday season, post-holiday lull, back to school, quarter-end pushes)
* long-term growth trend and slow laptop price erosion
* negative binomial order counts (overdispersed, lumpier for Enterprise),
  plus correlated daily shocks at global and regional level

Everything is driven by a single seed, so a scenario is reproducible from
its JSON spec alone and no large data files need to be stored.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from kpi_detective.config import CATEGORIES, CHANNELS, COUNTRY_TO_REGION, SEGMENTS
from kpi_detective.data.calendar import black_friday

DEFAULT_START = "2023-01-01"
DEFAULT_END = "2025-12-31"

TOTAL_DAILY_ORDERS = 20_000

COUNTRY_WEIGHT: dict[str, float] = {
    "US": 0.40 * 0.85,
    "Canada": 0.40 * 0.15,
    "Ireland": 0.30 * 0.08,
    "Germany": 0.30 * 0.35,
    "UK": 0.30 * 0.35,
    "France": 0.30 * 0.22,
    "Japan": 0.20 * 0.45,
    "Australia": 0.20 * 0.25,
    "India": 0.20 * 0.30,
    "Brazil": 0.10 * 0.60,
    "Mexico": 0.10 * 0.40,
}
SEGMENT_ORDER_SHARE = {"Consumer": 0.60, "Small Business": 0.25, "Enterprise": 0.15}
CHANNEL_MIX = {
    "Consumer": {"Online": 0.60, "Retail Partner": 0.35, "Direct Sales": 0.05},
    "Small Business": {"Online": 0.40, "Retail Partner": 0.30, "Direct Sales": 0.30},
    "Enterprise": {"Online": 0.10, "Retail Partner": 0.10, "Direct Sales": 0.80},
}
CATEGORY_MIX = {
    "Consumer": {"Laptops": 0.35, "Monitors": 0.22, "Servers": 0.005, "Accessories": 0.425},
    "Small Business": {"Laptops": 0.35, "Monitors": 0.27, "Servers": 0.05, "Accessories": 0.33},
    "Enterprise": {"Laptops": 0.40, "Monitors": 0.30, "Servers": 0.10, "Accessories": 0.20},
}
#: Extra units per order by segment; servers ship in much smaller quantities.
EXTRA_UNITS = {"Consumer": 0.1, "Small Business": 1.0, "Enterprise": 5.0}
SERVER_UNIT_FACTOR = 0.2
LIST_PRICE = {"Laptops": 1100.0, "Monitors": 320.0, "Servers": 6500.0, "Accessories": 45.0}
SEGMENT_PRICE_FACTOR = {"Consumer": 1.00, "Small Business": 0.95, "Enterprise": 0.85}
NB_DISPERSION = {"Consumer": 150.0, "Small Business": 100.0, "Enterprise": 60.0}
#: Sessions per order (inverse conversion) by channel.
SESSIONS_PER_ORDER = {"Online": 40.0, "Retail Partner": 12.0, "Direct Sales": 4.0}
WEEKEND_FACTOR = {"Consumer": 1.25, "Small Business": 0.50, "Enterprise": 0.20}
ANNUAL_GROWTH = 0.07
LAPTOP_PRICE_EROSION = 0.02


def cell_frame() -> pd.DataFrame:
    """All 396 dimension combinations, with the base daily order rate."""
    rows = []
    for country, cw in COUNTRY_WEIGHT.items():
        for seg in SEGMENTS:
            for ch in CHANNELS:
                for cat in CATEGORIES:
                    rate = (
                        TOTAL_DAILY_ORDERS
                        * cw
                        * SEGMENT_ORDER_SHARE[seg]
                        * CHANNEL_MIX[seg][ch]
                        * CATEGORY_MIX[seg][cat]
                    )
                    rows.append(
                        {
                            "region": COUNTRY_TO_REGION[country],
                            "country": country,
                            "channel": ch,
                            "product_category": cat,
                            "customer_segment": seg,
                            "base_orders": rate,
                        }
                    )
    return pd.DataFrame(rows)


def _yearly_multiplier(dates: pd.DatetimeIndex, cells: pd.DataFrame) -> np.ndarray:
    """(n_days, n_cells) multiplicative yearly-seasonality and event effects."""
    n_days, n_cells = len(dates), len(cells)
    mult = np.ones((n_days, n_cells))
    seg = cells["customer_segment"].to_numpy()
    cat = cells["product_category"].to_numpy()
    is_cons = seg == "Consumer"
    is_sb = seg == "Small Business"
    is_ent = seg == "Enterprise"

    month = dates.month.to_numpy()
    day = dates.day.to_numpy()
    doy_md = month * 100 + day  # e.g. 1115 for 15 Nov

    def rows(mask: np.ndarray) -> np.ndarray:
        return np.where(mask)[0]

    # Holiday season (15 Nov - 23 Dec)
    hol = rows((doy_md >= 1115) & (doy_md <= 1223))
    mult[np.ix_(hol, rows(is_cons))] *= 1.25
    mult[np.ix_(hol, rows(is_sb))] *= 1.05

    # Black Friday to Cyber Monday
    bf_mask = np.zeros(n_days, dtype=bool)
    for y in sorted(set(dates.year)):
        bf = pd.Timestamp(black_friday(y))
        bf_mask |= (dates >= bf) & (dates <= bf + pd.Timedelta(days=3))
    mult[np.ix_(rows(bf_mask), rows(is_cons))] *= 1.8

    # Post-holiday lull (26 Dec - 12 Jan)
    lull = rows((doy_md >= 1226) | (doy_md <= 112))
    mult[np.ix_(lull, rows(is_cons))] *= 0.70
    mult[np.ix_(lull, rows(is_sb))] *= 0.75
    mult[np.ix_(lull, rows(is_ent))] *= 0.60

    # Back to school (1 Aug - 10 Sep)
    bts = rows((doy_md >= 801) & (doy_md <= 910))
    mult[np.ix_(bts, rows(is_cons & (cat == "Laptops")))] *= 1.30
    mult[np.ix_(bts, rows(is_cons & (cat != "Laptops")))] *= 1.10

    # Summer enterprise slowdown (Jul - Aug)
    summer = rows((month == 7) | (month == 8))
    mult[np.ix_(summer, rows(is_ent))] *= 0.90

    # Quarter-end push: last 7 days of Mar, Jun, Sep, Dec
    month_end = (dates + pd.offsets.MonthEnd(0)).day.to_numpy()
    qe = rows(np.isin(month, [3, 6, 9, 12]) & (day > month_end - 7))
    mult[np.ix_(qe, rows(is_ent))] *= 1.60
    mult[np.ix_(qe, rows(is_sb))] *= 1.15
    return mult


def _generate(start: str, end: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    cells = cell_frame()
    n_days, n_cells = len(dates), len(cells)
    seg = cells["customer_segment"].to_numpy()
    cat = cells["product_category"].to_numpy()
    ch = cells["channel"].to_numpy()
    region = cells["region"].to_numpy()

    t_years = (dates - dates[0]).days.to_numpy() / 365.25
    trend = (1 + ANNUAL_GROWTH) ** t_years

    weekend = dates.dayofweek.to_numpy() >= 5
    weekly = np.ones((n_days, n_cells))
    for s, f in WEEKEND_FACTOR.items():
        weekly[np.ix_(np.where(weekend)[0], np.where(seg == s)[0])] = f

    yearly = _yearly_multiplier(dates, cells)

    # Correlated daily shocks: global and per-region
    global_shock = rng.lognormal(0.0, 0.008, size=n_days)
    region_names = sorted(set(region))
    region_shock = {r: rng.lognormal(0.0, 0.006, size=n_days) for r in region_names}
    shock = global_shock[:, None] * np.column_stack([region_shock[r] for r in region])

    mu = cells["base_orders"].to_numpy()[None, :] * trend[:, None] * weekly * yearly * shock

    # Negative binomial counts, dispersion by segment
    r = np.array([NB_DISPERSION[s] for s in seg])[None, :]
    p = r / (r + mu)
    orders = rng.negative_binomial(r, p)

    extra = np.array(
        [EXTRA_UNITS[s] * (SERVER_UNIT_FACTOR if c == "Servers" else 1.0) for s, c in zip(seg, cat, strict=False)]
    )[None, :]
    units = orders + rng.poisson(orders * extra)

    base_price = np.array([LIST_PRICE[c] * SEGMENT_PRICE_FACTOR[s] for s, c in zip(seg, cat, strict=False)])
    erosion = np.where(cat == "Laptops", (1 - LAPTOP_PRICE_EROSION) ** t_years[:, None], 1.0)
    price = base_price[None, :] * erosion * rng.lognormal(0.0, 0.01, size=(n_days, n_cells))

    spo = np.array([SESSIONS_PER_ORDER[c] for c in ch])[None, :]
    sessions = rng.poisson(mu * spo)

    frame = pd.DataFrame(
        {
            "date": np.repeat(dates.date, n_cells),
            "region": np.tile(region, n_days),
            "country": np.tile(cells["country"].to_numpy(), n_days),
            "channel": np.tile(ch, n_days),
            "product_category": np.tile(cat, n_days),
            "customer_segment": np.tile(seg, n_days),
            "sessions": sessions.ravel().astype("int64"),
            "orders": orders.ravel().astype("int64"),
            "units": units.ravel().astype("int64"),
            "unit_price": np.round(price.ravel(), 2),
        }
    )
    frame["revenue"] = np.round(frame["units"] * frame["unit_price"], 2)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


@lru_cache(maxsize=4)
def _cached(start: str, end: str, seed: int) -> pd.DataFrame:
    return _generate(start, end, seed)


def generate_sales(start: str | date = DEFAULT_START, end: str | date = DEFAULT_END, seed: int = 0) -> pd.DataFrame:
    """Generate the clean (incident-free) dataset. Returns a fresh copy."""
    return _cached(str(start), str(end), int(seed)).copy()


def daily_total(frame: pd.DataFrame, metric: str = "revenue") -> pd.Series:
    """Daily total of an additive metric, indexed by date."""
    return frame.groupby("date")[metric].sum().sort_index()
