"""check_data_quality: is the data itself broken? (forced first step, FR6)

Checks, all relative to the same 4-week weekday baseline:

* row counts per day (partial or late loads)
* NULL rates of metric columns
* segments whose revenue is suddenly all zero or NULL (tracking bugs), at the
  level of single dimensions and dimension pairs, reported at the coarsest
  level that explains the problem
* sessions recorded but no orders (a classic tracking-pixel failure)
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from kpi_detective.config import COUNTRY_TO_REGION, DIMENSIONS
from kpi_detective.tools.base import (
    ToolContext,
    aggregate,
    baseline_dates_for,
    period_weights,
    r,
    validate_window,
)

PARTIAL_LOAD_RATIO = 0.95
ZERO_SHARE_WINDOW = 0.90
ZERO_SHARE_BASELINE = 0.30
#: Region is implied by country, so the region x country pair is never checked.
_PAIRS = [p for p in combinations(DIMENSIONS, 2) if set(p) != {"region", "country"}]


def _frame(ctx: ToolContext) -> tuple[pd.DataFrame, list, dict]:
    win, weights = period_weights(ctx.anomaly.start, ctx.anomaly.end)
    dates = sorted(set(win) | set(weights))
    ph = ", ".join("?" for _ in dates)
    df = ctx.store.query(
        f"SELECT date, {', '.join(DIMENSIONS)}, revenue, orders, units, sessions "
        f"FROM {ctx.store.table} WHERE date IN ({ph})",
        dates,
    )
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["period"] = np.where(df["date"].isin(win), "window", "baseline")
    df["missing_revenue"] = df["revenue"].isna() | (df["revenue"] == 0)
    return df, win, weights


def check_data_quality(ctx: ToolContext) -> dict:
    validate_window(ctx)
    df, win, weights = _frame(ctx)
    total = aggregate(ctx, ["revenue"], [], {}).iloc[0]
    total_delta = float(total["revenue_current"] - total["revenue_baseline"])
    flags: list[dict] = []

    # --- row counts and freshness -------------------------------------------
    rows_by_day = df.groupby("date").size()
    base_rows = float(np.median([rows_by_day.get(d, 0) for d in weights])) if weights else 0.0
    day_totals = aggregate_by_day(ctx)
    row_counts = []
    for d in win:
        n = int(rows_by_day.get(d, 0))
        ratio = n / base_rows if base_rows else None
        row_counts.append({"date": d.isoformat(), "rows": n, "expected_rows": r(base_rows, 1), "row_ratio": r(ratio)})
        if ratio is not None and ratio < PARTIAL_LOAD_RATIO:
            delta = day_totals.get(d, 0.0)
            flags.append(
                {
                    "type": "partial_load",
                    "segment": {},
                    "dates": [d.isoformat()],
                    "row_ratio": r(ratio),
                    "revenue_delta": r(delta, 2),
                    "share_of_total_change": r(delta / total_delta) if total_delta else None,
                }
            )

    # --- NULL rates -----------------------------------------------------------
    null_rates = {}
    for col in ("revenue", "orders", "units", "sessions"):
        w = df.loc[df["period"] == "window", col].isna().mean()
        b = df.loc[df["period"] == "baseline", col].isna().mean()
        null_rates[col] = {"window": r(w), "baseline": r(b)}

    # --- zero / NULL filled segments -----------------------------------------
    seg_flags = _zero_filled_segments(df)
    for seg in seg_flags:
        agg = aggregate(ctx, ["revenue"], [], seg["segment"]).iloc[0]
        delta = float(agg["revenue_current"] - agg["revenue_baseline"])
        seg["revenue_delta"] = r(delta, 2)
        seg["share_of_total_change"] = r(delta / total_delta) if total_delta else None
        flags.append(seg)

    # --- sessions without orders ---------------------------------------------
    win_df = df[df["period"] == "window"]
    base_df = df[df["period"] == "baseline"]
    no_orders_w = float(((win_df["sessions"] > 0) & (win_df["orders"].fillna(0) == 0)).mean())
    no_orders_b = float(((base_df["sessions"] > 0) & (base_df["orders"].fillna(0) == 0)).mean())

    flags.sort(key=lambda f: -(abs(f.get("share_of_total_change") or 0)))
    return {
        "status": "issues_found" if flags else "clean",
        "flags": flags,
        "row_counts": row_counts,
        "null_rates": null_rates,
        "sessions_without_orders_share": {"window": r(no_orders_w), "baseline": r(no_orders_b)},
        "latest_date": ctx.store.date_bounds[1].isoformat(),
        "total_change": r(total_delta, 2),
    }


def aggregate_by_day(ctx: ToolContext) -> dict:
    """Revenue delta vs its own weekday baseline, for each window day."""
    daily = ctx.store.daily_total("revenue")
    out = {}
    for d in pd.date_range(ctx.anomaly.start, ctx.anomaly.end).date:
        base = [daily.get(pd.Timestamp(b), np.nan) for b in baseline_dates_for(d, ctx.anomaly.start)]
        out[d] = float(daily.get(pd.Timestamp(d), 0.0) - np.nanmean(base))
    return out


def _zero_filled_segments(df: pd.DataFrame) -> list[dict]:
    """Segments whose revenue went (almost) entirely zero/NULL in the window."""
    found: list[dict] = []
    singles: set[tuple[str, str]] = set()
    groups: list[tuple[str, ...]] = [(d,) for d in DIMENSIONS] + _PAIRS
    for dims in groups:
        g = df.groupby([*dims, "period"])["missing_revenue"].agg(["mean", "size"]).unstack("period")
        if "window" not in g.columns.get_level_values(1):
            continue
        for key, row in g.iterrows():
            key = key if isinstance(key, tuple) else (key,)
            w_share, b_share = row[("mean", "window")], row.get(("mean", "baseline"), np.nan)
            n_w = row[("size", "window")]
            if pd.isna(w_share) or n_w < 2:
                continue
            if w_share >= ZERO_SHARE_WINDOW and (pd.isna(b_share) or b_share <= ZERO_SHARE_BASELINE):
                seg = dict(zip(dims, key, strict=False))
                if len(dims) == 2 and any((d, v) in singles for d, v in seg.items()):
                    continue  # already explained at a coarser level
                if len(dims) == 1:
                    singles.add((dims[0], key[0]))
                found.append(
                    {
                        "type": "zero_or_null_revenue",
                        "segment": seg,
                        "window_missing_share": r(w_share),
                        "baseline_missing_share": r(b_share),
                    }
                )
    # keep only the coarsest explanation: drop flags covered by another flag
    return [f for f in found if not any(g is not f and covers(g["segment"], f["segment"]) for g in found)]


def covers(parent: dict[str, str], child: dict[str, str]) -> bool:
    """True if every row of ``child`` segment is inside ``parent`` segment."""
    for dim, val in parent.items():
        if child.get(dim) == val:
            continue
        if dim == "region" and COUNTRY_TO_REGION.get(child.get("country", "")) == val:
            continue
        return False
    return True
