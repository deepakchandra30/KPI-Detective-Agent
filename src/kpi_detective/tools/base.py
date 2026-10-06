"""Shared plumbing for the statistical toolkit.

Baseline convention (used by every tool): each anomaly day ``d`` is compared
with the mean of the same weekday in the 4 weeks *before the anomaly
window starts*. This keeps multi-day windows from contaminating their own
baseline (a naive ``d - 7`` would fall inside a 10-day window).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from kpi_detective.config import ADDITIVE_METRICS, BASELINE_WEEKS, DIMENSIONS
from kpi_detective.models import AnomalyEvent
from kpi_detective.store import SalesStore, build_where


class ToolError(ValueError):
    """Invalid tool input. Returned to the agent as a structured error."""


@dataclass
class ToolContext:
    store: SalesStore
    anomaly: AnomalyEvent

    @property
    def window(self) -> list[date]:
        return window_dates(self.anomaly.start, self.anomaly.end)


# ---------------------------------------------------------------- validation
def validate_metric(metric: str) -> str:
    if metric not in ADDITIVE_METRICS:
        raise ToolError(f"metric must be one of {ADDITIVE_METRICS}, got {metric!r}")
    return metric


def validate_dimension(ctx: ToolContext, dim: str) -> str:
    if dim not in DIMENSIONS:
        raise ToolError(f"dimension must be one of {DIMENSIONS}, got {dim!r}")
    return dim


def validate_filters(ctx: ToolContext, filters: dict | None) -> dict[str, str]:
    filters = dict(filters or {})
    for dim, val in filters.items():
        validate_dimension(ctx, dim)
        allowed = ctx.store.dimension_values[dim]
        if str(val) not in allowed:
            raise ToolError(f"{val!r} is not a valid value for {dim}; allowed: {allowed}")
    return {k: str(v) for k, v in filters.items()}


def validate_window(ctx: ToolContext) -> None:
    lo, hi = ctx.store.date_bounds
    a = ctx.anomaly
    if a.start > a.end:
        raise ToolError("anomaly start is after end")
    if a.start < lo + timedelta(days=7 * BASELINE_WEEKS) or a.end > hi:
        raise ToolError(f"anomaly window must lie within [{lo + timedelta(days=28)}, {hi}]")


# ------------------------------------------------------------------- periods
def window_dates(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def baseline_dates_for(day: date, window_start: date, weeks: int = BASELINE_WEEKS) -> list[date]:
    """Same weekday in the ``weeks`` weeks preceding ``window_start``."""
    k0 = math.ceil(((day - window_start).days + 1) / 7)
    return [day - timedelta(days=7 * k) for k in range(k0, k0 + weeks)]


def period_weights(start: date, end: date, weeks: int = BASELINE_WEEKS) -> tuple[list[date], dict[date, float]]:
    """Window dates and baseline-date weights (a baseline date can serve two window days)."""
    win = window_dates(start, end)
    counts: Counter[date] = Counter()
    for d in win:
        counts.update(baseline_dates_for(d, start, weeks))
    return win, {d: c / weeks for d, c in counts.items()}


def aggregate(
    ctx: ToolContext,
    metrics: list[str],
    dims: list[str],
    filters: dict[str, str],
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Current-period vs baseline totals grouped by ``dims``.

    Returns one row per dims combination with ``<metric>_current`` and
    ``<metric>_baseline`` columns.
    """
    start = start or ctx.anomaly.start
    end = end or ctx.anomaly.end
    win, weights = period_weights(start, end)
    all_dates = sorted(set(win) | set(weights))
    placeholders = ", ".join("?" for _ in all_dates)
    where, params = build_where(filters, extra=[f"date IN ({placeholders})"])
    params = [*all_dates, *params]
    group_cols = ["date", *dims]
    select_metrics = ", ".join(f"SUM({m}) AS {m}" for m in metrics)
    sql = (
        f"SELECT {', '.join(group_cols)}, {select_metrics} FROM {ctx.store.table} "
        f"{where} GROUP BY {', '.join(group_cols)}"
    )
    df = ctx.store.query(sql, params)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    win_set = set(win)
    df["_cur_w"] = df["date"].map(lambda d: 1.0 if d in win_set else 0.0)
    df["_base_w"] = df["date"].map(lambda d: weights.get(d, 0.0))
    out_cols = {}
    for m in metrics:
        vals = df[m].astype(float).fillna(0.0)
        df[f"{m}_current"] = vals * df["_cur_w"]
        df[f"{m}_baseline"] = vals * df["_base_w"]
        out_cols[f"{m}_current"] = "sum"
        out_cols[f"{m}_baseline"] = "sum"
    res = df.groupby(dims, as_index=False).agg(out_cols) if dims else pd.DataFrame([{c: df[c].sum() for c in out_cols}])
    return res


def r(x: float | None, nd: int = 4) -> float | None:
    """Round for JSON output; keeps NaN/inf out of tool results."""
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None
    return round(x, nd)


def pct(cur: float, base: float) -> float | None:
    return (cur - base) / base if base else None
