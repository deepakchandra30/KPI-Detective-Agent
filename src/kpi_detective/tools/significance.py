"""significance_test: is a segment's change beyond its normal variation?

For every value of a dimension we compute a daily statistic

    x_d = log((y_d + 1) / (baseline_d + 1))

where ``baseline_d`` is the same-weekday mean of the 4 previous weeks. The
window's statistics are compared with the same statistic on the preceding
``HISTORY_DAYS`` days (the null distribution) using a one-sided
Mann-Whitney U test. History days whose own baseline touches a known
calendar event are left out, so holiday swings do not widen the null
distribution. Because one- and two-day windows give Mann-Whitney
very little resolution, a robust z-test (median / MAD) is used when the
window has fewer than 3 days. p-values are corrected across tested values
with Benjamini-Hochberg.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import stats

from kpi_detective.config import BASELINE_WEEKS
from kpi_detective.data.calendar import events_between
from kpi_detective.store import build_where
from kpi_detective.tools.base import (
    ToolContext,
    ToolError,
    baseline_dates_for,
    r,
    validate_dimension,
    validate_filters,
    validate_metric,
    validate_window,
    window_dates,
)

HISTORY_DAYS = 365
ALPHA = 0.05
MIN_WINDOW_FOR_MWU = 3


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """BH-adjusted q-values, in the input order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out.tolist()


def significance_test(
    ctx: ToolContext,
    metric: str = "revenue",
    dimension: str = "region",
    filters: dict | None = None,
    values: list[str] | None = None,
) -> dict:
    validate_window(ctx)
    validate_metric(metric)
    validate_dimension(ctx, dimension)
    filters = validate_filters(ctx, filters)
    allowed = ctx.store.dimension_values[dimension]
    if values:
        bad = [v for v in values if v not in allowed]
        if bad:
            raise ToolError(f"invalid values for {dimension}: {bad}")

    ws, we = ctx.anomaly.start, ctx.anomaly.end
    lo = ws - timedelta(days=HISTORY_DAYS + 7 * BASELINE_WEEKS)
    lo = max(lo, ctx.store.date_bounds[0])
    where, params = build_where(filters, extra=["date BETWEEN ? AND ?"])
    df = ctx.store.query(
        f"SELECT date, {dimension} AS v, SUM({metric}) AS y FROM {ctx.store.table} {where} GROUP BY 1, 2",
        [lo, we, *params],
    )
    wide = df.pivot_table(index="date", columns="v", values="y", aggfunc="sum").fillna(0.0)
    wide.index = pd.to_datetime(wide.index)
    wide = wide.reindex(pd.date_range(lo, we), fill_value=0.0)
    tested = values or [str(c) for c in wide.columns]

    win = window_dates(ws, we)
    hist_days = [
        d
        for d in pd.date_range(ws - timedelta(days=HISTORY_DAYS), ws - timedelta(days=1)).date
        if d - timedelta(days=7 * BASELINE_WEEKS) >= lo
        and not events_between(d - timedelta(days=7 * BASELINE_WEEKS), d)
    ]
    if len(hist_days) < 28:
        raise ToolError("not enough history before the anomaly window for a significance test")

    def stat(series: pd.Series, d, base_days) -> float:
        base = np.mean([series.get(pd.Timestamp(b), 0.0) for b in base_days])
        return float(np.log((series.get(pd.Timestamp(d), 0.0) + 1) / (base + 1)))

    results = []
    for v in tested:
        s = wide[v] if v in wide.columns else pd.Series(0.0, index=wide.index)
        w_stats = np.array([stat(s, d, baseline_dates_for(d, ws)) for d in win])
        h_stats = np.array(
            [stat(s, d, [d - timedelta(days=7 * k) for k in range(1, BASELINE_WEEKS + 1)]) for d in hist_days]
        )
        med = float(np.median(h_stats))
        mad = float(stats.median_abs_deviation(h_stats, scale="normal")) or 1e-9
        z = (float(np.mean(w_stats)) - med) / (mad / np.sqrt(len(w_stats)))
        if len(w_stats) >= MIN_WINDOW_FOR_MWU:
            p = float(stats.mannwhitneyu(w_stats, h_stats, alternative="less").pvalue)
            method = "mann_whitney_u"
        else:
            p = float(stats.norm.cdf(z))
            method = "robust_z"
        results.append(
            {
                "value": v,
                "window_mean_log_ratio": r(np.mean(w_stats)),
                "window_pct_change": r(np.expm1(np.mean(w_stats))),
                "history_median_log_ratio": r(med),
                "history_robust_sd": r(mad),
                "z_score": r(z, 3),
                "p_value": r(p, 6),
                "method": method,
            }
        )
    qs = benjamini_hochberg([x["p_value"] for x in results])
    for x, q in zip(results, qs, strict=False):
        x["q_value"] = r(q, 6)
        x["significant"] = bool(q < ALPHA and x["z_score"] < 0)
    results.sort(key=lambda x: x["p_value"])
    return {
        "metric": metric,
        "dimension": dimension,
        "filters": filters,
        "history_days": len(hist_days),
        "correction": "benjamini_hochberg",
        "alpha": ALPHA,
        "results": results,
    }
