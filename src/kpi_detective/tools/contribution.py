"""compare_periods, contribution_analysis and drill_down.

``share_of_total_change`` is always relative to the change of the whole KPI,
so numbers from different drill levels can be compared directly.
``concentration`` is share of the parent's change divided by share of the
parent's baseline: 1.0 means the segment fell in proportion to its size,
well above 1.0 means the drop is concentrated there. ``rest_pct_change`` is
the change of the parent *without* this segment; it separates a dominant
child that really is the problem (US -51% while Canada is flat) from one
that merely moves with its siblings (US and Canada both -60%).
"""

from __future__ import annotations

from kpi_detective.config import ADDITIVE_METRICS
from kpi_detective.tools.base import (
    ToolContext,
    ToolError,
    aggregate,
    pct,
    r,
    validate_dimension,
    validate_filters,
    validate_metric,
    validate_window,
)

MAX_ROWS = 12


def _total_delta(ctx: ToolContext, metric: str) -> float:
    t = aggregate(ctx, [metric], [], {}).iloc[0]
    return float(t[f"{metric}_current"] - t[f"{metric}_baseline"])


def compare_periods(ctx: ToolContext, metric: str = "revenue", dimensions: list[str] | None = None) -> dict:
    """Anomaly window vs baseline for all metrics, optionally grouped by dimensions."""
    validate_window(ctx)
    validate_metric(metric)
    dimensions = [validate_dimension(ctx, d) for d in (dimensions or [])]
    if len(dimensions) > 2:
        raise ToolError("compare_periods accepts at most 2 dimensions")
    tot = aggregate(ctx, ADDITIVE_METRICS, [], {}).iloc[0]
    totals = {}
    for m in ADDITIVE_METRICS:
        b, c = float(tot[f"{m}_baseline"]), float(tot[f"{m}_current"])
        totals[m] = {"baseline": r(b, 2), "current": r(c, 2), "delta": r(c - b, 2), "pct_change": r(pct(c, b))}
    ratios = {
        "avg_unit_price": _ratio(tot, "revenue", "units"),
        "conversion_rate": _ratio(tot, "orders", "sessions", nd=5),
        "units_per_order": _ratio(tot, "units", "orders"),
    }
    out: dict = {"metric": metric, "dimensions": dimensions, "totals": totals, "ratios": ratios}
    if dimensions:
        g = aggregate(ctx, [metric], dimensions, {})
        g["delta"] = g[f"{metric}_current"] - g[f"{metric}_baseline"]
        total_delta = totals[metric]["delta"] or 0.0
        g = g.reindex(g["delta"].abs().sort_values(ascending=False).index).head(15)
        out["rows"] = [
            {
                "segment": {d: row[d] for d in dimensions},
                "baseline": r(row[f"{metric}_baseline"], 2),
                "current": r(row[f"{metric}_current"], 2),
                "delta": r(row["delta"], 2),
                "pct_change": r(pct(row[f"{metric}_current"], row[f"{metric}_baseline"])),
                "share_of_total_change": r(row["delta"] / total_delta) if total_delta else None,
            }
            for _, row in g.iterrows()
        ]
    return out


def _ratio(tot, num: str, den: str, nd: int = 4) -> dict:
    b = tot[f"{num}_baseline"] / tot[f"{den}_baseline"] if tot[f"{den}_baseline"] else None
    c = tot[f"{num}_current"] / tot[f"{den}_current"] if tot[f"{den}_current"] else None
    return {"baseline": r(b, nd), "current": r(c, nd), "pct_change": r(pct(c, b)) if b and c else None}


def contribution_analysis(
    ctx: ToolContext, metric: str = "revenue", dimension: str = "region", filters: dict | None = None
) -> dict:
    """Which values of ``dimension`` explain the change (within ``filters``)."""
    validate_window(ctx)
    validate_metric(metric)
    validate_dimension(ctx, dimension)
    filters = validate_filters(ctx, filters)
    if dimension in filters:
        raise ToolError(f"{dimension} is already fixed by the filters")

    total_delta = _total_delta(ctx, metric)
    g = aggregate(ctx, [metric], [dimension], filters)
    b_col, c_col = f"{metric}_baseline", f"{metric}_current"
    parent_b, parent_c = float(g[b_col].sum()), float(g[c_col].sum())
    parent_delta = parent_c - parent_b
    g["delta"] = g[c_col] - g[b_col]
    g = g.sort_values("delta")

    rows = []
    for _, row in g.head(MAX_ROWS).iterrows():
        b, c, dlt = float(row[b_col]), float(row[c_col]), float(row["delta"])
        share_parent = dlt / parent_delta if parent_delta else None
        base_share = b / parent_b if parent_b else None
        conc = share_parent / base_share if share_parent is not None and base_share else None
        rest_b, rest_c = parent_b - b, parent_c - c
        rows.append(
            {
                "value": str(row[dimension]),
                "baseline": r(b, 2),
                "current": r(c, 2),
                "delta": r(dlt, 2),
                "pct_change": r(pct(c, b)),
                "baseline_share": r(base_share),
                "share_of_parent_change": r(share_parent),
                "share_of_total_change": r(dlt / total_delta) if total_delta else None,
                "concentration": r(conc, 3),
                "rest_pct_change": r(pct(rest_c, rest_b)),
            }
        )
    return {
        "metric": metric,
        "dimension": dimension,
        "filters": filters,
        "parent": {
            "baseline": r(parent_b, 2),
            "current": r(parent_c, 2),
            "delta": r(parent_delta, 2),
            "pct_change": r(pct(parent_c, parent_b)),
            "share_of_total_change": r(parent_delta / total_delta) if total_delta else None,
        },
        "total_change": r(total_delta, 2),
        "rows": rows,
    }


def drill_down(
    ctx: ToolContext, metric: str = "revenue", filters: dict | None = None, dimension: str = "country"
) -> dict:
    """Contribution analysis inside a segment (e.g. EMEA -> country)."""
    if not filters:
        raise ToolError("drill_down needs at least one filter; use contribution_analysis at the top level")
    out = contribution_analysis(ctx, metric=metric, dimension=dimension, filters=filters)
    out["drilled_from"] = dict(filters)
    return out
