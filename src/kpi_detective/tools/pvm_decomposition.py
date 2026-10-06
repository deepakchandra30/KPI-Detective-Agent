"""decompose_revenue: price-volume-mix decomposition.

With ``u_i`` units and ``p_i`` average price for each value ``i`` of a
dimension, ``U`` total units and ``P0 = R0 / U0`` the baseline average price:

    volume = (U1 - U0) * P0
    mix    = U1 * (sum_i s1_i p0_i - P0)        where s_i = u_i / U
    price  = sum_i u1_i (p1_i - p0_i)

The three effects add up exactly to R1 - R0 (tested in tests/test_tools.py).
"""

from __future__ import annotations

from kpi_detective.tools.base import (
    ToolContext,
    aggregate,
    pct,
    r,
    validate_dimension,
    validate_filters,
    validate_window,
)


def decompose_revenue(ctx: ToolContext, dimension: str = "product_category", filters: dict | None = None) -> dict:
    validate_window(ctx)
    validate_dimension(ctx, dimension)
    filters = validate_filters(ctx, filters)
    g = aggregate(ctx, ["revenue", "units"], [dimension], filters)
    g = g[(g["units_baseline"] > 0)].copy()

    u0, u1 = float(g["units_baseline"].sum()), float(g["units_current"].sum())
    r0, r1 = float(g["revenue_baseline"].sum()), float(g["revenue_current"].sum())
    p0_bar = r0 / u0 if u0 else 0.0

    g["p0"] = g["revenue_baseline"] / g["units_baseline"]
    g["p1"] = (g["revenue_current"] / g["units_current"]).where(g["units_current"] > 0, g["p0"])
    g["s0"] = g["units_baseline"] / u0 if u0 else 0.0
    g["s1"] = g["units_current"] / u1 if u1 else 0.0
    g["price_eff"] = g["units_current"] * (g["p1"] - g["p0"])
    g["mix_eff"] = u1 * (g["s1"] - g["s0"]) * g["p0"]

    volume = (u1 - u0) * p0_bar
    mix = float(g["mix_eff"].sum())
    price = float(g["price_eff"].sum())
    delta = r1 - r0
    residual = delta - (volume + mix + price)

    values = []
    for _, row in g.sort_values("revenue_current").iterrows():
        values.append(
            {
                "value": str(row[dimension]),
                "units_baseline": r(row["units_baseline"], 1),
                "units_current": r(row["units_current"], 1),
                "units_pct_change": r(pct(row["units_current"], row["units_baseline"])),
                "avg_price_baseline": r(row["p0"], 2),
                "avg_price_current": r(row["p1"], 2),
                "price_pct_change": r(pct(row["p1"], row["p0"])),
                "unit_share_baseline": r(row["s0"]),
                "unit_share_current": r(row["s1"]),
                "price_effect": r(row["price_eff"], 2),
                "mix_effect": r(row["mix_eff"], 2),
            }
        )

    def share(x: float) -> float | None:
        return r(x / delta) if delta else None

    return {
        "dimension": dimension,
        "filters": filters,
        "revenue_baseline": r(r0, 2),
        "revenue_current": r(r1, 2),
        "revenue_delta": r(delta, 2),
        "units_pct_change": r(pct(u1, u0)),
        "avg_price_pct_change": r(pct(r1 / u1 if u1 else 0.0, p0_bar)),
        "effects": {
            "volume": r(volume, 2),
            "mix": r(mix, 2),
            "price": r(price, 2),
            "volume_share": share(volume),
            "mix_share": share(mix),
            "price_share": share(price),
        },
        "residual": r(residual, 2),
        "values": values,
    }
