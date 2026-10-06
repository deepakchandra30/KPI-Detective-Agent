"""Baseline 1: pure contribution analysis, no LLM, no loop (PRD 9.3).

Splits the change across every dimension and ranks single segments by
absolute revenue drop. It never says "data quality" or "seasonal": it is
the analyst who opens a pivot table and reports the biggest bar.
"""

from __future__ import annotations

import time

from kpi_detective.config import DIMENSIONS
from kpi_detective.models import AnomalyEvent
from kpi_detective.store import SalesStore
from kpi_detective.tools.base import ToolContext
from kpi_detective.tools.registry import run_tool

NAME = "rule_based"


def run(store: SalesStore, anomaly: AnomalyEvent) -> dict:
    t0 = time.perf_counter()
    ctx = ToolContext(store, anomaly)
    evidence = {"E0": {"id": "E0", "tool": "anomaly_event", "ok": True, "data": anomaly.to_dict()}}
    rows = []
    for dim in DIMENSIONS:
        ev = run_tool(ctx, "contribution_analysis", {"dimension": dim})
        ev["id"] = f"E{len(evidence)}"
        evidence[ev["id"]] = ev
        if not ev["ok"]:
            continue
        for i, row in enumerate(ev["data"]["rows"]):
            rows.append((row["delta"], dim, row, ev["id"], i))
    rows.sort(key=lambda t: t[0])  # most negative first

    causes, seen = [], set()
    for delta, dim, row, eid, i in rows:
        if delta >= 0 or (dim, row["value"]) in seen:
            continue
        seen.add((dim, row["value"]))
        share = row["share_of_total_change"]
        causes.append(
            {
                "rank": len(causes) + 1,
                "description": f"{dim}={row['value']} has the largest drop, {share:.0%} of the total change.",
                "segment": {dim: row["value"]},
                "mechanism": None,
                "share_of_change": share,
                "confidence": "medium",
                "evidence_ids": [eid],
                "claims": [
                    {
                        "label": "share_of_change",
                        "value": share,
                        "evidence_id": eid,
                        "path": f"rows[{i}].share_of_total_change",
                    }
                ],
            }
        )
        if len(causes) == 3:
            break
    report = {
        "classification": "business_change",
        "headline": "Largest segment drops by contribution analysis.",
        "claims": [],
        "causes": causes,
        "recommended_action": f"Investigate {causes[0]['segment']}" if causes else "No drop found.",
    }
    return {
        "report": report,
        "evidence": evidence,
        "steps": len(evidence) - 1,
        "seconds": time.perf_counter() - t0,
        "usage": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
    }
