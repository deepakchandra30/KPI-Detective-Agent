"""Baseline 2: one LLM call over a summary table. No tools, no loop (PRD 9.3).

The model sees the same numbers an analyst would paste into a chat:
overall totals and a contribution table per dimension. It must name the
cause in one shot.
"""

from __future__ import annotations

import time

from kpi_detective.agent import prompts
from kpi_detective.agent.llm import LLMClient, LLMError, parse_json
from kpi_detective.agent.policy import sanitize_report
from kpi_detective.config import DIMENSIONS
from kpi_detective.models import AnomalyEvent
from kpi_detective.store import SalesStore
from kpi_detective.tools.base import ToolContext
from kpi_detective.tools.registry import run_tool

NAME = "single_prompt"


def summary_evidence(store: SalesStore, anomaly: AnomalyEvent) -> list[dict]:
    ctx = ToolContext(store, anomaly)
    evidence = [{"id": "E0", "tool": "anomaly_event", "args": {}, "ok": True, "data": anomaly.to_dict()}]
    calls = [("compare_periods", {})] + [("contribution_analysis", {"dimension": d}) for d in DIMENSIONS]
    for tool, args in calls:
        ev = run_tool(ctx, tool, args)
        ev["id"] = f"E{len(evidence)}"
        evidence.append(ev)
    return evidence


def run(store: SalesStore, anomaly: AnomalyEvent, client: LLMClient) -> dict:
    t0 = time.perf_counter()
    evidence = summary_evidence(store, anomaly)
    before = client.usage.to_dict()
    user = (
        f"ANOMALY (E0): {anomaly.to_dict()}\n\nSUMMARY TABLES:\n"
        f"{prompts.evidence_block(evidence[1:], max_list=12)}\n\nIdentify the root causes."
    )
    try:
        report = sanitize_report(parse_json(client.complete(prompts.SINGLE_PROMPT_SYSTEM, user, node="single")))
    except LLMError as exc:
        report = sanitize_report({"classification": "inconclusive", "headline": str(exc)})
    after = client.usage.to_dict()
    usage = {k: (after[k] or 0) - (before[k] or 0) for k in ("calls", "input_tokens", "output_tokens")}
    u = client.usage
    usage["cost_usd"] = (
        round((usage["input_tokens"] * u.price_in + usage["output_tokens"] * u.price_out) / 1e6, 6)
        if u.price_in is not None and u.price_out is not None
        else None
    )
    return {
        "report": report,
        "evidence": {e["id"]: e for e in evidence},
        "steps": 0,
        "seconds": time.perf_counter() - t0,
        "usage": usage,
    }
