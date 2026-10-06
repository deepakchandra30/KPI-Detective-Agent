"""Policies: who decides the next tool call and who writes the report.

* :class:`LLMPolicy` is the real agent: an LLM plans and writes.
* :class:`HeuristicPolicy` is a scripted analyst playbook with the same
  interface. It needs no API key, so the whole pipeline (graph, tools,
  evaluator, critic, benchmark) runs offline, in CI and in the demo app.
  It is reported as its own system in the benchmark, never as "the agent".
"""

from __future__ import annotations

import json
from typing import Protocol

from kpi_detective.agent import prompts
from kpi_detective.agent.evaluator import active_hypotheses
from kpi_detective.agent.llm import LLMClient, LLMError, parse_json
from kpi_detective.agent.report import CLASSIFICATIONS, MAX_CAUSES, MECHANISMS, build_report
from kpi_detective.agent.segments import normalize
from kpi_detective.agent.state import InvestigationState
from kpi_detective.config import DIMENSIONS, MAX_TOOL_CALLS
from kpi_detective.tools.registry import TOOLS

TOP_LEVEL_DIMS = ["region", "channel", "product_category", "customer_segment"]
DECOMPOSE_ORDER = ["product_category", "channel", "customer_segment", "country"]
MAX_SEGMENT_DIMS = 2
MAX_DRILLS_PER_CANDIDATE = 2
PRICE_MIX_DIMS = ("product_category", "customer_segment")


class Policy(Protocol):
    name: str

    def plan(self, state: InvestigationState) -> dict: ...

    def write_report(self, state: InvestigationState, feedback: list[str] | None = None) -> dict: ...


def _depth(segment: dict) -> int:
    """Segment depth, counting region and country as one geographic level."""
    return len({"geo" if d in ("region", "country") else d for d in segment})


def _signature(tool: str, args: dict) -> str:
    return tool + json.dumps(args, sort_keys=True)


# ============================================================== heuristic
class HeuristicPolicy:
    """Deterministic playbook: split, drill, decompose, test. No LLM."""

    name = "heuristic"

    def _done(self, state: InvestigationState) -> set[str]:
        return {_signature(e["tool"], e.get("args", {})) for e in state.get("evidence", [])}

    def plan(self, state: InvestigationState) -> dict:
        done = self._done(state)
        left = MAX_TOOL_CALLS - state.get("step_count", 0)

        def todo(tool: str, args: dict, thought: str) -> dict | None:
            if _signature(tool, args) in done:
                return None
            return {"action": "call_tool", "tool": tool, "args": args, "thought": thought}

        # 1. split the change across every top-level dimension
        for dim in TOP_LEVEL_DIMS:
            act = todo("contribution_analysis", {"dimension": dim}, f"Split the change by {dim}.")
            if act:
                return act

        hyps = state.get("hypotheses", [])
        best_share: dict[str, float] = {}
        for h in hyps:
            if h["kind"] == "segment" and len(h["segment"]) == 1:
                (d,) = h["segment"]
                best_share[d] = max(best_share.get(d, 0.0), h.get("share_of_change") or 0.0)
        candidate_dims = sorted(best_share, key=lambda d: -best_share[d])
        # a dimension whose leading value is already a confirmed price/mix cause is explained
        explained_dims = {
            d for h in hyps if h["confirmed"] and h.get("mechanism") in ("price", "mix") for d in h["segment"]
        }
        # depth-first: finish following a drill-down thread before opening a new one
        candidates = [
            h
            for h in active_hypotheses(hyps)
            if h["kind"] == "segment" and not h["confirmed"] and h.get("significant") is not False
        ]
        candidates = sorted(
            candidates, key=lambda h: (not h["detail"].get("from_drill"), -(h.get("share_of_change") or 0))
        )[:3]

        for h in candidates:
            seg = h["segment"]
            dims = list(seg)
            # a) price and mix live in categories and customer segments: decompose those first
            if h["mechanism"] is None and len(seg) == 1 and dims[0] in PRICE_MIX_DIMS:
                act = todo("decompose_revenue", {"dimension": dims[0]}, f"Price, volume or mix for {seg}?")
                if act:
                    return act
            # b) volume drops: find the most specific segment (region -> country -> channel ...)
            if h["mechanism"] in (None, "volume") and _depth(seg) < MAX_SEGMENT_DIMS and left >= 3:
                other = [
                    d
                    for d in candidate_dims
                    if d not in seg and d not in explained_dims and d not in ("region", "country")
                ]
                if "region" in seg:
                    drill_dims = ["country", *other]
                elif "country" in seg:
                    drill_dims = other
                else:
                    drill_dims = [d for d in candidate_dims if d == "region"] + other
                for d in drill_dims[:MAX_DRILLS_PER_CANDIDATE]:
                    act = todo(
                        "drill_down", {"filters": seg, "dimension": d}, f"Is the drop in {seg} concentrated in one {d}?"
                    )
                    if act:
                        return act
            elif h["mechanism"] in (None, "volume") and "region" in seg and left >= 3:
                act = todo(
                    "drill_down",
                    {"filters": seg, "dimension": "country"},
                    f"Is the drop in {seg} concentrated in one country?",
                )
                if act:
                    return act
            # c) mechanism of the final segment
            if h["mechanism"] is None:
                dim = next(d for d in DECOMPOSE_ORDER if d not in seg and not (d == "country" and "region" in seg))
                act = todo(
                    "decompose_revenue", {"dimension": dim, "filters": seg}, f"Price, volume or mix inside {seg}?"
                )
                if act:
                    return act
            # d) significance against history
            if h["significant"] is None:
                last = dims[-1]
                rest = {k: v for k, v in seg.items() if k != last}
                args = {"dimension": last, "filters": rest} if rest else {"dimension": last}
                act = todo("significance_test", args, f"Is the {seg} drop beyond normal variation?")
                if act:
                    return act
        return {"action": "finish", "thought": "No further analysis would change the conclusion."}

    def write_report(self, state: InvestigationState, feedback: list[str] | None = None) -> dict:
        return build_report(state)


# ==================================================================== LLM
class LLMPolicy:
    """The LLM chooses tools and writes the report; tools do all the maths."""

    def __init__(self, client: LLMClient):
        self.client = client
        self.name = f"llm:{client.model}"

    def plan(self, state: InvestigationState) -> dict:
        schemas = [t.schema() for t in TOOLS.values() if t.name not in ("check_data_quality", "check_seasonality")]
        user = prompts.planner_user(state, schemas, MAX_TOOL_CALLS)
        for _attempt in range(2):
            try:
                out = parse_json(self.client.complete(prompts.PLANNER_SYSTEM, user, node="planner"))
                if out.get("action") == "finish":
                    return {"action": "finish", "thought": str(out.get("thought", ""))}
                if out.get("action") == "call_tool" and out.get("tool") in TOOLS:
                    return {
                        "action": "call_tool",
                        "tool": out["tool"],
                        "args": out.get("args") or {},
                        "thought": str(out.get("thought", "")),
                    }
                user += "\n\nYour last answer was not a valid action. Answer with the JSON schema only."
            except (LLMError, json.JSONDecodeError):
                user += "\n\nYour last answer was not valid JSON. Answer with JSON only."
        return {"action": "finish", "thought": "Planner failed to produce a valid action."}

    def write_report(self, state: InvestigationState, feedback: list[str] | None = None) -> dict:
        user = prompts.writer_user(state, feedback)
        try:
            raw = parse_json(self.client.complete(prompts.WRITER_SYSTEM, user, node="report_writer"))
        except (LLMError, json.JSONDecodeError) as exc:
            return {
                "classification": "inconclusive",
                "headline": f"Report generation failed: {exc}",
                "claims": [],
                "causes": [],
                "recommended_action": "Re-run the investigation.",
            }
        return sanitize_report(raw)


def sanitize_report(raw: dict) -> dict:
    """Coerce an LLM report into the schema without changing its content."""
    cls = raw.get("classification")
    causes = []
    for i, c in enumerate((raw.get("causes") or [])[:MAX_CAUSES], 1):
        seg = c.get("segment") or {}
        seg = normalize({k: v for k, v in seg.items() if k in DIMENSIONS}) if isinstance(seg, dict) else {}
        mech = c.get("mechanism")
        try:
            share = float(c["share_of_change"]) if c.get("share_of_change") is not None else None
        except (TypeError, ValueError):
            share = None
        causes.append(
            {
                "rank": i,
                "description": str(c.get("description", "")),
                "segment": seg,
                "mechanism": mech if mech in MECHANISMS else None,
                "share_of_change": share,
                "confidence": c.get("confidence") if c.get("confidence") in ("high", "medium", "low") else "low",
                "evidence_ids": [str(e) for e in c.get("evidence_ids") or []],
                "claims": [cl for cl in c.get("claims") or [] if isinstance(cl, dict)],
            }
        )
    return {
        "classification": cls if cls in CLASSIFICATIONS else "inconclusive",
        "headline": str(raw.get("headline", "")),
        "claims": [cl for cl in raw.get("claims") or [] if isinstance(cl, dict)],
        "causes": causes,
        "recommended_action": str(raw.get("recommended_action", "")),
    }
