"""LangGraph investigation graph (PRD section 8).

    start
      -> data_quality_check (forced)
      -> seasonality_check  (forced)
      -> planner -> tool_executor -> evaluator --+--> planner         (steps < 12 and (explained < 0.8 or open leads))
                                                 +--> report_writer
      -> report_writer -> critic --+--> report_writer   (unsupported claims, max 2 revisions)
                                   +--> end

The forced first steps are graph structure, not a request to the LLM.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

from langgraph.graph import END, START, StateGraph

from kpi_detective.agent import evaluator
from kpi_detective.agent.critic import verify_report
from kpi_detective.agent.policy import Policy
from kpi_detective.agent.state import InvestigationState
from kpi_detective.config import EXPLAINED_SHARE_TARGET, MAX_REPORT_REVISIONS, MAX_TOOL_CALLS
from kpi_detective.models import AnomalyEvent
from kpi_detective.store import SalesStore
from kpi_detective.tools.base import ToolContext
from kpi_detective.tools.registry import run_tool


@dataclass
class InvestigationResult:
    report: dict
    state: InvestigationState
    seconds: float
    usage: dict = field(default_factory=dict)

    @property
    def trace(self) -> list[dict]:
        return self.state.get("trace", [])

    @property
    def steps(self) -> int:
        return self.state.get("step_count", 0)


def _log(state: InvestigationState, node: str, **detail) -> list[dict]:
    return [*state.get("trace", []), {"node": node, "t": round(time.perf_counter(), 3), **detail}]


def build_graph(ctx: ToolContext, policy: Policy, use_critic: bool = True):
    def execute(state: InvestigationState, tool: str, args: dict, node: str, thought: str = "") -> dict:
        ev = run_tool(ctx, tool, args)
        ev["id"] = f"E{len(state['evidence'])}"
        st = copy.copy(state)
        st["evidence"] = [*state["evidence"], ev]
        st["step_count"] = state.get("step_count", 0) + 1
        evaluator.update(st, ev)
        st["trace"] = _log(
            state,
            node,
            tool=tool,
            args=args,
            evidence_id=ev["id"],
            ok=ev["ok"],
            error=ev.get("error"),
            thought=thought,
            ms=ev["ms"],
            explained_share=st["explained_share"],
        )
        return st

    def data_quality_check(state: InvestigationState) -> dict:
        return execute(state, "check_data_quality", {}, "data_quality_check", "Forced first step.")

    def seasonality_check(state: InvestigationState) -> dict:
        return execute(state, "check_seasonality", {}, "seasonality_check", "Forced second step.")

    def planner(state: InvestigationState) -> dict:
        action = policy.plan(state)
        return {
            "pending_action": action,
            "finished": action["action"] == "finish",
            "plan": [*state.get("plan", []), action.get("thought", "")],
            "trace": _log(state, "planner", action=action),
        }

    def tool_executor(state: InvestigationState) -> dict:
        a = state["pending_action"]
        out = execute(state, a["tool"], a.get("args", {}), "tool_executor", a.get("thought", ""))
        out["pending_action"] = None
        return out

    def report_writer(state: InvestigationState) -> dict:
        feedback = state.get("critic", {}).get("issues") if state.get("revisions", 0) else None
        report = policy.write_report(state, feedback)
        return {"draft_report": report, "trace": _log(state, "report_writer", revision=state.get("revisions", 0))}

    def critic(state: InvestigationState) -> dict:
        evidence = {e["id"]: e for e in state["evidence"]}
        result = verify_report(state["draft_report"], evidence)
        return {
            "critic": result,
            "verified": result["verified"],
            "revisions": state.get("revisions", 0) + 1,
            "trace": _log(
                state,
                "critic",
                **{k: result[k] for k in ("verified", "n_numbers", "n_unsupported")},
                issues=result["issues"][:10],
            ),
        }

    def route_after_evidence(state: InvestigationState) -> str:
        if state["step_count"] >= MAX_TOOL_CALLS:
            return "report_writer"
        if state.get("explained_share", 0) >= EXPLAINED_SHARE_TARGET and not evaluator.open_leads(state["hypotheses"]):
            return "report_writer"
        return "planner"

    def route_after_planner(state: InvestigationState) -> str:
        return "report_writer" if state.get("finished") else "tool_executor"

    def route_after_critic(state: InvestigationState) -> str:
        if not state["verified"] and state["revisions"] <= MAX_REPORT_REVISIONS:
            return "report_writer"
        return END

    g = StateGraph(InvestigationState)
    for name, fn in [
        ("data_quality_check", data_quality_check),
        ("seasonality_check", seasonality_check),
        ("planner", planner),
        ("tool_executor", tool_executor),
        ("report_writer", report_writer),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "data_quality_check")
    g.add_edge("data_quality_check", "seasonality_check")
    g.add_conditional_edges("seasonality_check", route_after_evidence, ["planner", "report_writer"])
    g.add_conditional_edges("planner", route_after_planner, ["tool_executor", "report_writer"])
    g.add_conditional_edges("tool_executor", route_after_evidence, ["planner", "report_writer"])
    if use_critic:
        g.add_node("critic", critic)
        g.add_edge("report_writer", "critic")
        g.add_conditional_edges("critic", route_after_critic, ["report_writer", END])
    else:
        g.add_edge("report_writer", END)
    return g.compile()


def investigate(
    store: SalesStore, anomaly: AnomalyEvent, policy: Policy, use_critic: bool = True
) -> InvestigationResult:
    """Run a full investigation and return the report, state and trace."""
    ctx = ToolContext(store, anomaly)
    graph = build_graph(ctx, policy, use_critic)
    anomaly_d = anomaly.to_dict()
    init: InvestigationState = {
        "anomaly": anomaly_d,
        "plan": [],
        "evidence": [{"id": "E0", "tool": "anomaly_event", "args": {}, "ok": True, "data": anomaly_d, "ms": 0.0}],
        "hypotheses": [],
        "explained_share": 0.0,
        "step_count": 0,
        "revisions": 0,
        "verified": False,
        "trace": [],
    }
    usage_before = _usage_snapshot(policy)
    t0 = time.perf_counter()
    final = graph.invoke(init, config={"recursion_limit": 100})
    seconds = time.perf_counter() - t0
    report = dict(final["draft_report"])
    report["explained_share"] = final.get("explained_share", 0.0)
    if use_critic:
        report["verification"] = final.get("critic", {})
    return InvestigationResult(report, final, seconds, _usage_delta(policy, usage_before))


def _usage_snapshot(policy: Policy) -> dict:
    client = getattr(policy, "client", None)
    return client.usage.to_dict() if client else {}


def _usage_delta(policy: Policy, before: dict) -> dict:
    after = _usage_snapshot(policy)
    if not after:
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "llm_seconds": 0.0, "cost_usd": 0.0}
    out = {k: (after[k] or 0) - (before.get(k) or 0) for k in ("calls", "input_tokens", "output_tokens", "llm_seconds")}
    client = policy.client  # type: ignore[attr-defined]
    u = client.usage
    out["cost_usd"] = (
        round((out["input_tokens"] * u.price_in + out["output_tokens"] * u.price_out) / 1e6, 6)
        if u.price_in is not None and u.price_out is not None
        else None
    )
    return out
