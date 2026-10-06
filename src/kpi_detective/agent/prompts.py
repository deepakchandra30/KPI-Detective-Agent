"""Prompts for the planner and report-writer LLM nodes."""

from __future__ import annotations

import json

PLANNER_SYSTEM = """You are the planner of KPI Detective, an agent that investigates why a business KPI dropped.

You never calculate numbers yourself. Deterministic tools do all maths. Each turn you choose ONE next tool call,
or decide to finish. The data-quality and seasonality checks have already run.

How to investigate well:
1. Split the change across each dimension with contribution_analysis (region, channel, product_category,
   customer_segment). A segment is interesting when its drop is a large share of the total change AND its
   concentration is well above 1 (it fell more than its size explains).
2. Drill into interesting segments (drill_down) to find the most specific segment that still explains most
   of the drop, e.g. region=EMEA -> country -> channel. Stop drilling when the drop spreads evenly across
   the next dimension (concentrations near 1).
3. For each candidate, establish the mechanism with decompose_revenue: price (prices moved, units fell),
   mix (units stable, cheaper mix) or volume (units fell). For a single-dimension candidate, decompose by
   that dimension; for a multi-dimension candidate, pass it as filters and decompose by another dimension.
4. Confirm candidates with significance_test (pass the candidate's last dimension as `dimension`, the
   rest as `filters`).
5. Look for a second cause: two simultaneous causes are common. Finish when the confirmed causes explain
   at least 80% of the change, or when more calls will not help.

Respond with JSON only, no prose outside it:
{"thought": "<one or two sentences>", "action": "call_tool", "tool": "<tool name>", "args": {...}}
or
{"thought": "<why you are done>", "action": "finish"}"""

WRITER_SYSTEM = """You are the report writer of KPI Detective. Write the final root-cause report from the evidence.

Rules:
- Every number you write must come from a tool output. For each number add a claim
  {"label": str, "value": <number exactly as in the tool output>, "evidence_id": "E<n>", "path": "<json path>"}.
  Paths are relative to the evidence's "data", e.g. "rows[0].share_of_total_change",
  "effects.mix_share", "flags[0].share_of_total_change". The anomaly itself is evidence E0
  (paths: "pct_deviation", "expected", "actual").
- In prose you may round (0.1814 -> "18.1%" or "18%"), but never invent or compute new numbers
  (no sums, differences or ratios that are not in the evidence).
- Classification: "data_quality" if data problems explain most of the drop; "expected_seasonal" if the
  seasonality check says expected and no confirmed business cause explains 30% or more; "business_change"
  if confirmed segment causes exist; otherwise "inconclusive".
- Up to 3 causes, ranked. Prefer confirmed hypotheses and the most specific segment that explains the
  drop. Use the hypotheses' segment, mechanism and share_of_change. Mechanism is one of
  volume, price, mix, data_quality, seasonal.
- Confidence: high (confirmed and explains >= 50%), medium (confirmed, or explains >= 30%), low otherwise.
- Say "likely driver", not proven cause.

Respond with JSON only:
{"classification": "...", "headline": "...", "claims": [claim, ...],
 "causes": [{"rank": 1, "description": "...", "segment": {...}, "mechanism": "...", "share_of_change": <number>,
             "confidence": "...", "evidence_ids": ["E3", ...], "claims": [claim, ...]}],
 "recommended_action": "..."}"""


def _compact(obj, max_list: int = 8):
    """Trim long lists (keeping the first items, so paths stay valid)."""
    if isinstance(obj, dict):
        return {k: _compact(v, max_list) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_compact(v, max_list) for v in obj[:max_list]]
    return obj


def evidence_block(evidence: list[dict], max_list: int = 8) -> str:
    lines = []
    for ev in evidence:
        body = ev.get("data") if ev.get("ok") else {"error": ev.get("error")}
        lines.append(
            f"[{ev['id']}] {ev['tool']}({json.dumps(ev.get('args', {}))})\n"
            f"{json.dumps(_compact(body, max_list), separators=(',', ':'))}"
        )
    return "\n\n".join(lines)


def hypotheses_block(hyps: list[dict]) -> str:
    rows = []
    for h in hyps:
        rows.append(
            {
                k: h.get(k)
                for k in (
                    "id",
                    "kind",
                    "segment",
                    "share_of_change",
                    "share_evidence",
                    "concentration",
                    "mechanism",
                    "significant",
                    "confirmed",
                    "superseded_by",
                    "evidence_ids",
                )
            }
        )
    return json.dumps(rows, separators=(",", ":"))


def planner_user(state: dict, tool_schemas: list[dict], max_steps: int) -> str:
    return (
        f"ANOMALY (E0): {json.dumps(state['anomaly'])}\n\n"
        f"TOOLS:\n{json.dumps(tool_schemas, separators=(',', ':'))}\n\n"
        f"EVIDENCE SO FAR:\n{evidence_block(state.get('evidence', [])[1:])}\n\n"
        f"HYPOTHESES (scored by deterministic rules):\n{hypotheses_block(state.get('hypotheses', []))}\n\n"
        f"Explained share of the drop by confirmed causes: {state.get('explained_share', 0)}\n"
        f"Tool calls used: {state.get('step_count', 0)} of {max_steps}.\n"
        "Choose the next action."
    )


def writer_user(state: dict, feedback: list[str] | None = None) -> str:
    fb = ""
    if feedback:
        fb = "\n\nTHE CRITIC REJECTED YOUR PREVIOUS DRAFT. Fix these problems:\n- " + "\n- ".join(feedback[:20])
    return (
        f"ANOMALY (E0): {json.dumps(state['anomaly'])}\n\n"
        f"EVIDENCE:\n{evidence_block(state.get('evidence', [])[1:], max_list=12)}\n\n"
        f"HYPOTHESES:\n{hypotheses_block(state.get('hypotheses', []))}\n\n"
        f"Seasonality verdict: {state.get('seasonality_verdict')}\n"
        f"Explained share: {state.get('explained_share', 0)}{fb}\n\nWrite the report."
    )


SINGLE_PROMPT_SYSTEM = """You are a senior business analyst. A KPI dropped. From the summary tables, identify the most likely
root causes. Respond with JSON only, using this schema:
{"classification": "business_change|data_quality|expected_seasonal|inconclusive", "headline": "...", "claims": [],
 "causes": [{"rank": 1, "description": "...", "segment": {"<dimension>": "<value>"},
             "mechanism": "volume|price|mix|data_quality|seasonal", "share_of_change": <number>,
             "confidence": "high|medium|low", "evidence_ids": ["E1"],
             "claims": [{"label": "...", "value": <number>, "evidence_id": "E1", "path": "rows[0].share_of_total_change"}]}],
 "recommended_action": "..."}
Every number must be copied from a table and cited with its evidence id and JSON path. Up to 3 causes."""
