"""Investigation state carried through the LangGraph graph (PRD section 8)."""

from __future__ import annotations

from typing import Any, TypedDict


class Hypothesis(TypedDict, total=False):
    id: str
    kind: str  # "segment" | "data_quality" | "seasonal"
    segment: dict[str, str]
    share_of_change: float | None  # share of the total KPI change explained
    share_evidence: tuple[str, str] | None  # (evidence id, json path) of that share
    pct_change: float | None
    concentration: float | None
    mechanism: str | None  # volume | price | mix | data_quality | seasonal
    mechanism_evidence: str | None
    significant: bool | None
    q_value: float | None
    significance_evidence: str | None
    evidence_ids: list[str]
    superseded_by: str | None
    confirmed: bool
    detail: dict[str, Any]


class InvestigationState(TypedDict, total=False):
    anomaly: dict  # date range, metric, expected/actual, deviation
    plan: list[str]
    evidence: list[dict]  # tool outputs, each with an "id"
    hypotheses: list[Hypothesis]
    explained_share: float
    step_count: int
    draft_report: dict
    verified: bool
    critic: dict
    revisions: int
    pending_action: dict | None
    finished: bool
    seasonality_verdict: str | None
    trace: list[dict]
    usage: dict
