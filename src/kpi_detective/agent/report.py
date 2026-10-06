"""Report schema and the deterministic report writer (FR11-FR13).

Report schema (shared by the LLM writer, the heuristic writer and baselines)::

    {
      "classification": "business_change" | "data_quality" | "expected_seasonal" | "inconclusive",
      "headline": str,
      "claims": [claim, ...],                 # numbers used in the headline
      "causes": [                             # ranked, at most 3
        {"rank", "description", "segment", "mechanism", "share_of_change",
         "confidence": "high" | "medium" | "low", "evidence_ids", "claims"}
      ],
      "recommended_action": str
    }
    claim = {"label", "value", "evidence_id", "path"}
"""

from __future__ import annotations

from kpi_detective.agent.critic import resolve
from kpi_detective.agent.evaluator import active_hypotheses
from kpi_detective.agent.segments import label
from kpi_detective.agent.state import Hypothesis, InvestigationState

CLASSIFICATIONS = ["business_change", "data_quality", "expected_seasonal", "inconclusive"]
MECHANISMS = ["volume", "price", "mix", "data_quality", "seasonal"]
DQ_MAJORITY = 0.5
SEASONAL_MAJORITY = 0.75
MAX_CAUSES = 3


class _Claims:
    """Collects claims while formatting numbers, so text and claims never drift apart."""

    def __init__(self, evidence: dict[str, dict]):
        self.evidence = evidence
        self.items: list[dict] = []

    def num(self, label_: str, ref: tuple[str, str] | list | None, fmt: str) -> str | None:
        if not ref:
            return None
        eid, path = ref
        try:
            item = self.evidence[eid]
            value = float(resolve(item.get("data", item), path))
        except (KeyError, TypeError, ValueError):
            return None
        self.items.append({"label": label_, "value": value, "evidence_id": eid, "path": path})
        return format(value, fmt)


def _confidence(h: Hypothesis) -> str:
    share = h.get("share_of_change") or 0
    if h["confirmed"] and share >= 0.5:
        return "high"
    if h["confirmed"] or share >= 0.3:
        return "medium"
    return "low"


def _describe(h: Hypothesis, c: _Claims) -> str:
    seg = label(h["segment"])
    share = c.num("share_of_change", h.get("share_evidence"), ".0%")
    share_txt = f", {share} of the total drop" if share else ""
    if h["kind"] == "data_quality":
        if h["detail"].get("type") == "partial_load":
            ratio = (
                c.num(
                    "row_ratio",
                    (h["mechanism_evidence"], h["share_evidence"][1].replace("share_of_total_change", "row_ratio")),
                    ".0%",
                )
                if h.get("share_evidence")
                else None
            )
            dates = ", ".join(h["detail"].get("dates", []))
            return f"Data for {dates} is only partly loaded ({ratio} of the usual rows){share_txt}."
        return f"Revenue for {seg} is recorded as zero or missing in the window{share_txt}."
    if h["kind"] == "seasonal":
        ly = c.num("last_year_change", (h["mechanism_evidence"], "last_year_same_period_change_pct"), "+.1%")
        return f"The same period last year moved {ly} as well: expected seasonal behaviour{share_txt}."

    pct = c.num("pct_change", h["detail"].get("pct_change_path"), "+.1%")
    head = f"{seg} revenue changed {pct}" if pct else f"{seg} revenue fell"
    paths = h["detail"].get("mechanism_paths", {})
    mech = h.get("mechanism")
    if mech == "price":
        p = c.num("price_pct_change", paths.get("price_pct_change") or paths.get("avg_price_pct_change"), "+.1%")
        u = c.num("units_pct_change", paths.get("units_pct_change"), "+.1%")
        why = f"price effect: average price {p}, units {u}"
    elif mech == "mix" and paths.get("unit_share_baseline"):
        s0 = c.num("unit_share_baseline", paths.get("unit_share_baseline"), ".1%")
        s1 = c.num("unit_share_current", paths.get("unit_share_current"), ".1%")
        why = f"mix effect: its share of units moved from {s0} to {s1}"
    elif mech == "mix":
        u = c.num("units_pct_change", paths.get("units_pct_change"), "+.1%")
        why = f"mix effect: units {u} but a cheaper product mix"
    elif mech == "volume":
        u = c.num("units_pct_change", paths.get("units_pct_change"), "+.1%")
        why = f"volume effect: units {u}"
    else:
        why = "mechanism not yet established"
    sig = ""
    if h.get("significant") is not None:
        q = c.num("q_value", h["detail"].get("q_value_path"), ".3f")
        sig = f"; significant vs history (q={q})" if h["significant"] else f"; not significant (q={q})"
    return f"{head} ({why}){share_txt}{sig}."


def classify(state: InvestigationState) -> str:
    hyps = active_hypotheses(state.get("hypotheses", []))
    dq = sum(h.get("share_of_change") or 0 for h in hyps if h["kind"] == "data_quality")
    if dq >= DQ_MAJORITY:
        return "data_quality"
    seasonal = next((h for h in hyps if h["kind"] == "seasonal"), None)
    business = [h for h in hyps if h["kind"] == "segment" and h["confirmed"]]
    big_business = [h for h in business if (h.get("share_of_change") or 0) >= 0.3]
    if seasonal and (seasonal.get("share_of_change") or 0) >= SEASONAL_MAJORITY and not big_business:
        return "expected_seasonal"
    if business:
        return "business_change"
    return "inconclusive"


def _action(classification: str, causes: list[Hypothesis]) -> str:
    if classification == "data_quality":
        seg = label(causes[0]["segment"]) if causes else "the affected data"
        return f"Contact data engineering about {seg} and re-run once the data is reloaded; do not escalate as a business issue."
    if classification == "expected_seasonal":
        return "No action needed: this is normal seasonal behaviour. Keep monitoring against last year."
    if classification == "inconclusive":
        return "The drop is not concentrated in a segment. Monitor for another few days before escalating."
    top = causes[0]
    seg = label(top["segment"])
    return {
        "price": f"Review the recent price change for {seg} with the pricing team and its effect on demand.",
        "mix": f"Review the {seg} pipeline with sales: revenue fell because the customer mix shifted, not total units.",
        "volume": f"Check operations for {seg} (site availability, ordering, stock) during the window.",
    }.get(top.get("mechanism") or "", f"Investigate {seg} further with the owning team.")


def build_report(state: InvestigationState) -> dict:
    """Deterministic report from the evaluator's hypotheses (used by the offline policy)."""
    evidence = {e["id"]: e for e in state.get("evidence", [])}
    classification = classify(state)
    hyps = active_hypotheses(state.get("hypotheses", []))
    if classification == "data_quality":
        order = [h for h in hyps if h["kind"] == "data_quality"] + [h for h in hyps if h["kind"] == "segment"]
    elif classification == "expected_seasonal":
        order = [h for h in hyps if h["kind"] == "seasonal"] + [h for h in hyps if h["kind"] == "segment"]
    else:
        order = [h for h in hyps if h["kind"] == "segment"] + [h for h in hyps if h["kind"] != "segment"]
    chosen = order[:MAX_CAUSES]

    causes = []
    for rank, h in enumerate(chosen, 1):
        claims = _Claims(evidence)
        desc = _describe(h, claims)
        causes.append(
            {
                "rank": rank,
                "description": desc,
                "segment": h["segment"],
                "mechanism": h.get("mechanism"),
                "share_of_change": h.get("share_of_change"),
                "confidence": _confidence(h),
                "evidence_ids": list(h["evidence_ids"]),
                "claims": claims.items,
                "hypothesis_id": h["id"],
            }
        )

    head = _Claims(evidence)
    dev = head.num("pct_deviation", ("E0", "pct_deviation"), "+.1%")
    a = state["anomaly"]
    headline = f"{a['metric'].capitalize()} {dev} vs expected for {a['start']} to {a['end']}: {classification.replace('_', ' ')}."
    return {
        "classification": classification,
        "headline": headline,
        "claims": head.items,
        "causes": causes,
        "recommended_action": _action(classification, chosen),
    }
