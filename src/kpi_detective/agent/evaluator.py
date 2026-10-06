"""Evaluator node: turns tool outputs into scored hypotheses (deterministic).

The LLM decides *what to look at*; this module decides, with fixed rules,
what the numbers say. Every hypothesis keeps pointers to the evidence that
produced it, so the report writer can cite them and the critic can check
them.

A segment hypothesis is **confirmed** once it has a mechanism (volume,
price or mix, from a price-volume-mix decomposition) and a significant
test result. Data-quality and seasonal hypotheses are confirmed by the
check that created them. ``explained_share`` sums the shares of confirmed,
non-nested hypotheses and drives the stopping rule (FR9), together with
``open_leads``: strong unconfirmed candidates that keep the loop going so a
second simultaneous cause is not missed once the first explains 80%.
"""

from __future__ import annotations

from statistics import median

from kpi_detective.agent.segments import covers, normalize
from kpi_detective.agent.state import Hypothesis, InvestigationState

MIN_SHARE = 0.15  # share of total change for a segment to become a hypothesis
MIN_CONCENTRATION = 1.3  # drop must be disproportionate to the segment's size
CHILD_PARENT_SHARE = 0.6  # child keeps >= 60% of the parent's share -> supersedes it
PRICE_CHANGE = 0.03
UNITS_CHANGE = 0.03
MIX_SHARE = 0.5
MIX_MIN_SHARE_SHIFT = 0.02  # a mix cause must lose at least 2 points of unit share
SIBLING_MIN_DROP = 0.05  # child must fall at least 5% ...
SIBLING_GAP = 0.5  # ... and at least half of its drop must not be shared by its siblings
LEAD_MIN_SHARE = 0.3  # unconfirmed candidates this strong keep the investigation going
LEAD_MIN_CONCENTRATION = 1.2


def is_concentrated(row: dict) -> bool:
    """Disproportionate drop: high concentration, or clearly worse than the rest of its parent."""
    conc, pc, rest = row.get("concentration"), row.get("pct_change"), row.get("rest_pct_change")
    if conc is not None and conc >= MIN_CONCENTRATION:
        return True
    if pc is None or rest is None or pc > -SIBLING_MIN_DROP:
        return False
    return pc - rest <= -SIBLING_GAP * abs(pc)


def open_leads(hyps: list[Hypothesis]) -> list[Hypothesis]:
    """Strong, unconfirmed candidates not covered by a confirmed cause: worth checking before stopping."""
    confirmed = [h for h in hyps if h["confirmed"] and h["kind"] == "segment"]
    leads = []
    for h in active_hypotheses(hyps):
        if h["kind"] != "segment" or h["confirmed"] or h.get("significant") is False:
            continue
        if (h.get("share_of_change") or 0) < LEAD_MIN_SHARE or (h.get("concentration") or 0) < LEAD_MIN_CONCENTRATION:
            continue
        if any(covers(c["segment"], h["segment"]) or covers(h["segment"], c["segment"]) for c in confirmed):
            continue
        leads.append(h)
    return leads


def _find(hyps: list[Hypothesis], segment: dict, kind: str = "segment") -> Hypothesis | None:
    seg = normalize(segment)
    for h in hyps:
        if h["kind"] == kind and h["segment"] == seg:
            return h
    return None


def _new(hyps: list[Hypothesis], kind: str, segment: dict, **kw) -> Hypothesis:
    h: Hypothesis = {
        "id": f"H{len(hyps) + 1}",
        "kind": kind,
        "segment": normalize(segment),
        "share_of_change": None,
        "share_evidence": None,
        "pct_change": None,
        "concentration": None,
        "mechanism": None,
        "mechanism_evidence": None,
        "significant": None,
        "q_value": None,
        "significance_evidence": None,
        "evidence_ids": [],
        "superseded_by": None,
        "confirmed": False,
        "detail": {},
    }
    h.update(kw)  # type: ignore[typeddict-item]
    hyps.append(h)
    return h


def _cite(h: Hypothesis, eid: str) -> None:
    if eid not in h["evidence_ids"]:
        h["evidence_ids"].append(eid)


def _supersede(hyps: list[Hypothesis], child: Hypothesis) -> None:
    """Nested segments: keep the most specific one that still explains most of the drop."""
    for other in hyps:
        if other is child or other["kind"] != "segment" or other["superseded_by"]:
            continue
        cs, os_ = child.get("share_of_change") or 0, other.get("share_of_change") or 0
        if other["segment"] != child["segment"] and covers(other["segment"], child["segment"]):
            if cs >= CHILD_PARENT_SHARE * os_:
                other["superseded_by"] = child["id"]
        elif child["segment"] != other["segment"] and covers(child["segment"], other["segment"]):  # noqa: SIM102
            if os_ >= CHILD_PARENT_SHARE * cs:
                child["superseded_by"] = other["id"]


# --------------------------------------------------------------------- updates
def _from_data_quality(hyps: list[Hypothesis], ev: dict) -> None:
    for i, flag in enumerate(ev["data"]["flags"]):
        h = _find(hyps, flag["segment"], "data_quality") or _new(hyps, "data_quality", flag["segment"])
        h.update(
            share_of_change=flag.get("share_of_total_change"),
            share_evidence=(ev["id"], f"flags[{i}].share_of_total_change"),
            mechanism="data_quality",
            mechanism_evidence=ev["id"],
            confirmed=True,
            detail={"type": flag["type"], **{k: flag[k] for k in ("dates", "row_ratio") if k in flag}},
        )
        _cite(h, ev["id"])


def _from_seasonality(hyps: list[Hypothesis], ev: dict, state: InvestigationState) -> None:
    d = ev["data"]
    state["seasonality_verdict"] = d.get("verdict")
    if d.get("verdict") == "expected_seasonal":
        h = _find(hyps, {}, "seasonal") or _new(hyps, "seasonal", {})
        h.update(
            share_of_change=d.get("seasonal_share_of_change"),
            share_evidence=(ev["id"], "seasonal_share_of_change"),
            mechanism="seasonal",
            mechanism_evidence=ev["id"],
            confirmed=True,
            detail={"last_year_same_period_change_pct": d.get("last_year_same_period_change_pct")},
        )
        _cite(h, ev["id"])


def _from_contribution(hyps: list[Hypothesis], ev: dict) -> None:
    d = ev["data"]
    for i, row in enumerate(d["rows"]):
        share = row.get("share_of_total_change")
        conc = row.get("concentration")
        if row["delta"] is None or row["delta"] >= 0 or share is None or share < MIN_SHARE:
            continue
        if not is_concentrated(row):
            continue
        seg = {**d["filters"], d["dimension"]: row["value"]}
        h = _find(hyps, seg) or _new(hyps, "segment", seg)
        h.update(
            share_of_change=share,
            share_evidence=(ev["id"], f"rows[{i}].share_of_total_change"),
            pct_change=row["pct_change"],
            concentration=conc,
        )
        h["detail"]["pct_change_path"] = (ev["id"], f"rows[{i}].pct_change")
        h["detail"]["from_drill"] = bool(d["filters"])
        _cite(h, ev["id"])
        _supersede(hyps, h)


def _from_decomposition(hyps: list[Hypothesis], ev: dict) -> None:
    d = ev["data"]
    filters, dim = d["filters"], d["dimension"]
    effects = d["effects"]

    # (a) the filtered segment itself, e.g. decompose(product_category, {country: UK, channel: Online})
    if filters:
        h = _find(hyps, filters)
        if h is not None and h["mechanism"] is None:
            price_changes = [v["price_pct_change"] for v in d["values"] if v["price_pct_change"] is not None]
            med_price = median(price_changes) if price_changes else 0.0
            units = d.get("units_pct_change") or 0.0
            if med_price >= PRICE_CHANGE and units <= -UNITS_CHANGE:
                mech = "price"
            elif (effects.get("mix_share") or 0) >= MIX_SHARE and abs(units) < UNITS_CHANGE:
                mech = "mix"  # inside the segment, the mix moved to cheaper values
            else:
                mech = "volume"
            h.update(mechanism=mech, mechanism_evidence=ev["id"])
            h["detail"]["mechanism_paths"] = {
                "units_pct_change": (ev["id"], "units_pct_change"),
                "avg_price_pct_change": (ev["id"], "avg_price_pct_change"),
            }
            _cite(h, ev["id"])

    # (b) individual values of the decomposed dimension
    priced: set[str] = set()
    for i, v in enumerate(d["values"]):
        seg = {**filters, dim: v["value"]}
        pc, uc = v.get("price_pct_change"), v.get("units_pct_change")
        if pc is None or uc is None:
            continue
        if abs(pc) >= PRICE_CHANGE and uc <= -UNITS_CHANGE:
            h = _find(hyps, seg)
            if h is None:
                continue
            h.update(mechanism="price", mechanism_evidence=ev["id"])
            h["detail"]["mechanism_paths"] = {
                "price_pct_change": (ev["id"], f"values[{i}].price_pct_change"),
                "units_pct_change": (ev["id"], f"values[{i}].units_pct_change"),
            }
            _cite(h, ev["id"])
            priced.add(v["value"])

    # mix: total units roughly stable and the mix effect dominates; annotate the value whose
    # unit share fell the most (only existing hypotheses, so sub-segment noise cannot create causes)
    units_total = d.get("units_pct_change") or 0.0
    if (effects.get("mix_share") or 0) >= MIX_SHARE and abs(units_total) < UNITS_CHANGE:
        moved = [
            (i, v)
            for i, v in enumerate(d["values"])
            if v["value"] not in priced
            and (v.get("mix_effect") or 0) < 0
            and (v.get("unit_share_baseline") or 0) - (v.get("unit_share_current") or 0) >= MIX_MIN_SHARE_SHIFT
        ]
        if moved:
            i, v = min(moved, key=lambda t: t[1]["mix_effect"])
            h = _find(hyps, {**filters, dim: v["value"]})
            if h is not None:
                h.update(mechanism="mix", mechanism_evidence=ev["id"])
                h["detail"]["mechanism_paths"] = {
                    "mix_share": (ev["id"], "effects.mix_share"),
                    "units_pct_change": (ev["id"], "units_pct_change"),
                    "unit_share_baseline": (ev["id"], f"values[{i}].unit_share_baseline"),
                    "unit_share_current": (ev["id"], f"values[{i}].unit_share_current"),
                }
                _cite(h, ev["id"])

    # (c) remaining hypotheses at this level with clear unit loss -> volume
    for i, v in enumerate(d["values"]):
        h = _find(hyps, {**filters, dim: v["value"]})
        if h is None or h["mechanism"] is not None:
            continue
        if (v.get("units_pct_change") or 0) <= -UNITS_CHANGE:
            h.update(mechanism="volume", mechanism_evidence=ev["id"])
            h["detail"]["mechanism_paths"] = {"units_pct_change": (ev["id"], f"values[{i}].units_pct_change")}
            _cite(h, ev["id"])


def _from_significance(hyps: list[Hypothesis], ev: dict) -> None:
    d = ev["data"]
    for i, res in enumerate(d["results"]):
        h = _find(hyps, {**d["filters"], d["dimension"]: res["value"]})
        if h is None:
            continue
        h.update(significant=res["significant"], q_value=res["q_value"], significance_evidence=ev["id"])
        h["detail"]["q_value_path"] = (ev["id"], f"results[{i}].q_value")
        _cite(h, ev["id"])


def _refresh(hyps: list[Hypothesis]) -> None:
    for h in hyps:
        if h["kind"] == "segment":
            h["confirmed"] = bool(h["mechanism"]) and h["significant"] is True and not h["superseded_by"]


def active_hypotheses(hyps: list[Hypothesis]) -> list[Hypothesis]:
    """Non-superseded hypotheses, best first."""
    act = [h for h in hyps if not h["superseded_by"]]
    return sorted(act, key=lambda h: (not h["confirmed"], -(h.get("share_of_change") or 0)))


def explained_share(hyps: list[Hypothesis]) -> float:
    picked: list[Hypothesis] = []
    for h in active_hypotheses(hyps):
        if not h["confirmed"] or (h.get("share_of_change") or 0) <= 0:
            continue
        if any(
            covers(p["segment"], h["segment"]) or covers(h["segment"], p["segment"])
            for p in picked
            if p["kind"] == h["kind"] == "segment"
        ):
            continue
        picked.append(h)
    return round(min(1.0, sum(h["share_of_change"] or 0 for h in picked)), 4)


def update(state: InvestigationState, ev: dict) -> None:
    """Fold one evidence item into the state's hypotheses (in place)."""
    hyps = state.setdefault("hypotheses", [])
    if ev.get("ok"):
        handler = {
            "check_data_quality": lambda: _from_data_quality(hyps, ev),
            "check_seasonality": lambda: _from_seasonality(hyps, ev, state),
            "contribution_analysis": lambda: _from_contribution(hyps, ev),
            "drill_down": lambda: _from_contribution(hyps, ev),
            "decompose_revenue": lambda: _from_decomposition(hyps, ev),
            "significance_test": lambda: _from_significance(hyps, ev),
        }.get(ev["tool"])
        if handler:
            handler()
    _refresh(hyps)
    state["explained_share"] = explained_share(hyps)
