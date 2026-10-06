"""Frozen matching rules (PRD 9.2). Written before any benchmark run.

1. A predicted cause matches a ground-truth cause when its segment equals
   the true segment exactly, after normalisation (region is dropped when
   implied by the country). Too broad ({country: UK} for a UK *online*
   outage) and too narrow ({UK, Online, Laptops}) are both wrong.
2. If the ground truth specifies a mechanism (price for I2, mix for I4,
   data_quality for I3/I6), the predicted mechanism must equal it.
3. Late loads (I6) have an empty segment: the cause must have mechanism
   data_quality and an empty segment.
4. Top-1: the first cause matches a true cause. Top-3: a true cause is in
   the first three; for two simultaneous causes (I7) *both* must be in the
   first three for top-3, and either may be first for top-1.
5. ``top1_segment`` is a lenient secondary metric: rule 1 only, mechanism
   ignored. It shows how often a system points at the right segment even
   when it cannot say why.
6. False alarms (I8) have no cause: top-1 and top-3 are correct exactly
   when the classification is ``expected_seasonal``.
"""

from __future__ import annotations

from kpi_detective.agent.segments import normalize


def cause_matches(pred: dict, truth: dict, check_mechanism: bool = True) -> bool:
    if normalize(pred.get("segment") or {}) != normalize(truth["segment"]):
        return False
    if not check_mechanism:
        return True
    return truth.get("mechanism") is None or pred.get("mechanism") == truth["mechanism"]


def score_report(report: dict, ground_truth: dict) -> dict:
    pred_causes = report.get("causes", [])
    true_causes = ground_truth["causes"]
    cls_ok = report.get("classification") == ground_truth["classification"]
    if not true_causes:
        top1 = top3 = cls_ok
    else:
        top1 = bool(pred_causes) and any(cause_matches(pred_causes[0], t) for t in true_causes)
        hits = [any(cause_matches(p, t) for p in pred_causes[:3]) for t in true_causes]
        top3 = all(hits) if ground_truth.get("require_all") else any(hits)
    # lenient variant: right segment, mechanism ignored (what a pivot table can find)
    if not true_causes:
        seg1 = cls_ok
    else:
        seg1 = bool(pred_causes) and any(cause_matches(pred_causes[0], t, check_mechanism=False) for t in true_causes)
    return {"top1": bool(top1), "top3": bool(top3), "top1_segment": bool(seg1), "classification_ok": bool(cls_ok)}
