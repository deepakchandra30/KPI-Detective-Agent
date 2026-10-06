"""Critic: every number in the report must trace back to a tool output (FR8).

Two kinds of numbers are checked:

1. **Claims**: structured ``{"value", "evidence_id", "path"}`` entries. The
   critic resolves ``path`` inside the cited evidence and compares values.
2. **Numbers in prose** (descriptions, headline, recommended action). Each
   must be a faithful rounding of some verified claim, e.g. "18%" is
   supported by a claim of -0.1814, "4.8M" by 4,812,300.

Anything else counts as an unsupported claim. The unsupported claim rate
reported in the benchmark is ``unsupported / total numbers``.
"""

from __future__ import annotations

import re
from typing import Any

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUMBER = re.compile(r"(?<![\w.])([-+−]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|[kKmMbB]\b)?")
_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")
_SCALE = {"%": 0.01, "k": 1e3, "K": 1e3, "m": 1e6, "M": 1e6, "b": 1e9, "B": 1e9}
TEXT_FIELDS = ("headline", "recommended_action")


def resolve(data: Any, path: str) -> Any:
    """Resolve a path like ``rows[0].share_of_total_change`` inside a JSON object."""
    cur = data
    for key, idx in _PATH_TOKEN.findall(path):
        if key:
            if not isinstance(cur, dict) or key not in cur:
                raise KeyError(path)
            cur = cur[key]
        else:
            i = int(idx)
            if not isinstance(cur, list) or i >= len(cur):
                raise KeyError(path)
            cur = cur[i]
    return cur


def _claim_ok(value: float, actual: float) -> bool:
    tol = max(0.006 * abs(actual), 0.005 if abs(actual) <= 1 else 0.0, 1e-9)
    return abs(value - actual) <= tol


def extract_numbers(text: str) -> list[tuple[float, float, str]]:
    """Numbers in prose as (value, rounding tolerance, raw text). Dates, years and small counts are skipped."""
    text = _DATE.sub(" ", text)
    out = []
    for m in _NUMBER.finditer(text):
        raw, unit = m.group(1).replace("−", "-").replace(",", ""), m.group(2)
        num = float(raw)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if unit is None and "." not in raw and (abs(num) <= 12 or 1900 <= abs(num) <= 2100):
            continue
        scale = _SCALE.get(unit or "", 1.0)
        tol = 0.5 * 10 ** (-decimals) * scale
        out.append((num * scale, tol * 1.02 + 1e-12, m.group(0).strip()))
    return out


def _texts(report: dict) -> list[str]:
    texts = [str(report.get(f, "")) for f in TEXT_FIELDS]
    texts += [str(c.get("description", "")) for c in report.get("causes", [])]
    return texts


def verify_report(report: dict, evidence: dict[str, dict]) -> dict:
    """Check a draft report against evidence. ``evidence`` maps id -> evidence item."""
    issues: list[str] = []
    verified_values: list[float] = []
    n_claims = n_bad_claims = 0

    for ci, cause in enumerate(report.get("causes", [])):
        claims = cause.get("claims", [])
        if not claims:
            issues.append(f"cause {ci + 1}: no claims cited")
        for eid in cause.get("evidence_ids", []):
            if eid not in evidence:
                issues.append(f"cause {ci + 1}: evidence {eid} does not exist")
        share_ok = cause.get("share_of_change") is None
        for cl in claims:
            n_claims += 1
            eid, path = cl.get("evidence_id"), cl.get("path", "")
            try:
                value = float(cl["value"])
                item = evidence[eid]
                actual = resolve(item.get("data", item), path)
                actual = float(actual)
            except (KeyError, TypeError, ValueError):
                n_bad_claims += 1
                issues.append(f"cause {ci + 1}: claim {cl} cannot be resolved")
                continue
            if _claim_ok(value, actual):
                verified_values.append(actual)
                if cause.get("share_of_change") is not None and _claim_ok(float(cause["share_of_change"]), actual):
                    share_ok = True
            else:
                n_bad_claims += 1
                issues.append(f"cause {ci + 1}: claim {cl.get('label', path)}={value} but {eid}.{path}={actual}")
        if not share_ok:
            n_bad_claims += 1
            n_claims += 1
            issues.append(f"cause {ci + 1}: share_of_change {cause.get('share_of_change')} is not backed by a claim")

    for cl in report.get("claims", []):  # report-level claims (headline numbers)
        n_claims += 1
        try:
            item = evidence[cl["evidence_id"]]
            actual = float(resolve(item.get("data", item), cl["path"]))
            if _claim_ok(float(cl["value"]), actual):
                verified_values.append(actual)
                continue
        except (KeyError, TypeError, ValueError):
            pass
        n_bad_claims += 1
        issues.append(f"report claim {cl} not supported")

    n_text = n_bad_text = 0
    for text in _texts(report):
        for value, tol, raw in extract_numbers(text):
            n_text += 1
            if not any(abs(abs(value) - abs(v)) <= tol for v in verified_values):
                n_bad_text += 1
                issues.append(f"number '{raw}' in text is not backed by any verified claim")

    total = n_claims + n_text
    unsupported = n_bad_claims + n_bad_text
    return {
        "verified": unsupported == 0 and not issues,
        "issues": issues,
        "n_numbers": total,
        "n_unsupported": unsupported,
        "unsupported_rate": round(unsupported / total, 4) if total else 0.0,
    }
