"""Helpers for segment dictionaries like ``{"country": "UK", "channel": "Online"}``."""

from __future__ import annotations

from kpi_detective.config import COUNTRY_TO_REGION, DIMENSIONS


def normalize(segment: dict[str, str]) -> dict[str, str]:
    """Canonical form: region is dropped when the country implies it; fixed key order."""
    seg = {k: str(v) for k, v in segment.items() if v is not None}
    if "country" in seg and "region" in seg and COUNTRY_TO_REGION.get(seg["country"]) == seg["region"]:
        seg.pop("region")
    return {d: seg[d] for d in DIMENSIONS if d in seg}


def covers(parent: dict[str, str], child: dict[str, str]) -> bool:
    """True if every row of ``child`` lies inside ``parent`` (parent is coarser or equal)."""
    for dim, val in parent.items():
        if child.get(dim) == val:
            continue
        if dim == "region" and COUNTRY_TO_REGION.get(child.get("country", "")) == val:
            continue
        return False
    return True


def overlaps(a: dict[str, str], b: dict[str, str]) -> bool:
    """True if the two segments can share rows (no conflicting dimension values)."""
    a_n, b_n = dict(a), dict(b)
    for seg in (a_n, b_n):
        if "country" in seg:
            seg.setdefault("region", COUNTRY_TO_REGION.get(seg["country"], ""))
    return all(b_n[d] == v for d, v in a_n.items() if d in b_n)


def label(segment: dict[str, str]) -> str:
    if not segment:
        return "all data"
    return ", ".join(f"{k}={v}" for k, v in normalize(segment).items())
