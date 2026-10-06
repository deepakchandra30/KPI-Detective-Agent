"""Tool registry: names, schemas shown to the LLM, and safe execution.

Every call returns structured JSON. Invalid input never raises into the
agent loop: it comes back as ``{"error": ...}`` so the planner can correct
itself, and it still counts against the step budget.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kpi_detective.config import ADDITIVE_METRICS, DIMENSIONS
from kpi_detective.store import ReadOnlyViolation
from kpi_detective.tools.base import ToolContext, ToolError
from kpi_detective.tools.contribution import compare_periods, contribution_analysis, drill_down
from kpi_detective.tools.data_quality import check_data_quality
from kpi_detective.tools.pvm_decomposition import decompose_revenue
from kpi_detective.tools.seasonality import check_seasonality, get_event_calendar
from kpi_detective.tools.significance import significance_test

_METRIC = {"type": "string", "enum": ADDITIVE_METRICS, "default": "revenue"}
_DIM = {"type": "string", "enum": DIMENSIONS}
_FILTERS = {
    "type": "object",
    "description": 'Segment to stay inside, e.g. {"region": "EMEA"}. Keys must be dimensions.',
    "additionalProperties": {"type": "string"},
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., dict]

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or [], "additionalProperties": False}


TOOLS: dict[str, ToolSpec] = {
    t.name: t
    for t in [
        ToolSpec(
            "check_data_quality",
            "Detect missing or partial loads, NULLs and segments whose revenue became all zero/NULL. Always run first.",
            _obj({}),
            check_data_quality,
        ),
        ToolSpec(
            "check_seasonality",
            "Compare the drop with the same weekday-aligned period last year. Verdicts: "
            "expected_seasonal, within_noise, unusual, insufficient_history.",
            _obj({"metric": _METRIC}),
            check_seasonality,
        ),
        ToolSpec(
            "get_event_calendar",
            "Known holidays and business events around the window, this year and last year.",
            _obj({}),
            get_event_calendar,
        ),
        ToolSpec(
            "compare_periods",
            "Window vs baseline totals for all metrics plus avg unit price, conversion rate and "
            "units per order; optional grouping by up to 2 dimensions.",
            _obj({"metric": _METRIC, "dimensions": {"type": "array", "items": _DIM, "maxItems": 2}}),
            compare_periods,
        ),
        ToolSpec(
            "contribution_analysis",
            "Split the change of a metric across the values of one dimension. Returns delta, "
            "share_of_total_change and concentration (>1 means the drop is concentrated there).",
            _obj({"metric": _METRIC, "dimension": _DIM, "filters": _FILTERS}, ["dimension"]),
            contribution_analysis,
        ),
        ToolSpec(
            "drill_down",
            'Contribution analysis inside a segment, e.g. filters {"region": "EMEA"} by country.',
            _obj({"metric": _METRIC, "filters": _FILTERS, "dimension": _DIM}, ["filters", "dimension"]),
            drill_down,
        ),
        ToolSpec(
            "decompose_revenue",
            "Price-volume-mix decomposition of the revenue change across one dimension. Shows "
            "per-value price and unit changes.",
            _obj({"dimension": _DIM, "filters": _FILTERS}, ["dimension"]),
            decompose_revenue,
        ),
        ToolSpec(
            "significance_test",
            "Is each value's change beyond its normal daily variation? Mann-Whitney U or robust z "
            "vs up to a year of event-free history, Benjamini-Hochberg corrected.",
            _obj(
                {
                    "metric": _METRIC,
                    "dimension": _DIM,
                    "filters": _FILTERS,
                    "values": {"type": "array", "items": {"type": "string"}},
                },
                ["dimension"],
            ),
            significance_test,
        ),
    ]
}


def run_tool(ctx: ToolContext, name: str, args: dict[str, Any] | None = None) -> dict:
    """Execute a tool safely. Returns ``{"tool", "args", "ok", "data"|"error", "ms"}``."""
    args = dict(args or {})
    t0 = time.perf_counter()
    spec = TOOLS.get(name)
    if spec is None:
        return {
            "tool": name,
            "args": args,
            "ok": False,
            "error": f"unknown tool {name!r}; available: {list(TOOLS)}",
            "ms": 0.0,
        }
    allowed = set(spec.parameters["properties"])
    unknown = set(args) - allowed
    if unknown:
        return {
            "tool": name,
            "args": args,
            "ok": False,
            "error": f"unknown arguments {sorted(unknown)}; allowed: {sorted(allowed)}",
            "ms": 0.0,
        }
    try:
        data = spec.fn(ctx, **args)
        ok, payload = True, {"data": data}
    except (ToolError, ReadOnlyViolation, TypeError) as exc:
        ok, payload = False, {"error": str(exc)}
    return {"tool": name, "args": args, "ok": ok, **payload, "ms": round((time.perf_counter() - t0) * 1000, 1)}
