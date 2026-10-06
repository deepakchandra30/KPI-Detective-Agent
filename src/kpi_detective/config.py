"""Shared constants: business dimensions, metrics and guardrail limits.

Everything that the tools validate against lives here, so the allowed
vocabulary for the agent is defined in exactly one place.
"""

from __future__ import annotations

REGIONS: dict[str, list[str]] = {
    "NA": ["US", "Canada"],
    "EMEA": ["Ireland", "Germany", "UK", "France"],
    "APAC": ["Japan", "Australia", "India"],
    "LATAM": ["Brazil", "Mexico"],
}
COUNTRY_TO_REGION: dict[str, str] = {c: r for r, cs in REGIONS.items() for c in cs}
COUNTRIES: list[str] = list(COUNTRY_TO_REGION)

CHANNELS: list[str] = ["Online", "Retail Partner", "Direct Sales"]
CATEGORIES: list[str] = ["Laptops", "Monitors", "Servers", "Accessories"]
SEGMENTS: list[str] = ["Consumer", "Small Business", "Enterprise"]

#: Dimensions the agent may slice by, in hierarchy-aware order.
DIMENSIONS: list[str] = ["region", "country", "channel", "product_category", "customer_segment"]

#: Parent dimension for hierarchical dimensions (used for drill-down and matching).
DIMENSION_PARENT: dict[str, str] = {"country": "region"}

#: Additive metrics. Ratio metrics are derived inside the tools.
ADDITIVE_METRICS: list[str] = ["revenue", "units", "orders", "sessions"]

DIMENSION_VALUES: dict[str, list[str]] = {
    "region": list(REGIONS),
    "country": COUNTRIES,
    "channel": CHANNELS,
    "product_category": CATEGORIES,
    "customer_segment": SEGMENTS,
}

#: Guardrails from the PRD (section 8).
MAX_TOOL_CALLS = 12
EXPLAINED_SHARE_TARGET = 0.80
MAX_REPORT_REVISIONS = 2

#: Baseline definition used by every tool: same weekday, previous 4 weeks.
BASELINE_WEEKS = 4

TABLE = "sales"
