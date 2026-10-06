"""The naive alert most BI tools fire: window total vs same weekday, last 4 weeks.

It is deliberately simple. It catches real incidents but also fires on
normal seasonal dips (the I8 false alarms), which is exactly what the agent
must learn to see through.
"""

from __future__ import annotations

from datetime import date

from kpi_detective.models import AnomalyEvent
from kpi_detective.store import SalesStore
from kpi_detective.tools.base import ToolContext, aggregate


def baseline_alert(store: SalesStore, metric: str, start: date, end: date) -> AnomalyEvent:
    placeholder = AnomalyEvent(metric, start, end, expected=0.0, actual=0.0)
    totals = aggregate(ToolContext(store, placeholder), [metric], [], {}).iloc[0]
    return AnomalyEvent(
        metric=metric,
        start=start,
        end=end,
        expected=round(float(totals[f"{metric}_baseline"]), 2),
        actual=round(float(totals[f"{metric}_current"]), 2),
        method="same_weekday_4w_baseline",
    )
