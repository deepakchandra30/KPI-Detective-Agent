"""check_seasonality and get_event_calendar.

check_seasonality compares this year's window-vs-baseline change with the
same weekday-aligned window 364 days earlier. If last year moved the same
way (a significant dip last year that matches this year's within noise, or
explains at least 75% of it), the drop is expected. Noise is measured as the robust (MAD) spread of
window-vs-baseline changes of the same length over the past year, using
only windows (and baselines) that touch no calendar event, so holiday
swings in the history do not inflate it.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import stats

from kpi_detective.data.calendar import events_between
from kpi_detective.tools.base import (
    ToolContext,
    baseline_dates_for,
    r,
    validate_metric,
    validate_window,
    window_dates,
)

YEAR = timedelta(days=364)  # weekday aligned
Z_EXPECTED = 2.5
#: Holiday timing shifts year to year, so a seasonal dip that explains at
#: least this share of the drop counts as expected even if not identical.
SEASONAL_SHARE_EXPECTED = 0.75


def _window_log_change(daily: pd.Series, start: date, end: date) -> float | None:
    cur, base = 0.0, 0.0
    for d in window_dates(start, end):
        bdays = baseline_dates_for(d, start)
        vals = [daily.get(pd.Timestamp(b)) for b in bdays]
        if daily.get(pd.Timestamp(d)) is None or any(v is None for v in vals):
            return None
        cur += float(daily[pd.Timestamp(d)])
        base += float(np.mean(vals))
    return float(np.log(cur / base)) if cur > 0 and base > 0 else None


def _quiet_window_noise(daily: pd.Series, start: date, length: int) -> float | None:
    """Robust spread of window-vs-baseline log changes in event-free windows of the past year."""
    hist = []
    s = start - timedelta(days=28 + 365)
    last = start - timedelta(days=28 + length)
    while s <= last:
        e = s + timedelta(days=length - 1)
        if not events_between(s - timedelta(days=28), e):
            v = _window_log_change(daily, s, e)
            if v is not None:
                hist.append(v)
        s += timedelta(days=1)
    if len(hist) < 30:
        return None
    return float(stats.median_abs_deviation(hist, scale="normal"))


def check_seasonality(ctx: ToolContext, metric: str = "revenue") -> dict:
    validate_window(ctx)
    validate_metric(metric)
    a = ctx.anomaly
    daily = ctx.store.daily_total(metric)
    length = a.days

    cur = _window_log_change(daily, a.start, a.end)
    ly = _window_log_change(daily, a.start - YEAR, a.end - YEAR)

    noise = _quiet_window_noise(daily, a.start, length)
    events = [e.to_dict() for e in events_between(a.start - timedelta(days=28), a.end)]
    out: dict = {
        "metric": metric,
        "current_change_pct": r(np.expm1(cur)) if cur is not None else None,
        "last_year_same_period_change_pct": r(np.expm1(ly)) if ly is not None else None,
        "last_year_window": [(a.start - YEAR).isoformat(), (a.end - YEAR).isoformat()],
        "noise_sd_log": r(noise),
        "events_near_window": events,
    }
    if cur is None or noise is None:
        out.update(verdict="insufficient_history", seasonal_share_of_change=None, z_raw=None, z_adjusted=None)
        return out

    z_raw = cur / noise
    out["z_raw"] = r(z_raw, 3)
    if ly is None:
        out.update(
            verdict="within_noise" if abs(z_raw) < Z_EXPECTED else "insufficient_history",
            seasonal_share_of_change=None,
            z_adjusted=None,
            adjusted_change_pct=None,
        )
        return out

    adj = cur - ly
    z_adj = adj / (noise * np.sqrt(2))  # difference of two independent window changes
    z_ly = ly / noise
    seasonal_share = float(np.clip(ly / cur, 0.0, 1.0)) if cur < 0 else 0.0
    last_year_had_dip = z_ly <= -Z_EXPECTED
    if abs(z_raw) < Z_EXPECTED:
        verdict = "within_noise"
    elif last_year_had_dip and (abs(z_adj) < Z_EXPECTED or seasonal_share >= SEASONAL_SHARE_EXPECTED):
        verdict = "expected_seasonal"
    else:
        verdict = "unusual"
    out.update(
        verdict=verdict,
        adjusted_change_pct=r(np.expm1(adj)),
        z_adjusted=r(z_adj, 3),
        z_last_year=r(z_ly, 3),
        seasonal_share_of_change=r(seasonal_share),
    )
    return out


def get_event_calendar(ctx: ToolContext) -> dict:
    """Known business events around the window, this year and last year."""
    a = ctx.anomaly
    this_year = events_between(a.start - timedelta(days=28), a.end)
    last_year = events_between(a.start - YEAR - timedelta(days=28), a.end - YEAR)
    return {
        "window": [a.start.isoformat(), a.end.isoformat()],
        "events_this_year": [e.to_dict() for e in this_year],
        "events_last_year": [e.to_dict() for e in last_year],
        "n_events": len(this_year),
    }
