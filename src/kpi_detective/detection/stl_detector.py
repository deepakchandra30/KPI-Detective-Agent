"""STL decomposition + robust z-score anomaly detection (FR1, FR2, FR4).

The KPI is modelled in log space, so weekly and yearly effects are
multiplicative. Residuals are scored with a robust z-score (median and MAD),
and consecutive flagged days are merged into :class:`AnomalyEvent` objects.

With at least two years of history MSTL fits weekly and yearly seasonality
separately; with less, a robust weekly STL with a long trend window is used
and the method name says so. With only 2-3 cycles, an in-sample yearly
component memorises each day of the year (noise included), so residuals
are computed against *last year's* yearly component. That keeps the
residual scale honest instead of near zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL, STL

from kpi_detective.models import AnomalyEvent

Z_THRESHOLD = 3.5
MAD_SCALE = 1.4826


@dataclass
class DetectionResult:
    """Per-day scores plus merged anomaly events."""

    frame: pd.DataFrame  # date, actual, expected, lower, upper, score, flag
    events: list[AnomalyEvent] = field(default_factory=list)
    method: str = ""


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * MAD_SCALE
    return (x - med) / (mad if mad > 0 else 1e-9)


def merge_events(frame: pd.DataFrame, metric: str, method: str, direction: str = "drop") -> list[AnomalyEvent]:
    """Merge consecutive flagged days into anomaly events."""
    flagged = frame[frame["flag"]]
    if direction == "drop":
        flagged = flagged[flagged["actual"] < flagged["expected"]]
    events: list[AnomalyEvent] = []
    if flagged.empty:
        return events
    groups = (flagged["date"].diff().dt.days != 1).cumsum()
    for _, g in flagged.groupby(groups):
        events.append(
            AnomalyEvent(
                metric=metric,
                start=g["date"].iloc[0].date(),
                end=g["date"].iloc[-1].date(),
                expected=round(float(g["expected"].sum()), 2),
                actual=round(float(g["actual"].sum()), 2),
                method=method,
                score=round(float(g["score"].abs().max()), 3),
            )
        )
    return events


def _decompose(y: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, str]:
    """Return trend, weekly, yearly components (yearly is zero without 2 years)."""
    if len(y) >= 2 * 365 + 7:
        res = MSTL(y, periods=(7, 365), stl_kwargs={"robust": True}).fit()
        return res.trend, res.seasonal["seasonal_7"], res.seasonal["seasonal_365"], "mstl_weekly_yearly"
    res = STL(y, period=7, trend=91, robust=True).fit()
    return res.trend, res.seasonal, pd.Series(0.0, index=y.index), "stl_weekly"


def detect_stl(
    series: pd.Series,
    metric: str = "revenue",
    threshold: float = Z_THRESHOLD,
    horizon: int = 14,
    direction: str = "drop",
) -> DetectionResult:
    """Flag days whose STL residual has |robust z| > threshold.

    STL is fitted on everything except the last ``horizon`` days. Those days
    are scored out of sample against the extrapolated decomposition
    (trend continued linearly from its last 4 weeks, weekly component from
    one week earlier, yearly component from one year earlier). Without
    this, a robust STL fitted through the end of the series quietly absorbs
    a fresh anomaly into its trend: the classic end-point problem of
    real-time decomposition.
    """
    s = series.asfreq("D").astype(float)
    if s.isna().any():
        s = s.interpolate(limit_direction="both")
    y = np.log(s.clip(lower=1.0))
    train = y.iloc[:-horizon]
    trend, weekly, yearly, method = _decompose(train)

    n = len(y)
    tail = trend.iloc[-28:]
    slope = np.polyfit(np.arange(len(tail)), tail.to_numpy(), 1)[0]
    trend_full = np.concatenate([trend.to_numpy(), trend.iloc[-1] + slope * np.arange(1, horizon + 1)])
    weekly_full = np.concatenate([weekly.to_numpy(), np.zeros(horizon)])
    for i in range(len(train), n):
        weekly_full[i] = weekly_full[i - 7]
    yearly_in = np.concatenate([yearly.to_numpy(), np.zeros(horizon)])
    yearly_lag = np.zeros(n)
    has_yearly = method == "mstl_weekly_yearly"
    if has_yearly:
        yearly_lag[365:] = yearly_in[:-365]
    expected_log = trend_full + weekly_full + yearly_lag

    resid = y.to_numpy() - expected_log
    scale_from = 365 if has_yearly else 0
    resid_train = resid[scale_from : len(train)]
    med = float(np.median(resid_train))
    mad = float(np.median(np.abs(resid_train - med)) * MAD_SCALE) or 1e-9
    z = (resid - med) / mad
    if has_yearly:
        z[:365] = 0.0  # no out-of-sample yearly reference in the first year

    frame = pd.DataFrame(
        {
            "date": s.index,
            "actual": series.reindex(s.index).to_numpy(),
            "expected": np.exp(expected_log + med),
            "lower": np.exp(expected_log + med - threshold * mad),
            "upper": np.exp(expected_log + med + threshold * mad),
            "score": z,
        }
    )
    frame["flag"] = np.abs(frame["score"]) > threshold
    return DetectionResult(frame, merge_events(frame, metric, method, direction), method)
