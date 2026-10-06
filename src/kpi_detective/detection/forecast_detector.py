"""Forecast-based detection (FR3).

A regression forecaster on log revenue: trend, day-of-week, yearly Fourier
terms and known calendar events, each event interacted with weekend/weekday
because B2B events barely move weekend sales. It is fitted on history up to
``horizon`` days before the end of the series, then the last ``horizon``
days are flagged when they fall outside the 95% prediction interval.

This is the role Prophet plays in the PRD, built with NumPy only, so it is
transparent, fast and has no heavy dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from kpi_detective.data.calendar import events_between
from kpi_detective.detection.stl_detector import DetectionResult, merge_events

N_FOURIER = 6


def design_matrix(dates: pd.DatetimeIndex, origin: pd.Timestamp) -> pd.DataFrame:
    t = (dates - origin).days.to_numpy() / 365.25
    X = {"const": np.ones(len(dates)), "trend": t}
    for d in range(1, 7):
        X[f"dow_{d}"] = (dates.dayofweek == d).astype(float)
    doy = dates.dayofyear.to_numpy() / 365.25
    for k in range(1, N_FOURIER + 1):
        X[f"sin_{k}"] = np.sin(2 * np.pi * k * doy)
        X[f"cos_{k}"] = np.cos(2 * np.pi * k * doy)
    weekend = (dates.dayofweek >= 5).astype(float)
    names: set[str] = set()
    evs = events_between(dates.min().date(), dates.max().date())
    for e in evs:
        key = e.name.split(" Q")[0]  # all quarter ends share one effect
        names.add(key)
        X.setdefault(f"ev_{key}_wd", np.zeros(len(dates)))
        X.setdefault(f"ev_{key}_we", np.zeros(len(dates)))
        m = ((dates.date >= e.start) & (dates.date <= e.end)).astype(float)
        X[f"ev_{key}_wd"] = np.maximum(X[f"ev_{key}_wd"], m * (1 - weekend))
        X[f"ev_{key}_we"] = np.maximum(X[f"ev_{key}_we"], m * weekend)
    return pd.DataFrame(X, index=dates)


def detect_forecast(
    series: pd.Series, metric: str = "revenue", horizon: int = 14, level: float = 0.95, direction: str = "drop"
) -> DetectionResult:
    s = series.asfreq("D").astype(float).interpolate(limit_direction="both")
    y = np.log(s.clip(lower=1.0))
    X = design_matrix(s.index, s.index[0])
    X = X.loc[:, X.iloc[:-horizon].abs().sum() > 0]  # drop features never seen in training
    train_X, train_y = X.iloc[:-horizon].to_numpy(), y.iloc[:-horizon].to_numpy()

    beta, *_ = np.linalg.lstsq(train_X, train_y, rcond=None)
    fitted = X.to_numpy() @ beta
    resid = train_y - fitted[:-horizon]
    dof = max(len(train_y) - train_X.shape[1], 1)
    sigma = float(np.sqrt(np.sum(resid**2) / dof))
    q = stats.t.ppf(0.5 + level / 2, dof)

    frame = pd.DataFrame(
        {
            "date": s.index,
            "actual": series.reindex(s.index).to_numpy(),
            "expected": np.exp(fitted),
            "lower": np.exp(fitted - q * sigma),
            "upper": np.exp(fitted + q * sigma),
            "score": (y.to_numpy() - fitted) / sigma,
        }
    )
    frame["flag"] = False
    test = frame.index >= len(frame) - horizon
    frame.loc[test, "flag"] = (frame.loc[test, "actual"] < frame.loc[test, "lower"]) | (
        frame.loc[test, "actual"] > frame.loc[test, "upper"]
    )
    return DetectionResult(frame, merge_events(frame, metric, "forecast_regression", direction), "forecast_regression")
