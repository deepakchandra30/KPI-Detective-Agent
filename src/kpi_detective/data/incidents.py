"""Incident injection (PRD section 5.2).

Each incident is a small, pure function that takes the clean frame and a
parameter dict and returns a modified copy. Every incident also knows its
own ground truth, so the benchmark is fully self-describing.

Incident types
--------------
I1 regional_outage      Online orders in one country drop for a few days
I2 price_increase       Category price rises, units fall via elasticity
I3 tracking_bug         One channel's orders/revenue recorded as zero or NULL
I4 mix_shift            Enterprise share falls, Consumer rises, units stable
I5 competitor_promo     Gradual decline of one category in one region
I6 late_load            Most recent day only partially loaded
I7 outage_plus_price    I1 and I2 at the same time
I8 false_alarm          Nothing injected: a normal seasonal dip
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

INCIDENT_TYPES: dict[str, str] = {
    "I1": "regional_outage",
    "I2": "price_increase",
    "I3": "tracking_bug",
    "I4": "mix_shift",
    "I5": "competitor_promo",
    "I6": "late_load",
    "I7": "outage_plus_price",
    "I8": "false_alarm",
}


def _mask(frame: pd.DataFrame, segment: dict[str, str], start: str, end: str | None) -> pd.Series:
    m = frame["date"] >= pd.Timestamp(start)
    if end is not None:
        m &= frame["date"] <= pd.Timestamp(end)
    for dim, val in segment.items():
        m &= frame[dim] == val
    return m


def _thin(frame: pd.DataFrame, mask: pd.Series, keep: np.ndarray | float, rng: np.random.Generator) -> None:
    """Binomially thin orders/units/sessions in place, then recompute revenue."""
    keep_arr = np.broadcast_to(np.asarray(keep, dtype=float), (int(mask.sum()),))
    keep_arr = np.clip(keep_arr, 0.0, 1.0)
    for col in ("orders", "units"):
        frame.loc[mask, col] = rng.binomial(frame.loc[mask, col].to_numpy(), keep_arr)
    frame.loc[mask, "revenue"] = np.round(frame.loc[mask, "units"] * frame.loc[mask, "unit_price"], 2)


def _end(start: str, days: int) -> str:
    return (pd.Timestamp(start) + pd.Timedelta(days=days - 1)).date().isoformat()


# --------------------------------------------------------------------------- I1
def regional_outage(frame: pd.DataFrame, p: dict, rng: np.random.Generator) -> pd.DataFrame:
    seg = {"country": p["country"], "channel": p.get("channel", "Online")}
    m = _mask(frame, seg, p["start"], _end(p["start"], p["days"]))
    _thin(frame, m, 1 - p["drop"], rng)
    frame.loc[m, "sessions"] = rng.binomial(frame.loc[m, "sessions"].to_numpy(), 1 - p["drop"])
    return frame


# --------------------------------------------------------------------------- I2
def price_increase(frame: pd.DataFrame, p: dict, rng: np.random.Generator) -> pd.DataFrame:
    seg = {"product_category": p["category"], **p.get("extra_segment", {})}
    m = _mask(frame, seg, p["start"], None)
    keep = (1 + p["increase"]) ** p["elasticity"]
    frame.loc[m, "unit_price"] = np.round(frame.loc[m, "unit_price"] * (1 + p["increase"]), 2)
    _thin(frame, m, keep, rng)
    return frame


# --------------------------------------------------------------------------- I3
def tracking_bug(frame: pd.DataFrame, p: dict, rng: np.random.Generator) -> pd.DataFrame:
    seg = {"channel": p["channel"], **p.get("extra_segment", {})}
    m = _mask(frame, seg, p["start"], _end(p["start"], p["days"]))
    if p.get("mode", "zero") == "null":
        frame["revenue"] = frame["revenue"].astype("float64")
        frame.loc[m, "revenue"] = np.nan
    else:
        frame.loc[m, ["orders", "units"]] = 0
        frame.loc[m, "revenue"] = 0.0
    return frame


# --------------------------------------------------------------------------- I4
def mix_shift(frame: pd.DataFrame, p: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Move a share of Enterprise units to Consumer, keeping daily units constant."""
    end = _end(p["start"], p["days"])
    ent = _mask(frame, {"customer_segment": "Enterprise"}, p["start"], end)
    con = _mask(frame, {"customer_segment": "Consumer"}, p["start"], end)
    ent_units = frame.loc[ent].groupby("date")["units"].sum()
    con_units = frame.loc[con].groupby("date")["units"].sum()
    _thin(frame, ent, 1 - p["shift"], rng)
    lost = ent_units - frame.loc[ent].groupby("date")["units"].sum()
    factor = 1 + lost / con_units
    f = frame.loc[con, "date"].map(factor).to_numpy()
    for col in ("orders", "units"):
        frame.loc[con, col] = np.round(frame.loc[con, col].to_numpy() * f).astype("int64")
    frame.loc[con, "revenue"] = np.round(frame.loc[con, "units"] * frame.loc[con, "unit_price"], 2)
    return frame


# --------------------------------------------------------------------------- I5
def competitor_promo(frame: pd.DataFrame, p: dict, rng: np.random.Generator) -> pd.DataFrame:
    seg = {"region": p["region"], "product_category": p.get("category", "Monitors")}
    m = _mask(frame, seg, p["start"], _end(p["start"], p["days"]))
    day_idx = (frame.loc[m, "date"] - pd.Timestamp(p["start"])).dt.days.to_numpy()
    keep = 1 - p["final_drop"] * (day_idx + 1) / p["days"]
    _thin(frame, m, keep, rng)
    return frame


# --------------------------------------------------------------------------- I6
def late_load(frame: pd.DataFrame, p: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Drop a share of rows on the most recent day(s): an incomplete ETL load."""
    last = frame["date"].max()
    drop_idx: list[int] = []
    for offset, share in enumerate(p["missing_share"]):
        day = last - pd.Timedelta(days=offset)
        idx = frame.index[frame["date"] == day].to_numpy()
        n = int(round(len(idx) * share))
        drop_idx.extend(rng.choice(idx, size=n, replace=False).tolist())
    return frame.drop(index=drop_idx).reset_index(drop=True)


# --------------------------------------------------------------------------- I7
def outage_plus_price(frame: pd.DataFrame, p: dict, rng: np.random.Generator) -> pd.DataFrame:
    frame = price_increase(frame, p["price"], rng)
    return regional_outage(frame, p["outage"], rng)


def false_alarm(frame: pd.DataFrame, p: dict, rng: np.random.Generator) -> pd.DataFrame:
    return frame


_APPLY: dict[str, Callable[[pd.DataFrame, dict, np.random.Generator], pd.DataFrame]] = {
    "I1": regional_outage,
    "I2": price_increase,
    "I3": tracking_bug,
    "I4": mix_shift,
    "I5": competitor_promo,
    "I6": late_load,
    "I7": outage_plus_price,
    "I8": false_alarm,
}


def apply_incident(frame: pd.DataFrame, incident_type: str, params: dict, seed: int) -> pd.DataFrame:
    """Return a copy of ``frame`` with the incident injected."""
    if incident_type not in _APPLY:
        raise ValueError(f"Unknown incident type {incident_type!r}")
    rng = np.random.default_rng(seed + 10_000)
    return _APPLY[incident_type](frame.copy(), params, rng)


def ground_truth(incident_type: str, params: dict) -> dict[str, Any]:
    """Ground truth label for an incident, used by the frozen matching rules."""
    if incident_type == "I1":
        causes = [
            {"segment": {"country": params["country"], "channel": params.get("channel", "Online")}, "mechanism": None}
        ]
        return {"classification": "business_change", "causes": causes, "require_all": False}
    if incident_type == "I2":
        seg = {"product_category": params["category"], **params.get("extra_segment", {})}
        return {
            "classification": "business_change",
            "causes": [{"segment": seg, "mechanism": "price"}],
            "require_all": False,
        }
    if incident_type == "I3":
        seg = {"channel": params["channel"], **params.get("extra_segment", {})}
        return {
            "classification": "data_quality",
            "causes": [{"segment": seg, "mechanism": "data_quality"}],
            "require_all": False,
        }
    if incident_type == "I4":
        return {
            "classification": "business_change",
            "causes": [{"segment": {"customer_segment": "Enterprise"}, "mechanism": "mix"}],
            "require_all": False,
        }
    if incident_type == "I5":
        seg = {"region": params["region"], "product_category": params.get("category", "Monitors")}
        return {
            "classification": "business_change",
            "causes": [{"segment": seg, "mechanism": None}],
            "require_all": False,
        }
    if incident_type == "I6":
        return {
            "classification": "data_quality",
            "causes": [{"segment": {}, "mechanism": "data_quality"}],
            "require_all": False,
        }
    if incident_type == "I7":
        out = ground_truth("I1", params["outage"])["causes"] + ground_truth("I2", params["price"])["causes"]
        return {"classification": "business_change", "causes": out, "require_all": True}
    if incident_type == "I8":
        return {"classification": "expected_seasonal", "causes": [], "require_all": False}
    raise ValueError(incident_type)
