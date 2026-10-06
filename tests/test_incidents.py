import pandas as pd
import pytest

from kpi_detective.data.generator import generate_sales
from kpi_detective.data.incidents import INCIDENT_TYPES, apply_incident, ground_truth


@pytest.fixture(scope="module")
def base():
    df = generate_sales(seed=0)
    return df[df["date"] <= "2025-06-30"].reset_index(drop=True)


def _sum(df, start, end, **seg):
    m = (df["date"] >= start) & (df["date"] <= end)
    for k, v in seg.items():
        m &= df[k] == v
    return df.loc[m]


def test_outage_hits_only_target(base):
    p = {"country": "US", "channel": "Online", "start": "2025-05-13", "days": 2, "drop": 0.6}
    out = apply_incident(base, "I1", p, seed=0)
    hit_before = _sum(base, "2025-05-13", "2025-05-14", country="US", channel="Online")["orders"].sum()
    hit_after = _sum(out, "2025-05-13", "2025-05-14", country="US", channel="Online")["orders"].sum()
    assert hit_after / hit_before == pytest.approx(0.4, abs=0.03)
    other = _sum(out, "2025-05-13", "2025-05-14", country="UK")["orders"].sum()
    assert other == _sum(base, "2025-05-13", "2025-05-14", country="UK")["orders"].sum()


def test_price_increase(base):
    p = {"category": "Laptops", "start": "2025-05-13", "increase": 0.10, "elasticity": -2.5, "extra_segment": {}}
    out = apply_incident(base, "I2", p, seed=0)
    b = _sum(base, "2025-05-13", "2025-05-20", product_category="Laptops")
    a = _sum(out, "2025-05-13", "2025-05-20", product_category="Laptops")
    assert (a["unit_price"].to_numpy() / b["unit_price"].to_numpy()).mean() == pytest.approx(1.10, abs=0.001)
    assert a["units"].sum() / b["units"].sum() == pytest.approx(1.1**-2.5, abs=0.03)


@pytest.mark.parametrize("mode", ["zero", "null"])
def test_tracking_bug(base, mode):
    p = {"channel": "Online", "start": "2025-05-13", "days": 1, "mode": mode, "extra_segment": {}}
    out = apply_incident(base, "I3", p, seed=0)
    rev = _sum(out, "2025-05-13", "2025-05-13", channel="Online")["revenue"]
    assert rev.isna().all() if mode == "null" else (rev == 0).all()
    assert _sum(out, "2025-05-13", "2025-05-13", channel="Online")["sessions"].sum() > 0


def test_mix_shift_keeps_units(base):
    p = {"start": "2025-05-13", "days": 5, "shift": 0.3}
    out = apply_incident(base, "I4", p, seed=0)
    b, a = _sum(base, "2025-05-13", "2025-05-17"), _sum(out, "2025-05-13", "2025-05-17")
    assert a["units"].sum() == pytest.approx(b["units"].sum(), rel=0.01)
    assert a["revenue"].sum() < 0.97 * b["revenue"].sum()


def test_gradual_decline_ramps(base):
    p = {"region": "NA", "category": "Monitors", "start": "2025-05-13", "days": 10, "final_drop": 0.8}
    out = apply_incident(base, "I5", p, seed=0)
    seg = {"region": "NA", "product_category": "Monitors"}
    first = (
        _sum(out, "2025-05-13", "2025-05-13", **seg)["units"].sum()
        / _sum(base, "2025-05-13", "2025-05-13", **seg)["units"].sum()
    )
    last = (
        _sum(out, "2025-05-22", "2025-05-22", **seg)["units"].sum()
        / _sum(base, "2025-05-22", "2025-05-22", **seg)["units"].sum()
    )
    assert first > 0.85 and last < 0.3


def test_late_load_drops_rows(base):
    out = apply_incident(base, "I6", {"missing_share": [0.5]}, seed=0)
    last = base["date"].max()
    assert (out["date"] == last).sum() == pytest.approx(0.5 * (base["date"] == last).sum(), abs=1)
    assert (out["date"] < last).sum() == (base["date"] < last).sum()


def test_false_alarm_is_noop(base):
    pd.testing.assert_frame_equal(apply_incident(base, "I8", {}, seed=0), base)


def test_ground_truth_covers_all_types():
    for t in INCIDENT_TYPES:
        params = {
            "I1": {"country": "UK"},
            "I2": {"category": "Laptops"},
            "I3": {"channel": "Online"},
            "I4": {},
            "I5": {"region": "NA"},
            "I6": {},
            "I8": {},
            "I7": {"outage": {"country": "UK"}, "price": {"category": "Servers"}},
        }[t]
        gt = ground_truth(t, params)
        assert gt["classification"] in {"business_change", "data_quality", "expected_seasonal"}
    assert ground_truth("I7", {"outage": {"country": "UK"}, "price": {"category": "Servers"}})["require_all"]
