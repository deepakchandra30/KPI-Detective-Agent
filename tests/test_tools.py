from datetime import date

import pytest
from statsmodels.stats.multitest import multipletests

from kpi_detective.store import ReadOnlyViolation
from kpi_detective.tools.base import baseline_dates_for, period_weights
from kpi_detective.tools.registry import TOOLS, run_tool
from kpi_detective.tools.significance import benjamini_hochberg


def test_baseline_never_overlaps_window():
    start = date(2025, 3, 3)
    for offset in range(10):
        d = date.fromordinal(start.toordinal() + offset)
        base = baseline_dates_for(d, start)
        assert len(base) == 4 and all(b < start for b in base)
        assert all((d - b).days % 7 == 0 for b in base)
    _, weights = period_weights(start, date(2025, 3, 12))
    assert sum(weights.values()) == pytest.approx(10)  # 10 window days x 4 dates / 4


def test_every_tool_returns_json(ctx_for):
    ctx = ctx_for("I1_01")
    args = {
        "check_data_quality": {},
        "check_seasonality": {},
        "get_event_calendar": {},
        "compare_periods": {"dimensions": ["region"]},
        "contribution_analysis": {"dimension": "channel"},
        "drill_down": {"filters": {"region": "NA"}, "dimension": "country"},
        "decompose_revenue": {"dimension": "product_category"},
        "significance_test": {"dimension": "channel"},
    }
    assert set(args) == set(TOOLS)
    import json

    for name, a in args.items():
        out = run_tool(ctx, name, a)
        assert out["ok"], (name, out.get("error"))
        json.dumps(out["data"])  # serialisable, no NaN objects


def test_input_validation_is_returned_not_raised(ctx_for):
    ctx = ctx_for("I1_01")
    assert not run_tool(ctx, "contribution_analysis", {"dimension": "planet"})["ok"]
    assert not run_tool(ctx, "contribution_analysis", {"dimension": "country", "filters": {"region": "Mars"}})["ok"]
    assert not run_tool(ctx, "drill_down", {"dimension": "country"})["ok"]
    assert not run_tool(ctx, "contribution_analysis", {"dimension": "channel", "sql": "DROP"})["ok"]
    assert not run_tool(ctx, "no_such_tool", {})["ok"]


def test_store_is_read_only(store_factory):
    store = store_factory("I1_01")
    with pytest.raises(ReadOnlyViolation):
        store.query("DELETE FROM sales")
    with pytest.raises(ReadOnlyViolation):
        store.query("SELECT 1; DROP TABLE sales")


def test_contribution_shares_add_up(ctx_for):
    d = run_tool(ctx_for("I2_01"), "contribution_analysis", {"dimension": "product_category"})["data"]
    assert sum(r["share_of_parent_change"] for r in d["rows"]) == pytest.approx(1.0, abs=1e-3)
    assert d["rows"][0]["value"] == "Laptops"


def test_pvm_decomposition_is_exact(ctx_for):
    for sid in ("I2_01", "I4_03", "I1_01"):
        d = run_tool(ctx_for(sid), "decompose_revenue", {"dimension": "customer_segment"})["data"]
        e = d["effects"]
        assert e["volume"] + e["mix"] + e["price"] == pytest.approx(d["revenue_delta"], rel=1e-6, abs=1.0)


def test_pvm_finds_price_and_mix(ctx_for):
    laptops = next(
        v
        for v in run_tool(ctx_for("I2_01"), "decompose_revenue", {"dimension": "product_category"})["data"]["values"]
        if v["value"] == "Laptops"
    )
    assert laptops["price_pct_change"] == pytest.approx(0.08, abs=0.01)
    mix = run_tool(ctx_for("I4_03"), "decompose_revenue", {"dimension": "customer_segment"})["data"]
    assert mix["effects"]["mix_share"] > 0.5 and abs(mix["units_pct_change"]) < 0.03


def test_benjamini_hochberg_matches_statsmodels():
    p = [0.001, 0.04, 0.03, 0.2, 0.5, 0.012]
    assert benjamini_hochberg(p) == pytest.approx(multipletests(p, method="fdr_bh")[1].tolist())


def test_significance_flags_planted_segment(ctx_for):
    d = run_tool(ctx_for("I1_01"), "significance_test", {"dimension": "channel", "filters": {"country": "US"}})["data"]
    res = {r["value"]: r for r in d["results"]}
    assert res["Online"]["significant"]
    assert not res["Direct Sales"]["significant"]


def test_data_quality_tracking_bug_coarsest_level(ctx_for):
    d = run_tool(ctx_for("I3_05"), "check_data_quality", {})["data"]
    assert d["status"] == "issues_found"
    assert d["flags"][0]["segment"] == {"region": "EMEA", "channel": "Online"}
    assert len(d["flags"]) == 1  # country-level duplicates are suppressed


def test_data_quality_partial_load(ctx_for):
    d = run_tool(ctx_for("I6_01"), "check_data_quality", {})["data"]
    assert d["flags"][0]["type"] == "partial_load"
    assert d["flags"][0]["row_ratio"] == pytest.approx(0.5, abs=0.01)


def test_data_quality_clean_on_business_incident(ctx_for):
    assert run_tool(ctx_for("I2_01"), "check_data_quality", {})["data"]["status"] == "clean"


def test_seasonality_verdicts(ctx_for):
    assert run_tool(ctx_for("I8_07"), "check_seasonality", {})["data"]["verdict"] == "expected_seasonal"
    assert run_tool(ctx_for("I3_01"), "check_seasonality", {})["data"]["verdict"] == "unusual"
