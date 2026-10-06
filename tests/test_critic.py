import pytest

from kpi_detective.agent.critic import extract_numbers, resolve, verify_report

EVIDENCE = {
    "E0": {"id": "E0", "data": {"pct_deviation": -0.0734}},
    "E3": {
        "id": "E3",
        "data": {"rows": [{"value": "Laptops", "share_of_total_change": 0.8916, "pct_change": -0.1174}]},
    },
}


def _report(description: str, claim_value: float = 0.8916, share: float = 0.8916) -> dict:
    return {
        "classification": "business_change",
        "headline": "Revenue -7.3% vs expected.",
        "claims": [{"label": "dev", "value": -0.0734, "evidence_id": "E0", "path": "pct_deviation"}],
        "causes": [
            {
                "rank": 1,
                "description": description,
                "segment": {"product_category": "Laptops"},
                "mechanism": "price",
                "share_of_change": share,
                "confidence": "high",
                "evidence_ids": ["E3"],
                "claims": [
                    {
                        "label": "share",
                        "value": claim_value,
                        "evidence_id": "E3",
                        "path": "rows[0].share_of_total_change",
                    },
                    {"label": "pct", "value": -0.1174, "evidence_id": "E3", "path": "rows[0].pct_change"},
                ],
            }
        ],
        "recommended_action": "Review laptop pricing.",
    }


def test_resolve_paths():
    assert resolve(EVIDENCE["E3"]["data"], "rows[0].pct_change") == -0.1174
    with pytest.raises(KeyError):
        resolve(EVIDENCE["E3"]["data"], "rows[3].pct_change")


def test_extract_numbers_skips_dates_and_counts():
    nums = [raw for _, _, raw in extract_numbers("On 2025-03-12, 3 days, Q4: revenue fell 11.7% (4.8M)")]
    assert nums == ["11.7%", "4.8M"]


def test_supported_report_passes():
    r = verify_report(_report("Laptops revenue fell 11.7%, 89% of the drop."), EVIDENCE)
    assert r["verified"] and r["n_unsupported"] == 0 and r["n_numbers"] == 6


def test_rounding_is_allowed_but_invention_is_not():
    assert verify_report(_report("Laptops fell 12% and explain 89.2% of the drop."), EVIDENCE)["verified"]
    bad = verify_report(_report("Laptops fell 11.7%, costing 23% of margin."), EVIDENCE)
    assert not bad["verified"] and bad["n_unsupported"] == 1
    assert any("23%" in i for i in bad["issues"])


def test_wrong_claim_value_and_unbacked_share():
    r = verify_report(_report("Laptops fell 11.7%.", claim_value=0.95, share=0.95), EVIDENCE)
    assert not r["verified"]
    assert r["n_unsupported"] >= 1
