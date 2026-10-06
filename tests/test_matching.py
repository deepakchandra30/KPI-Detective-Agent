from kpi_detective.evaluation.matching import cause_matches, score_report

OUTAGE = {"segment": {"country": "UK", "channel": "Online"}, "mechanism": None}
PRICE = {"segment": {"product_category": "Laptops"}, "mechanism": "price"}


def test_exact_segment_required():
    assert cause_matches({"segment": {"region": "EMEA", "country": "UK", "channel": "Online"}}, OUTAGE)
    assert not cause_matches({"segment": {"country": "UK"}}, OUTAGE)  # too broad
    assert not cause_matches({"segment": {"country": "UK", "channel": "Online", "product_category": "Laptops"}}, OUTAGE)


def test_mechanism_required_when_specified():
    assert cause_matches({"segment": {"product_category": "Laptops"}, "mechanism": "price"}, PRICE)
    assert not cause_matches({"segment": {"product_category": "Laptops"}, "mechanism": "volume"}, PRICE)
    assert cause_matches({"segment": {"product_category": "Laptops"}}, PRICE, check_mechanism=False)


def test_two_causes_need_both_in_top3():
    gt = {"classification": "business_change", "causes": [OUTAGE, PRICE], "require_all": True}
    one = {"classification": "business_change", "causes": [{**PRICE, "rank": 1}]}
    both = {
        "classification": "business_change",
        "causes": [{**PRICE}, {"segment": {"product_category": "Servers"}}, OUTAGE],
    }
    assert score_report(one, gt)["top1"] and not score_report(one, gt)["top3"]
    assert score_report(both, gt)["top3"]


def test_false_alarm_scored_on_classification():
    gt = {"classification": "expected_seasonal", "causes": [], "require_all": False}
    assert score_report({"classification": "expected_seasonal", "causes": []}, gt)["top1"]
    assert not score_report({"classification": "business_change", "causes": [PRICE]}, gt)["top1"]
