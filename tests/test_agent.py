import json

import pytest

from kpi_detective.agent.graph import investigate
from kpi_detective.agent.llm import Usage, parse_json
from kpi_detective.agent.policy import HeuristicPolicy, LLMPolicy
from kpi_detective.config import MAX_TOOL_CALLS
from kpi_detective.evaluation.matching import score_report


@pytest.mark.parametrize("sid", ["I1_01", "I2_01", "I3_05", "I4_03", "I5_01", "I6_01", "I8_07"])
def test_playbook_solves_dev_examples(sid, store_factory, dev_scenarios):
    scn = dev_scenarios[sid]
    res = investigate(store_factory(sid), scn.anomaly_event(), HeuristicPolicy())
    assert score_report(res.report, scn.ground_truth)["top1"], res.report
    assert res.report["verification"]["verified"]


def test_forced_first_steps_and_budget(store_factory, dev_scenarios):
    scn = dev_scenarios["I7_01"]
    res = investigate(store_factory("I7_01"), scn.anomaly_event(), HeuristicPolicy())
    tools = [t["tool"] for t in res.trace if "tool" in t]
    assert tools[:2] == ["check_data_quality", "check_seasonality"]
    assert res.steps <= MAX_TOOL_CALLS
    assert all(e["id"] == f"E{i}" for i, e in enumerate(res.state["evidence"]))


def test_data_quality_short_circuits(store_factory, dev_scenarios):
    scn = dev_scenarios["I6_01"]
    res = investigate(store_factory("I6_01"), scn.anomaly_event(), HeuristicPolicy())
    assert res.steps == 2 and res.report["classification"] == "data_quality"


class FakeClient:
    """Scripted LLM: exercises parsing, invalid actions, bad tools and the critic loop."""

    model = "fake"

    def __init__(self, planner_replies, writer_replies):
        self.planner = list(planner_replies)
        self.writer = list(writer_replies)
        self.usage = Usage(price_in=1.0, price_out=5.0)
        self.writer_prompts = []

    def complete(self, system, user, node=""):
        self.usage.calls += 1
        self.usage.input_tokens += 1000
        self.usage.output_tokens += 100
        if node == "planner":
            return self.planner.pop(0) if self.planner else '{"action": "finish", "thought": "done"}'
        self.writer_prompts.append(user)
        return self.writer.pop(0)


def _llm_report(share_text: str) -> str:
    return (
        "```json\n"
        + json.dumps(
            {
                "classification": "business_change",
                "headline": "Revenue fell.",
                "claims": [],
                "causes": [
                    {
                        "rank": 1,
                        "description": f"Laptops revenue fell, {share_text} of the drop (likely driver).",
                        "segment": {"product_category": "Laptops"},
                        "mechanism": "price",
                        "share_of_change": 1.0096,
                        "confidence": "high",
                        "evidence_ids": ["E3"],
                        "claims": [
                            {
                                "label": "share",
                                "value": 1.0096,
                                "evidence_id": "E3",
                                "path": "rows[0].share_of_total_change",
                            }
                        ],
                    }
                ],
                "recommended_action": "Review laptop pricing.",
            }
        )
        + "\n```"
    )


def test_llm_policy_with_critic_revision(store_factory, dev_scenarios):
    scn = dev_scenarios["I2_01"]
    client = FakeClient(
        planner_replies=[
            "not json at all",
            '{"action": "call_tool", "tool": "contribution_analysis", "args": {"dimension": "product_category"}}',
            '{"action": "call_tool", "tool": "contribution_analysis", "args": {"dimension": "galaxy"}}',
            '{"action": "finish", "thought": "Laptops explain the drop."}',
        ],
        writer_replies=[_llm_report("47%"), _llm_report("101%")],  # first draft invents a number
    )
    res = investigate(store_factory("I2_01"), scn.anomaly_event(), LLMPolicy(client))
    errors = [t for t in res.trace if t.get("ok") is False]
    assert len(errors) == 1 and "dimension must be one of" in errors[0]["error"]
    critics = [t for t in res.trace if t["node"] == "critic"]
    assert [c["verified"] for c in critics] == [False, True]
    assert "47%" in client.writer_prompts[1]  # feedback reached the writer
    assert res.report["verification"]["verified"]
    assert res.usage["cost_usd"] == pytest.approx(
        (client.usage.input_tokens * 1 + client.usage.output_tokens * 5) / 1e6
    )


def test_agent_without_critic_keeps_unsupported_claims(store_factory, dev_scenarios):
    scn = dev_scenarios["I2_01"]
    client = FakeClient(
        ['{"action": "call_tool", "tool": "contribution_analysis", "args": {"dimension": "product_category"}}'],
        [_llm_report("47%")],
    )
    res = investigate(store_factory("I2_01"), scn.anomaly_event(), LLMPolicy(client), use_critic=False)
    assert "verification" not in res.report
    assert "47%" in res.report["causes"][0]["description"]


def test_parse_json_variants():
    assert parse_json('Sure! {"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}
    assert parse_json('```json\n{"x": 1}\n```') == {"x": 1}
