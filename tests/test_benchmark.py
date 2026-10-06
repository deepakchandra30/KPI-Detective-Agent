from kpi_detective.evaluation import metrics
from kpi_detective.evaluation.run_benchmark import run_benchmark


def test_benchmark_smoke(dev_scenarios, tmp_path):
    scns = [dev_scenarios["I3_02"], dev_scenarios["I8_08"]]
    df = run_benchmark(scns, ["rule_based", "playbook"], out_dir=tmp_path, verbose=False)
    assert len(df) == 4 and (tmp_path / "runs.jsonl").exists()
    summary = metrics.summarize(df).set_index("system")
    assert summary.loc["playbook", "top1_accuracy"] == 1.0
    assert summary.loc["rule_based", "classification_accuracy"] == 0.0  # always says business_change
