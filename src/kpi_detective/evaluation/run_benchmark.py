"""Run the benchmark: every system on every scenario, scored with the frozen rules.

Systems
-------
rule_based          contribution analysis only (no LLM)
playbook            scripted analyst policy in the full graph, with critic (no LLM)
playbook_no_critic  same, critic removed
single_prompt       one LLM call over summary tables           (needs --provider)
agent_no_critic     LLM agent without the verification step     (needs --provider)
agent               full LLM agent                              (needs --provider)

Example::

    python -m kpi_detective.evaluation.run_benchmark --systems rule_based,playbook
    python -m kpi_detective.evaluation.run_benchmark --provider anthropic --systems agent,single_prompt --repeats 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from kpi_detective.agent.critic import verify_report
from kpi_detective.agent.graph import investigate
from kpi_detective.agent.policy import HeuristicPolicy, LLMPolicy
from kpi_detective.baselines import rule_based, single_prompt
from kpi_detective.data.scenarios import Scenario, load_scenarios, materialize
from kpi_detective.evaluation import metrics
from kpi_detective.evaluation.matching import score_report
from kpi_detective.store import SalesStore

OFFLINE = ["rule_based", "playbook", "playbook_no_critic"]
ONLINE = ["single_prompt", "agent_no_critic", "agent"]


def _answer_key(report: dict) -> str:
    top = report["causes"][0] if report.get("causes") else {}
    return json.dumps([report.get("classification"), top.get("segment"), top.get("mechanism")], sort_keys=True)


def run_system(system: str, store: SalesStore, scn: Scenario, client=None) -> dict:
    anomaly = scn.anomaly_event()
    if system == "rule_based":
        return rule_based.run(store, anomaly)
    if system == "single_prompt":
        return single_prompt.run(store, anomaly, client)
    policy = HeuristicPolicy() if system.startswith("playbook") else LLMPolicy(client)
    res = investigate(store, anomaly, policy, use_critic=not system.endswith("no_critic"))
    return {
        "report": res.report,
        "evidence": {e["id"]: e for e in res.state["evidence"]},
        "steps": res.steps,
        "seconds": res.seconds,
        "usage": res.usage,
        "trace": res.trace,
    }


def run_benchmark(
    scenarios: list[Scenario],
    systems: list[str],
    client=None,
    repeats: int = 1,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    records: list[dict] = []
    runs_file = None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        runs_file = (out_dir / "runs.jsonl").open("w")
    for k, scn in enumerate(scenarios, 1):
        with SalesStore.from_frame(materialize(scn)) as store:
            for system in systems:
                reps = repeats if system in ONLINE else 1  # offline systems are deterministic
                for rep in range(reps):
                    out = run_system(system, store, scn, client)
                    report = out["report"]
                    check = verify_report(report, out["evidence"])  # same check for every system
                    score = score_report(report, scn.ground_truth)
                    rec = {
                        "scenario_id": scn.id,
                        "incident_type": scn.incident_type,
                        "difficulty": scn.difficulty,
                        "magnitude": scn.magnitude,
                        "system": system,
                        "repeat": rep,
                        **score,
                        "classification": report.get("classification"),
                        "top_cause": json.dumps(report["causes"][0]["segment"]) if report.get("causes") else None,
                        "top_mechanism": report["causes"][0].get("mechanism") if report.get("causes") else None,
                        "answer_key": _answer_key(report),
                        "steps": out["steps"],
                        "seconds": out["seconds"],
                        "n_numbers": check["n_numbers"],
                        "n_unsupported": check["n_unsupported"],
                        "input_tokens": out["usage"].get("input_tokens", 0),
                        "output_tokens": out["usage"].get("output_tokens", 0),
                        "cost_usd": out["usage"].get("cost_usd"),
                    }
                    records.append(rec)
                    if runs_file:
                        runs_file.write(
                            json.dumps({**rec, "report": report, "trace": out.get("trace", [])}, default=str) + "\n"
                        )
                    if verbose:
                        mark = "ok " if score["top1"] else ("~3 " if score["top3"] else "-- ")
                        print(
                            f"[{k:2d}/{len(scenarios)}] {scn.id:6s} {system:20s} {mark} "
                            f"{report.get('classification'):17s} {rec['top_cause']} {rec['top_mechanism']}"
                        )
    if runs_file:
        runs_file.close()
    return metrics.records_frame(records)


def write_outputs(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    from kpi_detective.evaluation import plots

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "records.csv", index=False)
    summary = metrics.summarize(df)
    summary.to_csv(out_dir / "summary.csv", index=False)
    by_type = metrics.by_incident_type(df)
    if "rule_based" in by_type.columns:  # the baseline cannot name a mechanism: show its lenient score too
        by_type.insert(1, "rule_based (segment only)", metrics.by_incident_type(df, "top1_segment")["rule_based"])
    by_type.to_csv(out_dir / "top1_by_incident_type.csv")
    by_diff = metrics.by_difficulty(df)
    by_diff.to_csv(out_dir / "top1_by_difficulty.csv")
    md = [
        "# Benchmark results",
        "",
        metrics.markdown_table(summary),
        "",
        "## Top-1 accuracy by incident type",
        "",
        by_type.to_markdown(floatfmt=".0%"),
        "",
        "## Top-1 accuracy by incident magnitude (I8 excluded)",
        "",
        by_diff.to_markdown(floatfmt=".0%"),
        "",
    ]
    (out_dir / "results.md").write_text("\n".join(md))
    plots.heatmap(by_type, out_dir / "heatmap_top1_by_incident.png")
    plots.accuracy_vs_magnitude(df, out_dir / "accuracy_vs_magnitude.png")
    if df["cost_usd"].notna().any() and (df["cost_usd"].fillna(0) > 0).any():
        plots.cost_vs_accuracy(summary, out_dir / "cost_vs_accuracy.png")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default="data/scenarios")
    ap.add_argument("--systems", default=",".join(OFFLINE))
    ap.add_argument("--provider", choices=["anthropic", "ollama"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens")
    ap.add_argument("--repeats", type=int, default=1, help="repeat LLM systems to measure consistency")
    ap.add_argument("--only", default=None, help="comma-separated scenario ids or incident types (e.g. I7)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    systems = args.systems.split(",")
    client = None
    if any(s in ONLINE for s in systems):
        if not args.provider:
            ap.error(f"{[s for s in systems if s in ONLINE]} need --provider")
        from kpi_detective.agent.llm import make_client

        kw = {"price_in": args.price_in, "price_out": args.price_out} if args.provider == "anthropic" else {}
        client = make_client(args.provider, args.model, **kw)

    scenarios = load_scenarios(args.scenarios)
    if args.only:
        keys = set(args.only.split(","))
        scenarios = [s for s in scenarios if s.id in keys or s.incident_type in keys]
    t0 = time.time()
    df = run_benchmark(scenarios, systems, client, args.repeats, Path(args.out))
    summary = write_outputs(df, Path(args.out))
    print()
    print(metrics.markdown_table(summary))
    print(f"\n{len(df)} runs in {time.time() - t0:.0f}s. Outputs in {args.out}/")


if __name__ == "__main__":
    main()
