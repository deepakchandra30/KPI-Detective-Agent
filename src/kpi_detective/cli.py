"""Command-line entry point: ``kpi-detective <command>``.

kpi-detective scenarios                    # rebuild data/scenarios/{dev,test}
kpi-detective detect --scenario TI2_03     # run both detectors on a scenario
kpi-detective investigate --scenario TI7_02 [--provider anthropic]
kpi-detective benchmark --split test [--provider anthropic --systems agent,single_prompt]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCENARIO_ROOT = Path("data/scenarios")


def _find_scenario(sid: str):
    from kpi_detective.data.scenarios import Scenario

    for split in ("dev", "test"):
        p = SCENARIO_ROOT / split / f"{sid}.json"
        if p.exists():
            return Scenario.from_dict(json.loads(p.read_text()))
    raise SystemExit(f"scenario {sid} not found under {SCENARIO_ROOT}/{{dev,test}}")


def cmd_scenarios(args) -> None:
    from kpi_detective.data.scenarios import build_benchmark

    for split in ("dev", "test"):
        out = SCENARIO_ROOT / split
        out.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.json"):
            f.unlink()
        sc = build_benchmark(out, split=split)
        print(f"{split}: {len(sc)} scenarios -> {out}")


def cmd_detect(args) -> None:
    from kpi_detective.data.generator import daily_total
    from kpi_detective.data.scenarios import materialize
    from kpi_detective.detection.forecast_detector import detect_forecast
    from kpi_detective.detection.stl_detector import detect_stl

    scn = _find_scenario(args.scenario)
    series = daily_total(materialize(scn))
    for det in (detect_stl, detect_forecast):
        res = det(series)
        recent = [e for e in res.events if e.end.isoformat() >= scn.window_start]
        print(f"{res.method}: {len(res.events)} drop events, {len(recent)} in/after the alert window")
        for e in recent:
            print("   ", json.dumps(e.to_dict()))


def cmd_investigate(args) -> None:
    from kpi_detective.agent.graph import investigate
    from kpi_detective.agent.policy import HeuristicPolicy, LLMPolicy
    from kpi_detective.data.scenarios import materialize
    from kpi_detective.store import SalesStore

    scn = _find_scenario(args.scenario)
    if args.provider:
        from kpi_detective.agent.llm import make_client

        policy = LLMPolicy(make_client(args.provider, args.model))
    else:
        policy = HeuristicPolicy()
    with SalesStore.from_frame(materialize(scn)) as store:
        res = investigate(store, scn.anomaly_event(), policy, use_critic=not args.no_critic)
    print(f"Scenario {scn.id}: {scn.description}")
    print(f"Ground truth: {scn.ground_truth['classification']} {scn.ground_truth['causes']}\n")
    if args.trace:
        for t in res.trace:
            if t["node"] in ("data_quality_check", "seasonality_check", "tool_executor"):
                print(f"  [{t['evidence_id']}] {t['tool']}({json.dumps(t['args'])})  explained={t['explained_share']}")
        print()
    r = res.report
    print(r["headline"])
    for c in r["causes"]:
        print(f"  {c['rank']}. [{c['confidence']}] {c['description']}")
    print(f"Next action: {r['recommended_action']}")
    v = r.get("verification")
    if v:
        print(f"Critic: verified={v['verified']}, unsupported {v['n_unsupported']}/{v['n_numbers']} numbers")
    print(f"{res.steps} tool calls, {res.seconds:.2f}s, usage {res.usage}")


def cmd_benchmark(args) -> None:
    from kpi_detective.evaluation import run_benchmark

    argv = [
        "--scenarios",
        str(SCENARIO_ROOT / args.split),
        "--out",
        f"results/{args.split}",
        "--systems",
        args.systems,
        "--repeats",
        str(args.repeats),
    ]
    if args.provider:
        argv += ["--provider", args.provider]
    if args.model:
        argv += ["--model", args.model]
    run_benchmark.main(argv)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="kpi-detective", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scenarios", help="rebuild benchmark scenario specs").set_defaults(fn=cmd_scenarios)

    p = sub.add_parser("detect", help="run the detectors on a scenario")
    p.add_argument("--scenario", required=True)
    p.set_defaults(fn=cmd_detect)

    p = sub.add_parser("investigate", help="investigate a scenario's alert")
    p.add_argument("--scenario", required=True)
    p.add_argument("--provider", choices=["anthropic", "ollama", "gemini"])
    p.add_argument("--model")
    p.add_argument("--no-critic", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.set_defaults(fn=cmd_investigate)

    p = sub.add_parser("benchmark", help="run the benchmark on a split")
    p.add_argument("--split", choices=["dev", "test"], default="test")
    p.add_argument("--systems", default="rule_based,playbook,playbook_no_critic")
    p.add_argument("--provider", choices=["anthropic", "ollama", "gemini"])
    p.add_argument("--model")
    p.add_argument("--repeats", type=int, default=1)
    p.set_defaults(fn=cmd_benchmark)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
