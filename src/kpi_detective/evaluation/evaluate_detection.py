"""Detection-layer evaluation: does each detector flag the planted incident?

For incidents I1-I7 a detector is right if it flags at least one drop day
inside the alert window (recall). For I8 any flag is a false positive.

    python -m kpi_detective.evaluation.evaluate_detection --scenarios data/scenarios/test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from kpi_detective.data.generator import daily_total
from kpi_detective.data.scenarios import load_scenarios, materialize
from kpi_detective.detection.forecast_detector import detect_forecast
from kpi_detective.detection.stl_detector import detect_stl

NAIVE_THRESHOLD = -0.05  # a typical BI rule: alert when revenue is 5% below the 4-week weekday baseline


def evaluate(scenarios) -> pd.DataFrame:
    rows = []
    for scn in scenarios:
        series = daily_total(materialize(scn))
        ws, we = pd.Timestamp(scn.window_start), pd.Timestamp(scn.window_end)
        row = {
            "scenario_id": scn.id,
            "incident_type": scn.incident_type,
            "magnitude": scn.magnitude,
            "naive_4w_baseline": scn.alert["pct_deviation"] <= NAIVE_THRESHOLD,
        }
        for name, det in (("stl_robust_z", detect_stl), ("forecast_pi95", detect_forecast)):
            f = det(series).frame
            w = f[(f["date"] >= ws) & (f["date"] <= we)]
            row[name] = bool((w["flag"] & (w["actual"] < w["expected"])).any())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["naive_4w_baseline", "stl_robust_z", "forecast_pi95"]
    return df.groupby("incident_type")[cols].mean()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="data/scenarios/test")
    ap.add_argument("--out", default="results/test")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = evaluate(load_scenarios(args.scenarios))
    df.to_csv(out / "detection_records.csv", index=False)
    table = summarize(df)
    table.to_csv(out / "detection_by_incident_type.csv")
    from kpi_detective.evaluation.plots import detection_bars

    detection_bars(table, out / "detection_by_incident_type.png")
    incidents = df[df["incident_type"] != "I8"]
    alarms = df[df["incident_type"] == "I8"]
    cols = ["naive_4w_baseline", "stl_robust_z", "forecast_pi95"]
    overall = pd.DataFrame({"recall_I1_I7": incidents[cols].mean(), "false_positive_rate_I8": alarms[cols].mean()})
    overall.to_csv(out / "detection_summary.csv")
    text = [
        "# Detection results",
        "",
        overall.to_markdown(floatfmt=".0%"),
        "",
        "## Share flagged by incident type",
        "",
        table.to_markdown(floatfmt=".0%"),
        "",
    ]
    (out / "detection.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
