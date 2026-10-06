"""Benchmark metrics (PRD 9.2) from per-run records."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def records_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per system with the headline metrics."""
    rows = []
    for system, g in df.groupby("system", sort=False):
        first = g[g["repeat"] == 0]
        i8 = first[first["incident_type"] == "I8"]
        n_num, n_bad = first["n_numbers"].sum(), first["n_unsupported"].sum()
        cost = first["cost_usd"].dropna()
        rows.append(
            {
                "system": system,
                "scenarios": len(first),
                "top1_accuracy": first["top1"].mean(),
                "top3_accuracy": first["top3"].mean(),
                "top1_segment_only": first["top1_segment"].mean(),
                "classification_accuracy": first["classification_ok"].mean(),
                "false_alarm_handling": i8["classification_ok"].mean() if len(i8) else np.nan,
                "unsupported_claim_rate": n_bad / n_num if n_num else 0.0,
                "avg_steps": first["steps"].mean(),
                "avg_cost_usd": cost.mean() if len(cost) else np.nan,
                "avg_latency_s": first["seconds"].mean(),
                "consistency": consistency(g),
            }
        )
    return pd.DataFrame(rows)


def consistency(g: pd.DataFrame) -> float:
    """Share of scenarios whose (classification, top cause) is identical across repeats."""
    if g["repeat"].nunique() < 2:
        return np.nan
    same = []
    for _, s in g.groupby("scenario_id"):
        same.append(s["answer_key"].nunique() == 1)
    return float(np.mean(same))


def by_incident_type(df: pd.DataFrame, metric: str = "top1") -> pd.DataFrame:
    first = df[df["repeat"] == 0]
    return first.pivot_table(index="incident_type", columns="system", values=metric, aggfunc="mean", sort=False)


def by_difficulty(df: pd.DataFrame, metric: str = "top1") -> pd.DataFrame:
    first = df[(df["repeat"] == 0) & (df["incident_type"] != "I8")]
    order = ["small", "medium", "large"]
    t = first.pivot_table(index="difficulty", columns="system", values=metric, aggfunc="mean", sort=False)
    return t.reindex([o for o in order if o in t.index])


def markdown_table(df: pd.DataFrame) -> str:
    fmt = defaultdict(lambda: "{:.1%}".format)
    fmt.update(
        {
            "scenarios": "{:.0f}".format,
            "avg_steps": "{:.1f}".format,
            "avg_latency_s": "{:.2f}".format,
            "avg_cost_usd": "${:.4f}".format,
            "split": str,
        }
    )
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, str):
                cells.append(v)
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("n/a")
            else:
                cells.append(fmt[c](v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
