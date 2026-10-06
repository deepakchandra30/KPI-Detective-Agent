"""Result figures for the README (PRD 9.4)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kpi_detective.data.incidents import INCIDENT_TYPES  # noqa: E402

INK, MUTED, GRID = "#1f2328", "#57606a", "#d0d7de"
SERIES = ["#2f6feb", "#d1242f", "#1a7f37", "#9a6700", "#8250df", "#57606a"]


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.title.set_color(INK)


def heatmap(by_type: pd.DataFrame, path: Path) -> None:
    labels = [f"{i} {INCIDENT_TYPES[i].replace('_', ' ')}" for i in by_type.index]
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * by_type.shape[1], 0.5 * len(by_type) + 1.4))
    data = by_type.to_numpy(dtype=float)
    ax.imshow(data, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(by_type.shape[1]), by_type.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9, color="white" if v > 0.6 else INK)
    ax.set_title("Top-1 accuracy by incident type", fontsize=11, loc="left")
    _style(ax)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def accuracy_vs_magnitude(df: pd.DataFrame, path: Path) -> None:
    d = df[(df["repeat"] == 0) & (df["incident_type"] != "I8")].copy()
    bins = [0, 0.02, 0.04, 0.08, 1.0]
    labels = ["<2%", "2-4%", "4-8%", ">8%"]
    d["bucket"] = pd.cut(d["magnitude"], bins=bins, labels=labels, include_lowest=True)
    t = d.pivot_table(index="bucket", columns="system", values="top1", aggfunc="mean", observed=False)
    counts = d[d["system"] == d["system"].iloc[0]].groupby("bucket", observed=False).size()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    styles = ["-", "--", ":", "-."]
    for k, col in enumerate(t.columns):
        ax.plot(
            range(len(t)),
            t[col],
            marker="o",
            lw=2,
            ls=styles[k % len(styles)],
            color=SERIES[k % len(SERIES)],
            label=col,
        )
    ax.set_xticks(range(len(t)), [f"{lab}\n(n={counts.get(lab, 0)})" for lab in t.index])
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Size of the alert (|revenue deviation| vs baseline)", color=MUTED, fontsize=9)
    ax.set_title("Top-1 accuracy vs incident magnitude", fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cost_vs_accuracy(summary: pd.DataFrame, path: Path) -> None:
    s = summary.dropna(subset=["avg_cost_usd"])
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for k, (_, r) in enumerate(s.iterrows()):
        ax.scatter(r["avg_cost_usd"], r["top1_accuracy"], s=60, color=SERIES[k % len(SERIES)])
        ax.annotate(
            r["system"],
            (r["avg_cost_usd"], r["top1_accuracy"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color=INK,
        )
    ax.set_xlabel("Average cost per investigation (USD)", color=MUTED, fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Cost vs top-1 accuracy", fontsize=11, loc="left")
    ax.grid(color=GRID, lw=0.6)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def detection_bars(table: pd.DataFrame, path: Path) -> None:
    """Detection recall by incident type for each detector (I8 = false-positive rate)."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    n = table.shape[1]
    w = 0.8 / n
    x = np.arange(len(table))
    for k, col in enumerate(table.columns):
        ax.bar(x + (k - (n - 1) / 2) * w, table[col], width=w, color=SERIES[k % len(SERIES)], label=col)
    ax.set_xticks(x, table.index)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 1.05)
    ax.set_title("Share of scenarios flagged in the alert window (I8: lower is better)", fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
