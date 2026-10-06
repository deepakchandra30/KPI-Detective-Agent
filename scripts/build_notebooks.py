"""Build and execute the two analysis notebooks (outputs are committed)."""

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
SETUP = """import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})"""


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


eda = [
    md(
        "# 01 · EDA of the synthetic sales data\n\nChecks that the generator produces the realism the PRD asks "
        "for (section 5.1): weekly and yearly seasonality, a growth trend and noisy counts, then shows what the "
        "planted incidents look like next to normal seasonal dips."
    ),
    code(
        SETUP
        + """
from kpi_detective.data.generator import generate_sales, daily_total
df = generate_sales(seed=0)
print(f"{len(df):,} rows, {df['date'].nunique()} days, {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
df.head()"""
    ),
    md("## Revenue mix by dimension"),
    code("""shares = {d: (df.groupby(d)['revenue'].sum() / df['revenue'].sum()).sort_values(ascending=False)
          for d in ['region', 'channel', 'product_category', 'customer_segment']}
pd.concat(shares, names=['dimension', 'value']).to_frame('revenue share').style.format('{:.1%}')"""),
    md("## Daily revenue: trend, yearly events and weekly rhythm"),
    code("""t = daily_total(df)
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(t.index, t / 1e6, lw=0.7, color='#1f2328')
ax.plot(t.index, t.rolling(28, center=True).mean() / 1e6, lw=2, color='#2f6feb', label='28-day mean')
ax.set_ylabel('Revenue (M)'); ax.legend(frameon=False); ax.set_title('Daily revenue, 2023-2025', loc='left')
plt.show()"""),
    code("""seg = df.assign(dow=df['date'].dt.day_name().str[:3]).groupby(['customer_segment', 'dow'])['orders'].sum().unstack(0)
seg = seg.loc[['Mon','Tue','Wed','Thu','Fri','Sat','Sun']]
(seg / seg.mean()).plot(kind='bar', figsize=(8, 3.2), color=['#2f6feb', '#d1242f', '#1a7f37'], rot=0,
                        title='Orders by weekday relative to segment average (weekend lift for Consumer, dip for B2B)')
plt.show()"""),
    code("""from kpi_detective.data.calendar import black_friday
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
for ax, year in zip(axes, [2024, 2025]):
    w = t[f'{year}-11-01':f'{year}-12-31'] if year == 2024 else t[f'{year}-11-01':f'{year}-12-31']
    ax.plot(w.index, w / 1e6, color='#1f2328')
    ax.axvline(pd.Timestamp(black_friday(year)), color='#d1242f', ls='--', lw=1)
    ax.set_title(f'Nov-Dec {year}: holiday season, Black Friday, quarter end, lull', loc='left', fontsize=9)
    ax.tick_params(axis='x', rotation=45)
plt.show()"""),
    md(
        "## Noise level\n\nThe naive alert compares a window with the same weekday over the previous 4 weeks. "
        "In quiet periods its 3-day deviation has a standard deviation of roughly 1.3%, so incidents below about "
        "2-3% of total revenue are genuinely hard to see."
    ),
    code("""def window_dev(t, start, n=3):
    w = [start + pd.Timedelta(days=i) for i in range(n)]
    cur = sum(t[d] for d in w)
    base = sum(np.mean([t[d - pd.Timedelta(days=7 * k)] for k in range(1, 5)]) for d in w)
    return cur / base - 1
quiet = list(pd.date_range('2025-02-12', '2025-03-18')) + list(pd.date_range('2025-05-10', '2025-06-18'))
devs = pd.Series([window_dev(t, s) for s in quiet])
print(f"3-day window deviation in quiet periods: mean {devs.mean():+.2%}, sd {devs.std():.2%}")"""),
    md("## Planted incidents vs a normal seasonal dip"),
    code("""from kpi_detective.data.scenarios import load_scenarios, materialize
scns = {s.id: s for s in load_scenarios(ROOT / 'data/scenarios/dev')}
fig, axes = plt.subplots(2, 4, figsize=(14, 5.5))
for ax, sid in zip(axes.ravel(), ['I1_01', 'I2_02', 'I3_01', 'I4_03', 'I5_04', 'I6_01', 'I7_02', 'I8_02']):
    s = scns[sid]
    clean = daily_total(generate_sales(seed=s.seed)); inc = daily_total(materialize(s))
    lo = pd.Timestamp(s.window_start) - pd.Timedelta(days=21)
    ax.plot(clean[lo:s.window_end].index, clean[lo:s.window_end] / 1e6, color='#8c959f', lw=1, label='clean')
    ax.plot(inc[lo:].index, inc[lo:] / 1e6, color='#d1242f', lw=1.5, label='with incident')
    ax.axvspan(pd.Timestamp(s.window_start), pd.Timestamp(s.window_end), color='#d1242f', alpha=0.08)
    ax.set_title(f"{sid}: {s.description[:38]}", fontsize=8, loc='left'); ax.tick_params(labelsize=7, axis='x', rotation=45)
axes[0, 0].legend(frameon=False, fontsize=7); plt.tight_layout(); plt.show()"""),
    md(
        "I8 (bottom right) is the trap: the naive alert fires because revenue is far below the previous four "
        "weeks, but the same dip happened last year. The forecast detector and `check_seasonality` both know this."
    ),
]

results = [
    md(
        "# 02 · Benchmark results\n\nHeld-out **test** split (sampled parameters, unseen data seeds) unless noted. "
        "Produced by `kpi-detective benchmark --split test` and "
        "`python -m kpi_detective.evaluation.evaluate_detection`."
    ),
    code(
        SETUP
        + """
res = ROOT / 'results'
summary = pd.read_csv(res / 'test' / 'summary.csv')
summary.style.format({c: '{:.1%}' for c in summary.columns if summary[c].dtype == float and c not in
                      ('avg_steps', 'avg_latency_s', 'avg_cost_usd')} | {'avg_steps': '{:.1f}', 'avg_latency_s': '{:.2f}'})"""
    ),
    md(
        "## Dev vs test\n\nThe playbook rules were written while looking at the dev scenarios, so dev accuracy is "
        "optimistic. The gap between the two rows is the honest measure of over-fitting."
    ),
    code("""both = pd.concat({sp: pd.read_csv(res / sp / 'summary.csv').set_index('system') for sp in ['dev', 'test']})
both[['top1_accuracy', 'top3_accuracy', 'top1_segment_only', 'classification_accuracy', 'false_alarm_handling']].style.format('{:.0%}')"""),
    md("## Accuracy by incident type"),
    code("""from IPython.display import Image, display
display(Image(res / 'test' / 'heatmap_top1_by_incident.png', width=650))
display(Image(res / 'test' / 'accuracy_vs_magnitude.png', width=650))"""),
    md("## Detection layer"),
    code("""det = pd.read_csv(res / 'test' / 'detection_summary.csv', index_col=0)
display(det.style.format('{:.0%}'))
display(Image(res / 'test' / 'detection_by_incident_type.png', width=650))"""),
    md("## Failures on the test split\n\nEvery miss of the full playbook, with what it answered instead."),
    code("""rec = pd.read_csv(res / 'test' / 'records.csv')
pb = rec[(rec['system'] == 'playbook') & ~rec['top1']]
import json
desc = {p.stem: json.loads(p.read_text())['description'] for p in (ROOT / 'data/scenarios/test').glob('*.json')}
pb.assign(planted=pb['scenario_id'].map(desc))[['scenario_id', 'planted', 'magnitude', 'classification', 'top_cause', 'top_mechanism']]"""),
    md(
        "Nine misses, three patterns (details in the README):\n\n"
        "1. **Small alerts (5 of 9)**: TI1_05, TI5_06, TI5_07, TI7_01 and TI7_02 all have alerts of 3.1% or less, "
        "against roughly 1.3% noise. Segment drops are real but not significant, or the 12-call budget runs out "
        "before the second cause of an I7 is confirmed.\n"
        "2. **Dominant child ambiguity (1 of 9)**: in TI5_01 the US is about 85% of NA, so a drop in NA monitors "
        "is just as concentrated in US monitors and the playbook stops one level too deep.\n"
        "3. **Seasonal dips that differ from last year (3 of 9)**: TI8_02, TI8_04 and TI8_05. When last year's "
        "dip explains too little of this year's, the segment that carries the seasonal effect (Consumer after "
        "Christmas, Enterprise after quarter end) gets confirmed and reported as a business cause. "
        "Year-over-year comparison per segment is the fix."
    ),
]

for name, cells in (("01_eda_synthetic_data", eda), ("02_results_analysis", results)):
    nb = nbf.v4.new_notebook(
        cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
    )
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(ROOT / "notebooks")}}
    )
    nbf.write(nb, ROOT / "notebooks" / f"{name}.ipynb")
    print("wrote", name)
