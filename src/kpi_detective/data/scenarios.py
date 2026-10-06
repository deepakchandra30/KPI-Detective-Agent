"""Benchmark scenarios: 60 incidents with planted, known causes (PRD 9.1).

A scenario is a small JSON spec: base-data seed, incident type and
parameters, the alert window and the ground truth. Data is regenerated
deterministically from the spec, so the repo stores ~60 KB of JSON
instead of ~3 GB of tables.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from kpi_detective.data.generator import generate_sales
from kpi_detective.data.incidents import INCIDENT_TYPES, apply_incident, ground_truth
from kpi_detective.models import AnomalyEvent

N_SEEDS = 3

# Windows chosen so that neither the window nor its 4-week baseline overlaps a
# known calendar event: incidents should be the only thing going on.
QUIET_STARTS = [
    "2025-02-11",
    "2025-02-19",
    "2025-03-04",
    "2025-03-12",
    "2025-05-13",
    "2025-05-21",
    "2025-06-03",
    "2025-06-10",
    "2025-10-29",
    "2025-11-04",
    "2024-05-14",
    "2024-10-30",
    "2024-06-04",
    "2024-02-20",
]

# Normal seasonal dips that a naive "vs last 4 weeks" alert fires on.
FALSE_ALARM_WINDOWS = [
    ("2025-01-02", "2025-01-04"),
    ("2025-01-06", "2025-01-08"),
    ("2025-01-02", "2025-01-09"),
    ("2025-01-07", "2025-01-10"),
    ("2024-12-26", "2024-12-28"),
    ("2025-12-26", "2025-12-29"),
    ("2025-04-01", "2025-04-03"),
    ("2025-07-01", "2025-07-03"),
    ("2025-10-01", "2025-10-03"),
    ("2024-04-01", "2024-04-03"),
    ("2024-10-01", "2024-10-03"),
]


@dataclass
class Scenario:
    id: str
    incident_type: str
    seed: int
    params: dict
    window_start: str
    window_end: str
    ground_truth: dict
    description: str = ""
    alert: dict = field(default_factory=dict)

    @property
    def incident_name(self) -> str:
        return INCIDENT_TYPES[self.incident_type]

    @property
    def data_end(self) -> str:
        """Data always ends on the last alert day: we investigate the latest data."""
        return self.window_end

    @property
    def magnitude(self) -> float:
        return abs(self.alert.get("pct_deviation", 0.0))

    @property
    def difficulty(self) -> str:
        m = self.magnitude
        return "large" if m >= 0.08 else "medium" if m >= 0.04 else "small"

    def anomaly_event(self) -> AnomalyEvent:
        return AnomalyEvent.from_dict(self.alert)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Scenario:
        return cls(**d)

    def save(self, directory: str | Path) -> Path:
        path = Path(directory) / f"{self.id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path


def materialize(scn: Scenario) -> pd.DataFrame:
    """Rebuild the scenario's dataset: clean data, truncated, incident injected."""
    frame = generate_sales(seed=scn.seed)
    frame = frame[frame["date"] <= pd.Timestamp(scn.data_end)].reset_index(drop=True)
    return apply_incident(frame, scn.incident_type, scn.params, seed=scn.seed)


def load_scenarios(directory: str | Path) -> list[Scenario]:
    files = sorted(Path(directory).glob("*.json"))
    return [Scenario.from_dict(json.loads(f.read_text())) for f in files]


def _plus(d: str, days: int) -> str:
    return (date.fromisoformat(d) + timedelta(days=days)).isoformat()


def _specs() -> list[tuple[str, dict, str, str, str]]:
    """(type, params, window_start, window_end, description) for all 60 scenarios."""
    q = QUIET_STARTS
    out: list[tuple[str, dict, str, str, str]] = []

    # I1 regional outage: online orders in one country drop for N days
    i1 = [
        ("US", 0.6, 2),
        ("UK", 0.8, 2),
        ("Germany", 0.8, 2),
        ("US", 0.4, 2),
        ("Japan", 0.9, 3),
        ("Germany", 0.9, 3),
        ("US", 0.8, 1),
    ]
    for k, (country, drop, days) in enumerate(i1):
        s = q[k]
        out.append(
            (
                "I1",
                {"country": country, "channel": "Online", "start": s, "days": days, "drop": drop},
                s,
                _plus(s, days - 1),
                f"Online outage in {country}, -{drop:.0%} for {days}d",
            )
        )

    # I2 price increase: persistent from start, alert on first 5 days
    i2 = [
        ("Laptops", 0.08, -2.5, {}),
        ("Laptops", 0.12, -2.5, {}),
        ("Servers", 0.10, -3.0, {}),
        ("Laptops", 0.08, -3.0, {"region": "EMEA"}),
        ("Monitors", 0.15, -3.0, {}),
        ("Laptops", 0.15, -2.0, {}),
        ("Laptops", 0.10, -2.5, {"region": "NA"}),
    ]
    for k, (cat, inc, el, extra) in enumerate(i2):
        s = q[(k + 3) % len(q)]
        p = {"category": cat, "start": s, "increase": inc, "elasticity": el, "extra_segment": extra}
        where = f" in {extra['region']}" if extra else ""
        out.append(("I2", p, s, _plus(s, 4), f"{cat} price +{inc:.0%}{where}, elasticity {el}"))

    # I3 tracking bug: one channel's revenue recorded as zero or NULL
    i3 = [
        ("Online", 2, "zero", {}),
        ("Retail Partner", 1, "zero", {}),
        ("Direct Sales", 1, "null", {}),
        ("Online", 2, "null", {}),
        ("Online", 2, "zero", {"region": "EMEA"}),
        ("Direct Sales", 3, "zero", {}),
        ("Retail Partner", 2, "null", {"region": "NA"}),
    ]
    for k, (ch, days, mode, extra) in enumerate(i3):
        s = q[(k + 6) % len(q)]
        p = {"channel": ch, "start": s, "days": days, "mode": mode, "extra_segment": extra}
        where = f" in {extra['region']}" if extra else ""
        out.append(("I3", p, s, _plus(s, days - 1), f"{ch}{where} revenue recorded as {mode} for {days}d"))

    # I4 mix shift: Enterprise units move to Consumer, total units stable
    i4 = [(0.25, 5), (0.30, 5), (0.35, 4), (0.20, 7), (0.40, 3), (0.30, 7), (0.45, 5)]
    for k, (shift, days) in enumerate(i4):
        s = q[(k + 9) % len(q)]
        out.append(
            (
                "I4",
                {"start": s, "days": days, "shift": shift},
                s,
                _plus(s, days - 1),
                f"{shift:.0%} of Enterprise units shift to Consumer for {days}d",
            )
        )

    # I5 competitor promotion: gradual 10-day decline, alert on the last 5 days
    i5 = [
        ("NA", "Monitors", 0.8),
        ("EMEA", "Monitors", 0.9),
        ("NA", "Monitors", 0.6),
        ("APAC", "Laptops", 0.6),
        ("EMEA", "Monitors", 0.7),
        ("NA", "Monitors", 0.9),
        ("LATAM", "Laptops", 0.8),
    ]
    for k, (region, cat, drop) in enumerate(i5):
        s = q[(k + 1) % len(q)]
        p = {"region": region, "category": cat, "start": s, "days": 10, "final_drop": drop}
        out.append(("I5", p, _plus(s, 5), _plus(s, 9), f"{cat} in {region} decline gradually to -{drop:.0%} over 10d"))

    # I6 late data load: most recent day(s) partially loaded
    i6 = [[0.5], [0.3], [0.7], [0.6, 0.2], [0.4], [0.2], [0.8]]
    for k, shares in enumerate(i6):
        end = _plus(q[(k + 4) % len(q)], 3)
        start = _plus(end, -(len(shares) - 1))
        desc = " / ".join(f"{s:.0%}" for s in shares)
        out.append(("I6", {"missing_share": shares}, start, end, f"Latest load missing {desc} of rows"))

    # I7 outage + price change at once
    i7 = [
        ("Germany", "Laptops", 0.10),
        ("US", "Laptops", 0.08),
        ("UK", "Servers", 0.10),
        ("Japan", "Laptops", 0.12),
        ("US", "Monitors", 0.15),
        ("France", "Laptops", 0.10),
        ("Germany", "Servers", 0.12),
    ]
    for k, (country, cat, inc) in enumerate(i7):
        s = q[(k + 7) % len(q)]
        p = {
            "outage": {"country": country, "channel": "Online", "start": _plus(s, 1), "days": 2, "drop": 0.7},
            "price": {"category": cat, "start": s, "increase": inc, "elasticity": -2.5, "extra_segment": {}},
        }
        out.append(("I7", p, s, _plus(s, 3), f"Online outage in {country} + {cat} price +{inc:.0%}"))

    # I8 false alarm: no injection, a normal seasonal dip
    for start, end in FALSE_ALARM_WINDOWS:
        out.append(("I8", {}, start, end, "Normal seasonal dip, nothing injected"))
    return out


# --------------------------------------------------------------- held-out split
TEST_SEED_OFFSET = 3
TEST_QUIET_STARTS = [
    "2025-02-25",
    "2025-03-10",
    "2025-05-06",
    "2025-05-27",
    "2025-06-16",
    "2025-10-30",
    "2025-11-06",
    "2024-05-07",
    "2024-05-28",
    "2024-06-11",
    "2024-11-05",
    "2024-03-05",
    "2024-02-27",
    "2025-03-03",
]
TEST_FALSE_ALARM_WINDOWS = [
    ("2025-01-03", "2025-01-06"),
    ("2025-01-08", "2025-01-11"),
    ("2024-12-27", "2024-12-30"),
    ("2025-12-27", "2025-12-30"),
    ("2025-04-02", "2025-04-04"),
    ("2025-07-02", "2025-07-04"),
    ("2025-10-02", "2025-10-04"),
    ("2024-07-01", "2024-07-03"),
    ("2024-04-02", "2024-04-05"),
    ("2024-10-02", "2024-10-04"),
    ("2025-01-05", "2025-01-07"),
]


def _test_specs(seed: int = 2026) -> list[tuple[str, dict, str, str, str]]:
    """Held-out scenarios: parameters *sampled* (not hand-picked), new windows, new data seeds.

    Nothing in the agent or the playbook was tuned on this split.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    q = TEST_QUIET_STARTS
    countries = ["US", "UK", "Germany", "Japan", "France", "Canada", "Australia", "India", "Brazil"]
    regions = ["NA", "EMEA", "APAC", "LATAM"]

    def pick(xs):
        return xs[int(rng.integers(len(xs)))]

    def u(a, b, nd=2):
        return round(float(rng.uniform(a, b)), nd)

    out: list[tuple[str, dict, str, str, str]] = []
    for k in range(7):
        c, drop, days, s = pick(countries), u(0.5, 0.9), int(rng.integers(1, 4)), q[k]
        out.append(
            (
                "I1",
                {"country": c, "channel": "Online", "start": s, "days": days, "drop": drop},
                s,
                _plus(s, days - 1),
                f"Online outage in {c}, -{drop:.0%} for {days}d",
            )
        )
    for k in range(7):
        cat, inc, el, s = pick(["Laptops", "Laptops", "Servers", "Monitors"]), u(0.06, 0.15), u(-3.0, -2.0, 1), q[k + 7]
        extra = {"region": pick(regions[:3])} if rng.random() < 0.3 else {}
        where = f" in {extra['region']}" if extra else ""
        out.append(
            (
                "I2",
                {"category": cat, "start": s, "increase": inc, "elasticity": el, "extra_segment": extra},
                s,
                _plus(s, 4),
                f"{cat} price +{inc:.0%}{where}, elasticity {el}",
            )
        )
    for k in range(7):
        ch, days, mode, s = (
            pick(["Online", "Retail Partner", "Direct Sales"]),
            int(rng.integers(1, 4)),
            pick(["zero", "null"]),
            q[(k + 3) % len(q)],
        )
        extra = {"region": pick(regions[:2])} if rng.random() < 0.3 else {}
        where = f" in {extra['region']}" if extra else ""
        out.append(
            (
                "I3",
                {"channel": ch, "start": s, "days": days, "mode": mode, "extra_segment": extra},
                s,
                _plus(s, days - 1),
                f"{ch}{where} revenue recorded as {mode} for {days}d",
            )
        )
    for k in range(7):
        shift, days, s = u(0.2, 0.45), int(rng.integers(3, 8)), q[(k + 5) % len(q)]
        out.append(
            (
                "I4",
                {"start": s, "days": days, "shift": shift},
                s,
                _plus(s, days - 1),
                f"{shift:.0%} of Enterprise units shift to Consumer for {days}d",
            )
        )
    for k in range(7):
        reg, cat, drop, s = pick(regions), pick(["Monitors", "Monitors", "Laptops"]), u(0.6, 0.9), q[(k + 8) % len(q)]
        out.append(
            (
                "I5",
                {"region": reg, "category": cat, "start": s, "days": 10, "final_drop": drop},
                _plus(s, 5),
                _plus(s, 9),
                f"{cat} in {reg} decline gradually to -{drop:.0%} over 10d",
            )
        )
    for k in range(7):
        shares = [u(0.2, 0.8)] if rng.random() < 0.7 else [u(0.4, 0.8), u(0.1, 0.3)]
        end = _plus(q[(k + 1) % len(q)], 3)
        start = _plus(end, -(len(shares) - 1))
        out.append(
            (
                "I6",
                {"missing_share": shares},
                start,
                end,
                "Latest load missing " + " / ".join(f"{x:.0%}" for x in shares) + " of rows",
            )
        )
    for k in range(7):
        c, cat, inc, s = (
            pick(countries[:6]),
            pick(["Laptops", "Servers", "Monitors"]),
            u(0.08, 0.15),
            q[(k + 4) % len(q)],
        )
        p = {
            "outage": {"country": c, "channel": "Online", "start": _plus(s, 1), "days": 2, "drop": u(0.6, 0.9)},
            "price": {"category": cat, "start": s, "increase": inc, "elasticity": -2.5, "extra_segment": {}},
        }
        out.append(("I7", p, s, _plus(s, 3), f"Online outage in {c} + {cat} price +{inc:.0%}"))
    for start, end in TEST_FALSE_ALARM_WINDOWS:
        out.append(("I8", {}, start, end, "Normal seasonal dip, nothing injected"))
    return out


def build_benchmark(out_dir: str | Path | None = None, split: str = "dev") -> list[Scenario]:
    """Create all scenarios of a split and compute each alert (expected vs actual revenue).

    ``dev``: hand-designed scenarios, used while building the tools and the playbook.
    ``test``: held-out scenarios with sampled parameters and unseen data seeds.
    """
    from kpi_detective.detection.baseline_alert import baseline_alert
    from kpi_detective.store import SalesStore

    scenarios: list[Scenario] = []
    counters: dict[str, int] = {}
    specs = _specs() if split == "dev" else _test_specs()
    prefix = "" if split == "dev" else "T"
    offset = 0 if split == "dev" else TEST_SEED_OFFSET
    for k, (itype, params, ws, we, desc) in enumerate(specs):
        counters[itype] = counters.get(itype, 0) + 1
        scn = Scenario(
            id=f"{prefix}{itype}_{counters[itype]:02d}",
            incident_type=itype,
            seed=offset + k % N_SEEDS,
            params=params,
            window_start=ws,
            window_end=we,
            ground_truth=ground_truth(itype, params),
            description=desc,
        )
        with SalesStore.from_frame(materialize(scn)) as store:
            event = baseline_alert(store, "revenue", date.fromisoformat(ws), date.fromisoformat(we))
        scn.alert = event.to_dict()
        scenarios.append(scn)
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            scn.save(out_dir)
    return scenarios
