from __future__ import annotations

from pathlib import Path

import pytest

from kpi_detective.data.scenarios import Scenario, load_scenarios, materialize
from kpi_detective.store import SalesStore
from kpi_detective.tools.base import ToolContext

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def dev_scenarios() -> dict[str, Scenario]:
    return {s.id: s for s in load_scenarios(ROOT / "data" / "scenarios" / "dev")}


@pytest.fixture(scope="session")
def store_factory(dev_scenarios):
    """Session cache of read-only stores, one per scenario id."""
    stores: dict[str, SalesStore] = {}

    def get(sid: str) -> SalesStore:
        if sid not in stores:
            stores[sid] = SalesStore.from_frame(materialize(dev_scenarios[sid]))
        return stores[sid]

    yield get
    for s in stores.values():
        s.close()


@pytest.fixture
def ctx_for(store_factory, dev_scenarios):
    def make(sid: str) -> ToolContext:
        return ToolContext(store_factory(sid), dev_scenarios[sid].anomaly_event())

    return make
