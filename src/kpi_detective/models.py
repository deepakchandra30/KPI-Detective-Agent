"""Core data structures passed between layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date


@dataclass(frozen=True)
class AnomalyEvent:
    """Output of the detection layer (FR4) and input to the agent."""

    metric: str
    start: date
    end: date
    expected: float
    actual: float
    method: str = "baseline"
    score: float | None = None
    details: dict = field(default_factory=dict)

    @property
    def pct_deviation(self) -> float:
        return (self.actual - self.expected) / self.expected if self.expected else 0.0

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["end"] = self.end.isoformat()
        d["pct_deviation"] = round(self.pct_deviation, 4)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> AnomalyEvent:
        return cls(
            metric=d["metric"],
            start=date.fromisoformat(d["start"]),
            end=date.fromisoformat(d["end"]),
            expected=float(d["expected"]),
            actual=float(d["actual"]),
            method=d.get("method", "baseline"),
            score=d.get("score"),
            details=d.get("details", {}),
        )
