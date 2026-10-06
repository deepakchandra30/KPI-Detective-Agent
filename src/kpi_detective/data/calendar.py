"""Business event calendar.

The same table drives two things:

* the synthetic generator (events shape demand), and
* the ``get_event_calendar`` tool (the agent's lookup of known events),

which mirrors a real company where marketing and finance calendars are
known in advance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class BusinessEvent:
    name: str
    start: date
    end: date
    affected: str
    typical_effect: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["end"] = self.end.isoformat()
        return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th occurrence (1-based) of ``weekday`` (Mon=0) in a month."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def black_friday(year: int) -> date:
    return _nth_weekday(year, 11, 4, 4)


def _quarter_end_weeks(year: int) -> list[tuple[date, date]]:
    ends = [date(year, 3, 31), date(year, 6, 30), date(year, 9, 30), date(year, 12, 31)]
    return [(e - timedelta(days=6), e) for e in ends]


def events_for_year(year: int) -> list[BusinessEvent]:
    bf = black_friday(year)
    ev = [
        BusinessEvent(
            "Post-holiday lull",
            date(year, 1, 1),
            date(year, 1, 12),
            "All segments, strongest for Enterprise",
            "Demand 25-40% below December",
        ),
        BusinessEvent(
            "Back to school",
            date(year, 8, 1),
            date(year, 9, 10),
            "Consumer, mainly Laptops",
            "Consumer laptops +30%",
        ),
        BusinessEvent(
            "Summer enterprise slowdown",
            date(year, 7, 1),
            date(year, 8, 31),
            "Enterprise",
            "Enterprise demand about -10%",
        ),
        BusinessEvent(
            "Holiday season",
            date(year, 11, 15),
            date(year, 12, 23),
            "Consumer",
            "Consumer demand +25%",
        ),
        BusinessEvent(
            "Black Friday to Cyber Monday",
            bf,
            bf + timedelta(days=3),
            "Consumer",
            "Consumer demand +80% on top of holiday season",
        ),
        BusinessEvent(
            "Post-holiday lull",
            date(year, 12, 26),
            date(year, 12, 31),
            "All segments, strongest for Enterprise",
            "Demand 25-40% below December",
        ),
    ]
    for qs, qe in _quarter_end_weeks(year):
        ev.append(
            BusinessEvent(
                f"Quarter end Q{(qe.month - 1) // 3 + 1}",
                qs,
                qe,
                "Enterprise and Small Business",
                "Enterprise +60%, Small Business +15%",
            )
        )
    return sorted(ev, key=lambda e: (e.start, e.name))


def events_between(start: date, end: date) -> list[BusinessEvent]:
    """All events overlapping the closed interval [start, end]."""
    out: list[BusinessEvent] = []
    for year in range(start.year - 1, end.year + 1):
        out.extend(e for e in events_for_year(year) if e.start <= end and e.end >= start)
    return out
