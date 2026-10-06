"""Read-only DuckDB access layer.

The agent never touches the database directly. Tools go through
:class:`SalesStore`, which opens DuckDB in read-only mode and additionally
rejects anything that is not a single SELECT/WITH statement.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from datetime import date
from functools import cached_property
from pathlib import Path

import duckdb
import pandas as pd

from kpi_detective.config import DIMENSIONS, TABLE

_READ_ONLY_SQL = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


class ReadOnlyViolation(RuntimeError):
    """Raised when a non-SELECT statement is attempted."""


class SalesStore:
    def __init__(self, path: str | Path, table: str = TABLE, _tmpdir: str | None = None):
        self.path = str(path)
        self.table = table
        self._tmpdir = _tmpdir
        self.con = duckdb.connect(self.path, read_only=True)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_frame(cls, frame: pd.DataFrame, path: str | Path | None = None, table: str = TABLE) -> SalesStore:
        """Persist a frame to a DuckDB file, then reopen it read-only."""
        tmpdir = None
        if path is None:
            tmpdir = tempfile.mkdtemp(prefix="kpi_detective_")
            path = os.path.join(tmpdir, "sales.duckdb")
        path = str(path)
        if os.path.exists(path):
            os.remove(path)
        with duckdb.connect(path) as rw:
            rw.register("frame_view", frame)
            rw.execute(f"CREATE TABLE {table} AS SELECT * FROM frame_view")
        return cls(path, table, _tmpdir=tmpdir)

    def close(self) -> None:
        self.con.close()
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def __enter__(self) -> SalesStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----------------------------------------------------------------- query
    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        if not _READ_ONLY_SQL.match(sql) or ";" in sql.strip().rstrip(";"):
            raise ReadOnlyViolation("Only single SELECT statements are allowed")
        return self.con.execute(sql, params or []).df()

    @cached_property
    def date_bounds(self) -> tuple[date, date]:
        row = self.query(f"SELECT min(date) AS lo, max(date) AS hi FROM {self.table}").iloc[0]
        return pd.Timestamp(row["lo"]).date(), pd.Timestamp(row["hi"]).date()

    @cached_property
    def dimension_values(self) -> dict[str, list[str]]:
        out = {}
        for dim in DIMENSIONS:
            vals = self.query(f"SELECT DISTINCT {dim} AS v FROM {self.table} ORDER BY 1")["v"]
            out[dim] = [str(v) for v in vals]
        return out

    def daily_total(self, metric: str = "revenue", filters: dict[str, str] | None = None) -> pd.Series:
        where, params = build_where(filters or {})
        df = self.query(
            f"SELECT date, SUM({metric}) AS v FROM {self.table} {where} GROUP BY date ORDER BY date", params
        )
        s = pd.Series(df["v"].to_numpy(dtype=float), index=pd.to_datetime(df["date"]), name=metric)
        return s


def build_where(filters: dict[str, str], extra: list[str] | None = None) -> tuple[str, list]:
    """Parameterised WHERE clause. Dimension names must already be validated."""
    clauses = list(extra or [])
    params: list = []
    for dim, val in filters.items():
        clauses.append(f"{dim} = ?")
        params.append(val)
    return ("WHERE " + " AND ".join(clauses)) if clauses else "", params
