import numpy as np
import pandas as pd

from kpi_detective.data.calendar import black_friday, events_between
from kpi_detective.data.generator import cell_frame, daily_total, generate_sales


def test_shape_and_grain():
    df = generate_sales(seed=0)
    days = df["date"].nunique()
    assert len(df) == days * len(cell_frame()) == days * 396
    assert not df.duplicated(["date", "country", "channel", "product_category", "customer_segment"]).any()
    assert (df[["orders", "units", "sessions", "revenue"]] >= 0).all().all()
    assert (df["units"] >= df["orders"]).all()


def test_revenue_is_units_times_price():
    df = generate_sales(seed=0).head(5000)
    assert np.allclose(df["revenue"], df["units"] * df["unit_price"], atol=0.01)


def test_deterministic_by_seed():
    a, b, c = generate_sales(seed=1), generate_sales(seed=1), generate_sales(seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a["orders"].equals(c["orders"])


def test_weekly_seasonality_by_segment():
    df = generate_sales(seed=0)
    df = df[(df["date"] >= "2025-02-15") & (df["date"] < "2025-03-20")]
    df["weekend"] = df["date"].dt.dayofweek >= 5
    per_day = df.groupby(["customer_segment", "weekend", "date"])["orders"].sum().groupby(level=[0, 1]).mean()
    assert per_day["Consumer", True] > per_day["Consumer", False]  # weekend lift
    assert per_day["Enterprise", True] < 0.4 * per_day["Enterprise", False]  # B2B weekend dip


def test_black_friday_spike_and_january_lull():
    df = generate_sales(seed=0)
    t = daily_total(df)
    consumer = daily_total(df[df["customer_segment"] == "Consumer"])
    bf = pd.Timestamp(black_friday(2024))
    assert consumer[bf] > 1.5 * consumer[bf - pd.Timedelta(days=14)]  # Black Friday is a consumer event
    assert t["2025-01-06":"2025-01-10"].mean() < 0.8 * t["2024-12-02":"2024-12-06"].mean()


def test_calendar():
    assert black_friday(2025).isoformat() == "2025-11-28"
    names = {e.name for e in events_between(pd.Timestamp("2025-03-28").date(), pd.Timestamp("2025-04-01").date())}
    assert "Quarter end Q1" in names
