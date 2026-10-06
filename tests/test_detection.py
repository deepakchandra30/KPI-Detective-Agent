import pandas as pd

from kpi_detective.data.generator import daily_total
from kpi_detective.data.scenarios import materialize
from kpi_detective.detection.forecast_detector import detect_forecast
from kpi_detective.detection.stl_detector import detect_stl, robust_z


def _in_window(res, scn):
    f = res.frame
    w = f[(f["date"] >= pd.Timestamp(scn.window_start)) & (f["date"] <= pd.Timestamp(scn.window_end))]
    return bool((w["flag"] & (w["actual"] < w["expected"])).any())


def test_robust_z_ignores_outliers():
    import numpy as np

    z = robust_z(np.array([0.0, 0.1, -0.1, 0.05, -0.05, 50.0]))
    assert abs(z[-1]) > 100 and abs(z[0]) < 1


def test_detectors_flag_large_incident(dev_scenarios):
    scn = dev_scenarios["I3_01"]
    series = daily_total(materialize(scn))
    for det in (detect_stl, detect_forecast):
        res = det(series)
        assert _in_window(res, scn)
        ev = res.events[-1]
        assert ev.actual < ev.expected and ev.pct_deviation < -0.2


def test_forecast_detector_knows_the_calendar(dev_scenarios):
    scn = dev_scenarios["I8_07"]  # quarter-end hangover: a normal dip
    assert not _in_window(detect_forecast(daily_total(materialize(scn))), scn)
