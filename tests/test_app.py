from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_app_runs_an_investigation():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    at.selectbox[0].set_value("TI3_02").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("data quality" in m.value for m in at.markdown if m.value.startswith("####"))
