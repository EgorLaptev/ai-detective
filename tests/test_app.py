from pathlib import Path
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]

def test_app_initial_and_modes():
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
    assert not app.exception
    for mode in ['B','C']:
        app.radio[0].set_value(mode).run()
        assert not app.exception
    app.selectbox[1].set_value('contradiction').run()
    assert not app.exception

def test_app_removal_question():
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
    app.selectbox[0].set_value(app.selectbox[0].options[2]).run()
    app.selectbox[1].set_value('removal').run()
    assert not app.exception
