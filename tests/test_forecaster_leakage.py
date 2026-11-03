"""Leakage and baseline checks for the replenishment forecaster.

Run: python3 tests/test_forecaster_leakage.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replenishment_engine.forecaster import ReplenishmentForecaster

SENTINEL = 1e12  # a test-week value no honest training set can contain


def synthetic_panel(days=120):
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    rng = np.random.default_rng(0)
    demand = 500_000 + 100_000 * (dates.dayofweek >= 5) + rng.normal(0, 20_000, days)
    return pd.DataFrame({
        "date": dates,
        "dom": dates.day,
        "dow": dates.dayofweek,
        "month": dates.month,
        "uncensored_demand": demand,
    })


def test_training_never_sees_test_week():
    panel = synthetic_panel()
    fc = ReplenishmentForecaster()
    cutoff_pos = 100
    cutoff = panel["date"].iloc[cutoff_pos]
    # Poison the test week: if any of it leaks into training, predictions explode.
    panel.loc[cutoff_pos + 1: cutoff_pos + 7, "uncensored_demand"] = SENTINEL
    feat = fc.build_terminal_dataset(panel)
    for model_type in ["lgb_calendar_holidays", "pure_exogenous", "naive_last_week"]:
        res = fc.forecast_7day(feat, cutoff, model_type=model_type)
        assert max(res["daily_preds"]) < 1e7, f"{model_type} trained on test-week targets"


def test_naive_is_same_weekday_last_week():
    panel = synthetic_panel()
    fc = ReplenishmentForecaster()
    cutoff = panel["date"].iloc[100]
    res = fc.forecast_7day(fc.build_terminal_dataset(panel), cutoff, model_type="naive_last_week")
    series = panel.set_index("date")["uncensored_demand"]
    expected = [series[pd.Timestamp(d) - pd.Timedelta(days=7)] for d in res["dates"]]
    assert np.allclose(res["daily_preds"], expected), "naive baseline is not seasonal naive"
    assert len(set(res["daily_preds"])) > 1, "naive baseline repeats one value"


if __name__ == "__main__":
    test_training_never_sees_test_week()
    test_naive_is_same_weekday_last_week()
    print("ok")
