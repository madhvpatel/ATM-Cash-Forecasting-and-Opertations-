"""Week-type buffer calibration: holiday weeks and quiet weeks get their own P99 multiplier.

Run: python3 tests/test_buffer_week_type.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replenishment_engine.demand_model import MIN_PAST_WEEKS, conformal_multiplier


def weeks(n_origins, holiday_every):
    """One row per origin; every `holiday_every`-th origin is a holiday week with a higher ratio."""
    origin = pd.Series(np.arange(n_origins))
    holiday = origin % holiday_every == 0
    ratio = pd.Series(np.where(holiday, 1.5, 1.1))
    return ratio, origin, holiday


def test_each_type_uses_its_own_weeks():
    ratio, origin, holiday = weeks(40, 2)
    past = pd.Series(True, index=origin.index)
    assert conformal_multiplier(ratio, past, origin, holiday, 0.99) == 1.5
    assert conformal_multiplier(ratio, past, origin, ~holiday, 0.99) == 1.1


def test_rare_type_falls_back_to_pooled():
    ratio, origin, holiday = weeks(MIN_PAST_WEEKS * 2, MIN_PAST_WEEKS * 4)  # one holiday week only
    past = pd.Series(True, index=origin.index)
    pooled = float(np.quantile(ratio, 0.99))
    assert conformal_multiplier(ratio, past, origin, holiday, 0.99) == pooled


def test_too_little_history_is_uncalibrated():
    ratio, origin, holiday = weeks(MIN_PAST_WEEKS - 1, 2)
    past = pd.Series(True, index=origin.index)
    assert conformal_multiplier(ratio, past, origin, holiday, 0.99) is None


if __name__ == "__main__":
    test_each_type_uses_its_own_weeks()
    test_rare_type_falls_back_to_pooled()
    test_too_little_history_is_uncalibrated()
    print("ok")
