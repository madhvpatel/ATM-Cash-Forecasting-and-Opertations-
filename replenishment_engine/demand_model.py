"""Holiday-aware demand model shared by the scorecard, the portal and the engine.

One global LightGBM across all 100 cash-file ATMs, forecasting the 7 days after a
cutoff (rows from simulated/maharashtra/run_rolling_origin.build_rows). Training
uses only targets on or before the cutoff.

The cash buffer is calibrated from the backtest, not from in-sample residuals:
the P99 of actual/forecast 7-day totals over past weeks of
outputs/scorecard_100/predictions_weekly.csv (run scripts/run_scorecard_100.py).
"""

from __future__ import annotations

import functools
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd

from replenishment_engine.optimizer import norm_ppf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MASTER = DATA / "master" / "cash_files"
SCORECARD_WEEKLY = ROOT / "outputs" / "scorecard_100" / "predictions_weekly.csv"

MODEL_NAME = "lgb_holiday"
SEED = 22
N_ESTIMATORS = 200
LEARNING_RATE = 0.05
NUM_LEAVES = 63
MIN_CHILD_SAMPLES = 50
CONFORMAL_WINDOW_WEEKS = 26
MIN_PAST_WEEKS = 8
# A forecast week "touches a holiday" if any day is a holiday or within this many days before one.
# Those weeks need a wider buffer (P99 ratio ~1.36 vs ~1.24), see docs/SCORECARD_100_ATMS.md.
HOLIDAY_WEEK_LEAD_DAYS = 3
# ponytail: used only before 8 backtest weeks exist; roughly the naive forecast's P99 (+57-69%).
UNCALIBRATED_MULTIPLIER = 1.6

CALENDAR = [
    "dow", "dom", "month", "is_weekend", "is_2_4_sat", "is_month_end",
    "is_payday", "is_regional_holiday", "pre_holiday", "horizon",
]
CATEGORICAL = ["terminal_id", "loc_type", "city", "rbi_region", "holiday_family", "next_holiday_family"]

# Holiday names in the RBI matrix are combined strings that change year to year
# ("Mahatma Gandhi Jayanti/Dasara/..."), so map them to a family. First match wins:
# the order puts the cash-heavy festivals ahead of the observances they share a day with.
HOLIDAY_FAMILIES = [
    ("diwali", r"diwali|deepavali|naraka|laxmi|lakshmi puja|govardhan|bali pratipada|bhai"),
    ("holi", r"holi|dhuleti|dhulandi|dol jatra|yaosang"),
    ("dussehra", r"dasara|dussehra|dusshera|vijaya|durga puja|maha ?navami|ayudha|dasain"),
    ("ganesh", r"ganesh|vinayaka"),
    ("eid_fitr", r"ramzan|fitr"),
    ("eid_adha", r"bakri|zuha|adha"),
    ("muharram", r"muharram|ashura"),
    ("christmas", r"christmas"),
    ("onam", r"onam|thiruvonam"),
    ("pongal_sankranti", r"pongal|sankranti|makar|magh bihu|uttarayan"),
    ("new_year_spring", r"ugadi|gudi|cheti chand|vishu|bihu|baisakhi|vaisakhi|tamil new year|poila"),
    ("independence", r"independence"),
    ("republic", r"republic"),
    ("gandhi", r"gandhi"),
    ("ambedkar", r"ambedkar"),
    ("good_friday", r"good friday"),
    ("janmashtami", r"janmashtami|krishna"),
    ("ram_navami", r"ram navami|rama navami"),
    ("mahavir", r"mahavir"),
    ("buddha", r"buddha"),
    ("guru_nanak", r"guru nanak|gurpurab|guru purab"),
    ("labour_state_day", r"may day|labour|maharashtra din|state day|formation day"),
    ("mahashivratri", r"shivaratri|shivratri"),
    ("milad", r"milad"),
    ("raksha_bandhan", r"raksha bandhan|rakhi"),
    ("chhath", r"chhath"),
    ("new_year_day", r"new year.s day|losoong|namsoong"),
    ("election", r"election"),
]
HOLIDAY_LOOKAHEAD_DAYS = 7
PRIOR_LIFT_MIN_GAP_DAYS = 300  # last year's occurrence; always before any cutoff for horizon <= 7
HOLIDAY_FEATURES = [
    "holiday_family", "next_holiday_family", "days_to_holiday",
    "is_bank_closed", "closure_run", "days_to_closure", "prior_holiday_lift",
]
MAJOR_FESTIVALS = {"diwali", "holi", "dussehra", "ganesh", "eid_fitr", "eid_adha", "christmas", "onam", "pongal_sankranti"}


def load_rolling_origin():
    path = ROOT / "simulated" / "maharashtra" / "run_rolling_origin.py"
    spec = importlib.util.spec_from_file_location("run_rolling_origin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def holiday_family(name: str) -> str:
    lowered = name.lower()
    for family, pattern in HOLIDAY_FAMILIES:
        if re.search(pattern, lowered):
            return family
    return "other"


def region_calendar(holidays: pd.DataFrame, regions, dates: np.ndarray) -> pd.DataFrame:
    """Per region and date: bank-closure and holiday features, all known from the calendar."""
    families = {
        (r, d): holiday_family(n)
        for r, d, n in holidays[["region", "calendar_date", "holiday_name"]].itertuples(index=False)
    }
    span = pd.date_range(dates[0], pd.Timestamp(dates[-1]) + pd.Timedelta(days=2 * HOLIDAY_LOOKAHEAD_DAYS))
    none_ahead = HOLIDAY_LOOKAHEAD_DAYS + 1
    parts = []
    for region in regions:
        fam = np.array([families.get((region, d), "none") for d in span], dtype=object)
        closed = np.array([
            d.dayofweek == 6 or (d.dayofweek == 5 and (d.day - 1) // 7 + 1 in (2, 4)) or f != "none"
            for d, f in zip(span, fam)
        ])
        n = len(span)
        run_len = np.zeros(n, dtype=int)
        i = 0
        while i < n:
            if closed[i]:
                j = i
                while j < n and closed[j]:
                    j += 1
                run_len[i:j] = j - i
                i = j
            else:
                i += 1
        days_to_closure = np.full(n, none_ahead)
        closure_run = np.where(closed, run_len, 0)
        days_to_holiday = np.full(n, none_ahead)
        next_family = np.full(n, "none", dtype=object)
        for i in range(n):
            if closed[i]:
                days_to_closure[i] = 0
            if fam[i] != "none":
                days_to_holiday[i] = 0
            for k in range(1, HOLIDAY_LOOKAHEAD_DAYS + 1):
                if i + k >= n:
                    break
                if days_to_closure[i] == none_ahead and closed[i + k]:
                    days_to_closure[i] = k
                    closure_run[i] = run_len[i + k]
                if next_family[i] == "none" and fam[i + k] != "none":
                    next_family[i] = fam[i + k]
                    if days_to_holiday[i] == none_ahead:
                        days_to_holiday[i] = k
        parts.append(pd.DataFrame({
            "rbi_region": region, "date": span, "holiday_family": fam, "next_holiday_family": next_family,
            "days_to_holiday": days_to_holiday, "is_bank_closed": closed.astype(int),
            "closure_run": closure_run, "days_to_closure": days_to_closure,
        }))
    return pd.concat(parts, ignore_index=True)


def add_prior_holiday_lift(panel: pd.DataFrame) -> pd.DataFrame:
    """This ATM's lift on the same holiday family at least PRIOR_LIFT_MIN_GAP_DAYS earlier.

    Lift is demand over the 28-day mean ending the day before, so it is causal.
    """
    panel = panel.copy()
    trailing = panel.groupby("terminal_id")["demand"].transform(lambda s: s.shift(1).rolling(28, min_periods=28).mean())
    lift = panel["demand"] / trailing
    panel["prior_holiday_lift"] = np.nan
    events = panel[panel["holiday_family"] != "none"]
    gap = pd.Timedelta(days=PRIOR_LIFT_MIN_GAP_DAYS)
    for _, group in events.groupby(["terminal_id", "holiday_family"]):
        dates = group["date"].to_numpy()
        for idx, date in zip(group.index, dates):
            earlier = group.index[dates <= date - gap]
            if len(earlier):
                panel.loc[idx, "prior_holiday_lift"] = lift[earlier[-1]]
    return panel


def load_panel() -> tuple[pd.DataFrame, np.ndarray]:
    demand = pd.read_csv(DATA / "processed" / "daily_cash_out.csv", parse_dates=["transaction_date"])
    demand = demand.rename(columns={"transaction_date": "date", "net_demand": "demand"})[["terminal_id", "date", "demand"]]
    calendar = pd.read_csv(MASTER / "dim_calendar.csv", parse_dates=["calendar_date"])
    calendar = calendar.rename(columns={"calendar_date": "date"})[["date", "is_payday"]]
    meta = pd.read_csv(MASTER / "terminal_metadata.csv")[["terminal_id", "location"]]
    geo = pd.read_csv(MASTER / "terminal_geography.csv")[["terminal_id", "city", "rbi_region"]]
    holidays = pd.read_csv(MASTER / "holiday_calendar.csv", parse_dates=["calendar_date"])
    holidays = holidays.loc[holidays["holiday_type"] == "bank_holiday", ["region", "calendar_date", "holiday_name"]]
    holiday_set = set(zip(holidays["region"], holidays["calendar_date"]))

    panel = (
        demand.merge(calendar, on="date", how="left", validate="many_to_one")
        .merge(meta, on="terminal_id", how="left", validate="many_to_one")
        .merge(geo, on="terminal_id", how="left", validate="many_to_one")
    )
    if panel[["is_payday", "location", "rbi_region"]].isna().any().any():
        raise AssertionError("a master table does not cover every ATM-day")
    panel["loc_type"] = panel["location"].str.split(" - ", n=1).str[1]
    panel["dom"] = panel["date"].dt.day
    panel["dow"] = panel["date"].dt.dayofweek
    panel["month"] = panel["date"].dt.month
    week_of_month = (panel["dom"] - 1) // 7 + 1
    panel["is_2_4_sat"] = ((panel["dow"] == 5) & week_of_month.isin([2, 4])).astype(int)
    panel["is_regional_holiday"] = [
        int((r, d) in holiday_set) for r, d in zip(panel["rbi_region"], panel["date"])
    ]
    next_day = panel["date"] + pd.Timedelta(days=1)
    tomorrow_holiday = np.array([(r, d) in holiday_set for r, d in zip(panel["rbi_region"], next_day)])
    panel["pre_holiday"] = (tomorrow_holiday & (panel["is_regional_holiday"] == 0)).astype(int)

    panel = panel.sort_values(["terminal_id", "date"]).reset_index(drop=True)
    dates = np.sort(panel["date"].unique())
    if panel.duplicated(["terminal_id", "date"]).any():
        raise AssertionError("duplicate ATM-day")
    if len(panel) != panel["terminal_id"].nunique() * len(dates):
        raise AssertionError("panel is not balanced")
    if not np.all(np.diff(dates.astype("datetime64[D]").astype(np.int64)) == 1):
        raise AssertionError("panel dates are not continuous")
    calendar_by_region = region_calendar(holidays, sorted(panel["rbi_region"].unique()), dates)
    panel = panel.merge(calendar_by_region, on=["rbi_region", "date"], how="left", validate="many_to_one")
    if (panel["is_bank_closed"].isna()).any() or ((panel["holiday_family"] != "none") != (panel["is_regional_holiday"] == 1)).any():
        raise AssertionError("region calendar does not match the holiday flags")
    panel = add_prior_holiday_lift(panel.sort_values(["terminal_id", "date"]).reset_index(drop=True))
    return panel, dates


def add_features(frame: pd.DataFrame, panel: pd.DataFrame, dates: np.ndarray) -> pd.DataFrame:
    terminals = np.sort(panel["terminal_id"].unique())
    matrix = panel.pivot(index="terminal_id", columns="date", values="demand").loc[terminals, dates].to_numpy()
    row = np.searchsorted(terminals, frame["terminal_id"].to_numpy())
    frame = frame.copy()
    # Same-weekday lags are on or before the cutoff for horizons up to 7.
    frame["lag_21"] = matrix[row, frame["target_idx"].to_numpy() - 21]
    frame["seasonal_mean_4w"] = frame[["lag_7", "lag_14", "lag_21", "lag_28"]].mean(axis=1)
    level = frame["roll_mean_28"]
    for col in ["lag_1", "lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_14",
                "roll_std_7", "roll_std_14", "roll_std_28"]:
        frame[f"r_{col}"] = frame[col] / level
    frame["log_level"] = np.log(level)
    attrs = panel.drop_duplicates("terminal_id")[["terminal_id", "city", "rbi_region"]]
    frame = frame.merge(attrs, on="terminal_id", how="left", validate="many_to_one")
    holiday_cols = panel[["terminal_id", "date", *HOLIDAY_FEATURES]].rename(columns={"date": "target_date"})
    frame = frame.merge(holiday_cols, on=["terminal_id", "target_date"], how="left", validate="many_to_one")
    # Volume tier from the first 180 days only, so it is known before any cutoff.
    early = panel[panel["date"] < dates[180]].groupby("terminal_id")["demand"].mean()
    tiers = pd.qcut(early, 3, labels=["low", "mid", "high"])
    frame["volume_tier"] = frame["terminal_id"].map(tiers).astype(str)
    return frame


def design(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    x_train, x_test = train[features].copy(), test[features].copy()
    cats = [c for c in CATEGORICAL if c in features]
    for col in cats:
        dtype = pd.CategoricalDtype(categories=sorted(train[col].unique()))  # unseen -> NaN
        x_train[col] = x_train[col].astype(dtype)
        x_test[col] = x_test[col].astype(dtype)
    return x_train, x_test, cats


def fit_predict(lightgbm, train, test, features, target, weight=None) -> np.ndarray:
    model, x_test = fit_model(lightgbm, train, test, features, target, weight)
    return np.clip(model.predict(x_test), 0, None)


def fit_model(lightgbm, train, test, features, target, weight=None):
    """Trained LGBMRegressor plus the test design matrix it predicts on."""
    x_train, x_test, cats = design(train, test, features)
    model = lightgbm.LGBMRegressor(
        objective="regression_l1",
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        min_child_samples=MIN_CHILD_SAMPLES,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        force_col_wise=True,
    )
    model.fit(x_train, target, sample_weight=weight, categorical_feature=cats)
    return model, x_test


def level_features(ro) -> list[str]:
    return CALENDAR + ro.HISTORY + ["terminal_id", "loc_type"]


def holiday_model_features(ro) -> list[str]:
    return level_features(ro) + HOLIDAY_FEATURES


@functools.lru_cache(maxsize=1)
def model_inputs():
    """Rolling-origin module, LightGBM, panel, dates and the full row frame (built once per process)."""
    ro = load_rolling_origin()
    lightgbm, _ = ro.load_lightgbm()
    panel, dates = load_panel()
    frame = add_features(ro.build_rows(panel, dates), panel, dates)
    return ro, lightgbm, panel, dates, frame


@functools.lru_cache(maxsize=None)
def _fit_week(cutoff: pd.Timestamp):
    """Model trained on everything up to the cutoff, its test matrix, and the test rows (horizons 1-7)."""
    ro, lightgbm, _, dates, frame = model_inputs()
    positions = np.flatnonzero(dates == np.datetime64(cutoff))
    if len(positions) != 1:
        raise ValueError(f"cutoff {cutoff.date()} is not a panel date")
    cutoff_idx = int(positions[0])
    train = frame.loc[frame["target_idx"] <= cutoff_idx]
    test = frame.loc[frame["origin_idx"] == cutoff_idx]
    if train["target_idx"].max() > cutoff_idx or test["target_idx"].min() <= cutoff_idx:
        raise AssertionError("train and test dates overlap")
    model, x_test = fit_model(lightgbm, train, test, holiday_model_features(ro), train["demand"])
    return model, x_test, test


@functools.lru_cache(maxsize=None)
def _forecast_week(cutoff: pd.Timestamp) -> pd.DataFrame:
    model, x_test, test = _fit_week(cutoff)
    pred = np.clip(model.predict(x_test), 0, None)
    out = test[["terminal_id", "target_date", "horizon"]].assign(pred=pred)
    out["holiday_week"] = is_holiday_day(test["days_to_holiday"]).groupby(test["terminal_id"]).transform("any")
    return out.reset_index(drop=True)


def is_holiday_day(days_to_holiday: pd.Series) -> pd.Series:
    """Holiday, or within HOLIDAY_WEEK_LEAD_DAYS before one. Known from the RBI calendar in advance."""
    return days_to_holiday <= HOLIDAY_WEEK_LEAD_DAYS


def forecast_week(cutoff) -> pd.DataFrame:
    """Forecast for every ATM, horizons 1-7 after the cutoff.

    Columns: terminal_id, target_date, horizon, pred, holiday_week (same for all 7 rows of an ATM).
    """
    return _forecast_week(pd.Timestamp(cutoff)).copy()


def feature_importance(cutoff) -> list[dict]:
    """Each feature's share of total split gain in the model behind forecast_week(cutoff), largest first.

    One model serves all 100 ATMs, so this is fleet-wide, not per ATM.
    """
    model, _, _ = _fit_week(pd.Timestamp(cutoff))
    gain = model.booster_.feature_importance(importance_type="gain")
    shares = gain / gain.sum()
    ranked = sorted(zip(model.booster_.feature_name(), shares), key=lambda fs: -fs[1])
    return [{"feature": f, "share": round(float(s), 4)} for f, s in ranked]


@functools.lru_cache(maxsize=1)
def _scorecard_weekly() -> pd.DataFrame:
    if not SCORECARD_WEEKLY.exists():
        raise FileNotFoundError(f"{SCORECARD_WEEKLY} is missing; run python3 scripts/run_scorecard_100.py first")
    weekly = pd.read_csv(SCORECARD_WEEKLY, parse_dates=["origin_date"])
    missing = {MODEL_NAME, "holiday_week"} - set(weekly.columns)
    if missing:
        raise ValueError(f"{SCORECARD_WEEKLY} has no {sorted(missing)} column; re-run the scorecard")
    return weekly


def conformal_multiplier(ratio: pd.Series, past: pd.Series, origin: pd.Series, same_type: pd.Series, q: float):
    """P-q of past actual/forecast ratios among weeks of the same type; pooled if that type is too rare.

    Returns None when even the pooled window has fewer than MIN_PAST_WEEKS origins.
    """
    for rows in (past & same_type, past):
        if origin[rows].nunique() >= MIN_PAST_WEEKS:
            return max(1.0, float(np.nanquantile(ratio[rows], q)))
    return None


def buffer_multiplier(cutoff, holiday_week: bool, service_level: float = 0.99) -> tuple[float, bool]:
    """Load multiplier on the 7-day forecast that met the service level in past backtest weeks.

    Calibrated separately for weeks that touch a holiday and weeks that don't. Uses only weeks whose
    last day is on or before the cutoff. Returns (multiplier, calibrated).
    """
    cutoff = pd.Timestamp(cutoff)
    weekly = _scorecard_weekly()
    week_end = weekly["origin_date"] + pd.Timedelta(days=7)
    past = (week_end <= cutoff) & (weekly["origin_date"] > cutoff - pd.Timedelta(weeks=CONFORMAL_WINDOW_WEEKS))
    ratio = weekly["demand"] / weekly[MODEL_NAME].replace(0, np.nan)
    same_type = weekly["holiday_week"] == bool(holiday_week)
    multiplier = conformal_multiplier(ratio, past, weekly["origin_date"], same_type, service_level)
    if multiplier is None:
        return UNCALIBRATED_MULTIPLIER, False
    return multiplier, True


def cycle_multiplier(buffers: dict, tid) -> float:
    """P99 multiplier for this ATM in one portal cycle: holiday-week or quiet-week calibration."""
    holiday = int(tid) in buffers["holiday_week_atms"]
    return buffers["p99_multiplier_holiday_week" if holiday else "p99_multiplier_quiet_week"]


def equivalent_sigma(pred_sum: float, multiplier: float, service_level: float = 0.99) -> float:
    """Sigma that makes the optimizer's normal buffer equal the calibrated one at this service level.

    ponytail: exact at the calibrated service level only; other levels on the cost frontier
    still assume a normal shape. Calibrate one multiplier per level if the frontier is used for decisions.
    """
    return pred_sum * (multiplier - 1.0) / norm_ppf(service_level)
