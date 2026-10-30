from __future__ import annotations

import pandas as pd

from .calendar import calendar_values


CALENDAR_FEATURES = [
    "day_of_week",
    "month",
    "quarter",
    "is_weekend",
    "is_payday",
    "is_holiday",
    "days_to_month_end",
    "is_month_start",
    "is_month_end",
]

HISTORY_FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
]

FEATURE_COLUMNS = CALENDAR_FEATURES + HISTORY_FEATURES


def build_training_features(transactions: pd.DataFrame) -> pd.DataFrame:
    transactions = transactions.copy()
    if "terminal_id" not in transactions.columns:
        transactions["terminal_id"] = 1001
    dates = pd.to_datetime(transactions["transaction_date"], errors="raise")
    if dates.isna().any():
        raise ValueError("transaction_date cannot be missing")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    transactions["transaction_date"] = dates.dt.normalize()
    demand_column = "net_demand" if "net_demand" in transactions.columns else "amount_dispensed"
    daily = (
        transactions.groupby(
            ["terminal_id", "transaction_date", "denomination"], as_index=False
        )[demand_column]
        .sum()
        .rename(columns={demand_column: "net_demand"})
        .sort_values(["terminal_id", "denomination", "transaction_date"])
    )
    daily["transaction_date"] = pd.to_datetime(daily["transaction_date"])
    if daily.empty:
        raise ValueError("demand history cannot be empty")
    latest = daily["transaction_date"].max()
    for key, series in daily.groupby(["terminal_id", "denomination"]):
        observed = pd.DatetimeIndex(series["transaction_date"])
        expected = pd.date_range(observed.min(), observed.max(), freq="D")
        if not observed.equals(expected):
            raise ValueError(f"Missing calendar days in demand history for {key}")
        if observed.max() != latest:
            raise ValueError(f"Stale series {key}: history must end on {latest.date()}")
    calendar = daily["transaction_date"].apply(calendar_values).apply(pd.Series)
    daily = pd.concat([daily.reset_index(drop=True), calendar.reset_index(drop=True)], axis=1)

    grouped = daily.groupby(["terminal_id", "denomination"], group_keys=False)["net_demand"]
    daily["lag_1"] = grouped.shift(1)
    daily["lag_7"] = grouped.shift(7)
    daily["lag_14"] = grouped.shift(14)
    daily["rolling_mean_7"] = grouped.transform(lambda values: values.shift(1).rolling(7).mean())
    daily["rolling_mean_14"] = grouped.transform(lambda values: values.shift(1).rolling(14).mean())
    daily["rolling_std_7"] = grouped.transform(lambda values: values.shift(1).rolling(7).std(ddof=0))
    return daily.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def future_feature_row(day: pd.Timestamp, history: list[float]) -> dict[str, float]:
    if len(history) < 14:
        raise ValueError("At least 14 days of history are required")
    recent_7 = pd.Series(history[-7:], dtype=float)
    return {
        **calendar_values(day),
        "lag_1": history[-1],
        "lag_7": history[-7],
        "lag_14": history[-14],
        "rolling_mean_7": recent_7.mean(),
        "rolling_mean_14": pd.Series(history[-14:], dtype=float).mean(),
        "rolling_std_7": recent_7.std(ddof=0),
    }
