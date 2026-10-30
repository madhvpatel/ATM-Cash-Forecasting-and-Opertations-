"""Publish the primary cash-file dataset and its reference tables together."""
from pathlib import Path

import pandas as pd

from .calendar import FIXED_HOLIDAY_NAMES, make_calendar


def write_primary_tables(
    data_dir: str | Path, transactions: pd.DataFrame, terminals: pd.DataFrame
) -> None:
    data_dir = Path(data_dir)
    master = data_dir / "master" / "cash_files"
    processed = data_dir / "processed"
    master.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(processed / "daily_cash_out.csv", index=False)
    terminals.to_csv(master / "terminal_metadata.csv", index=False)
    dates = pd.to_datetime(transactions["transaction_date"])
    calendar = make_calendar(dates.min(), dates.max())
    calendar.to_csv(master / "dim_calendar.csv", index=False)
    holidays = calendar.loc[calendar.is_holiday.eq(1), ["calendar_date"]].copy()
    holidays["holiday_name"] = holidays["calendar_date"].map(
        lambda day: FIXED_HOLIDAY_NAMES[(day.month, day.day)]
    )
    holidays.to_csv(master / "holidays.csv", index=False)
    calendar.loc[calendar.is_payday.eq(1), ["calendar_date", "is_payday"]].to_csv(
        master / "paydays.csv", index=False
    )
