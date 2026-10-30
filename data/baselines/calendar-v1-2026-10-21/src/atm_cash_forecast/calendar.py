from __future__ import annotations

import calendar as month_calendar
from datetime import date

import pandas as pd


FIXED_HOLIDAYS = {(1, 26), (8, 15), (10, 2)}
FIXED_HOLIDAY_NAMES = {
    (1, 26): "Republic Day",
    (8, 15): "Independence Day",
    (10, 2): "Gandhi Jayanti",
}


def _last_working_day(day: date) -> int:
    last_day = month_calendar.monthrange(day.year, day.month)[1]
    candidate = date(day.year, day.month, last_day)
    while candidate.weekday() >= 5:
        candidate = candidate.replace(day=candidate.day - 1)
    return candidate.day


def calendar_values(value: pd.Timestamp | date | str) -> dict[str, int]:
    """Return calendar inputs known before the forecast is generated."""
    day = pd.Timestamp(value)
    is_holiday = int((day.month, day.day) in FIXED_HOLIDAYS)
    is_payday = int(day.day in {1, 15} or day.day == _last_working_day(day.date()))
    return {
        "day_of_week": day.dayofweek,
        "month": day.month,
        "quarter": day.quarter,
        "is_weekend": int(day.dayofweek >= 5),
        "is_payday": is_payday,
        "is_holiday": is_holiday,
        "days_to_month_end": day.days_in_month - day.day,
        "is_month_start": int(day.day <= 3),
        "is_month_end": int(day.day >= day.days_in_month - 2),
    }


def make_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    rows = [{"calendar_date": day.date(), **calendar_values(day)} for day in dates]
    return pd.DataFrame(rows)


def calendar_context(value: pd.Timestamp | date | str) -> dict[str, object]:
    """Return human-readable calendar context for forecast explainability."""
    day = pd.Timestamp(value)
    values = calendar_values(day)
    signals: list[dict[str, str]] = []

    holiday_name = FIXED_HOLIDAY_NAMES.get((day.month, day.day))
    if holiday_name:
        signals.append(
            {
                "type": "holiday",
                "label": holiday_name,
                "meaning": "Public holidays can change ATM footfall and cash demand.",
            }
        )
    if values["is_payday"]:
        signals.append(
            {
                "type": "payday",
                "label": "Payday window",
                "meaning": "The 1st, 15th and last working day are treated as higher-demand salary windows.",
            }
        )
    if values["is_weekend"]:
        signals.append(
            {
                "type": "weekend",
                "label": "Weekend",
                "meaning": "Weekend behaviour is learned separately from working days.",
            }
        )
    if values["is_month_start"]:
        signals.append(
            {
                "type": "month-boundary",
                "label": "Month start",
                "meaning": "The first three days of a month are supplied as a separate model signal.",
            }
        )
    if values["is_month_end"]:
        signals.append(
            {
                "type": "month-boundary",
                "label": "Month end",
                "meaning": "The final three days of a month are supplied as a separate model signal.",
            }
        )
    if not signals:
        signals.append(
            {
                "type": "regular",
                "label": "Regular working day",
                "meaning": "No payday, holiday, weekend or month-boundary flag is active.",
            }
        )

    return {
        "weekday": day.day_name(),
        "date_label": day.strftime("%d %b %Y"),
        "days_to_month_end": values["days_to_month_end"],
        "signals": signals,
    }
