"""Mock a switch extract for one ATM.

Standalone script. No other file imports it. Run:

    python3 simulated/maharashtra/mock_switch_log.py

Approved withdrawals are partitioned so their net cash effect equals the
observed daily cash-out of SYNATM0001. Declines, inquiries, and reversals
are added at fixed assumed rates. No real card numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PANEL = HERE / "data" / "atm_daily.csv"
OUT = HERE / "data" / "switch_log_SYNATM0001.csv"
FIELDS = HERE / "data" / "switch_log_SYNATM0001_fields.csv"
NOTES = HERE / "data" / "switch_log_SYNATM0001_notes.json"

ATM_ID = "SYNATM0001"
TERMINAL_ID = 1
# February 2025 mean Cash_Out by hour for this ATM, from hourly snapshots.
HOUR_WEIGHT = np.array(
    [2032, 2318, 2725, 1679, 2696, 6318, 7939, 11139, 13596, 9107, 7379, 6486,
     4632, 7832, 6907, 7371, 10711, 12975, 17896, 15721, 8982, 9246, 4925, 3621],
    dtype=float,
)
LADDER = np.array([500, 1000, 2000, 2500, 3000, 4000, 5000, 8000, 10000])
LADDER_W = np.array([8, 16, 18, 6, 8, 10, 22, 6, 14], dtype=float)
# Chance, per successful withdrawal, of also writing one of these rows.
EXTRA = [
    ("INQ", "310000", "00", "Approved", 0.18, 0),
    ("MINI", "380000", "00", "Approved", 0.06, 0),
    ("CWD", "010000", "51", "Insufficient funds", 0.05, 1),
    ("CWD", "010000", "55", "Incorrect PIN", 0.02, 1),
    ("CWD", "010000", "05", "Do not honour", 0.02, 1),
    ("CWD", "010000", "91", "Issuer inoperative", 0.01, 1),
    ("CWD", "010000", "82", "Unable to dispense", 0.02, 1),
]
REVERSAL_RATE = 0.012


def partition(total: int, rng: np.random.Generator) -> list[int]:
    left = int(total)
    amounts: list[int] = []
    while left > 0:
        eligible = LADDER[LADDER <= left]
        close_out = left <= 10_000 and (left < 500 or not len(eligible) or rng.random() < 0.12)
        if close_out:
            amounts.append(left)
            break
        weights = LADDER_W[LADDER <= left]
        weights = weights / weights.sum()
        pick = int(rng.choice(eligible, p=weights))
        if 0 < left - pick < 500:
            amounts.append(left)
            break
        amounts.append(pick)
        left -= pick
    if sum(amounts) != total:
        raise AssertionError(f"Partition missed {total}: {sum(amounts)}")
    return amounts


def notes_for(amount: int, rng: np.random.Generator) -> tuple[int, int, int]:
    """Split a multiple of 100 into ₹500, ₹200 and ₹100 notes."""
    max_500 = amount // 500
    desired = int(round(0.782 * amount / 500))
    low = max(0, desired - 1)
    high = max_500
    n500 = int(rng.integers(low, high + 1)) if high >= low else 0
    rem = amount - 500 * n500
    n200 = min(rem // 200, int(round(rem * 0.486 / 200)))
    n100 = (rem - 200 * n200) // 100
    if 500 * n500 + 200 * n200 + 100 * n100 != amount:
        raise AssertionError(amount)
    return n500, n200, n100


def stamp(day: pd.Timestamp, rng: np.random.Generator) -> pd.Timestamp:
    hour = int(rng.choice(np.arange(24), p=HOUR_WEIGHT / HOUR_WEIGHT.sum()))
    return day + pd.Timedelta(
        hours=hour,
        minutes=int(rng.integers(0, 60)),
        seconds=int(rng.integers(0, 60)),
    )


def build_cards(rng: np.random.Generator) -> pd.DataFrame:
    n = 3000
    card_type = rng.choice(["debit", "credit", "prepaid"], size=n, p=[0.86, 0.09, 0.05])
    on_us = np.fromiter(
        (rng.random() < {"debit": 0.70, "credit": 0.15, "prepaid": 0.40}[kind] for kind in card_type),
        dtype=bool,
        count=n,
    )
    issuer = np.where(on_us, "ISS-ONUS", rng.choice(["ISS-A", "ISS-B", "ISS-C"], size=n))
    return pd.DataFrame({
        "card_token": [f"TOK{i:06d}" for i in range(1, n + 1)],
        "card_type": card_type,
        "on_us": on_us,
        "account_type": rng.choice(["SAV", "CUR"], size=n, p=[0.82, 0.18]),
        "issuer_id": issuer,
    })


def main() -> None:
    panel = pd.read_csv(PANEL, parse_dates=["date"])
    atm = panel[panel["terminal_id"] == TERMINAL_ID].sort_values("date")
    if atm.empty:
        raise SystemExit(f"{ATM_ID} is not in {PANEL}")
    if int((atm["demand"] % 100).max()) != 0:
        raise SystemExit("Demand is not a multiple of 100")
    location = str(atm["location"].iloc[0])
    rng = np.random.default_rng(20261022)
    cards = build_cards(rng)
    card_w = 1 / np.arange(1, len(cards) + 1) ** 0.7
    card_w = card_w / card_w.sum()
    rows: list[dict] = []

    for day, demand in zip(atm["date"], atm["demand"].astype(int)):
        day_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        seq = 0

        def add(txn_type, proc, code, text, requested, dispensed, when, card, reversal_of=""):
            nonlocal seq
            seq += 1
            effect = int(dispensed)
            n500 = n200 = n100 = 0
            if dispensed:
                n500, n200, n100 = notes_for(abs(int(dispensed)), day_rng)
                if dispensed < 0:
                    n500, n200, n100 = -n500, -n200, -n100
            rows.append({
                "txn_id": f"{ATM_ID}-{day.strftime('%Y%m%d')}-{seq:04d}",
                "txn_ts": when.strftime("%Y-%m-%d %H:%M:%S"),
                "business_date": day.strftime("%Y-%m-%d"),
                "atm_id": ATM_ID,
                "terminal_id": TERMINAL_ID,
                "location": location,
                "mti": "0430" if txn_type == "REV" else "0210",
                "stan": f"{seq:06d}",
                "rrn": f"{day.strftime('%y%m%d')}{seq:06d}",
                "proc_code": proc,
                "txn_type": txn_type,
                "response_code": code,
                "response_text": text,
                "requested_amount": int(requested),
                "dispensed_amount": int(abs(dispensed)) if txn_type == "REV" else int(max(dispensed, 0)),
                "cash_effect": effect,
                "currency": "INR",
                "notes_500": n500,
                "notes_200": n200,
                "notes_100": n100,
                "card_token": card.card_token,
                "card_type": card.card_type,
                "on_us": bool(card.on_us),
                "account_type": card.account_type,
                "acquirer_id": "ACQ-SYN",
                "issuer_id": card.issuer_id,
                "mcc": 6011,
                "reversal_of": reversal_of,
                "source": "mock_switch_extract",
            })

        def card_row():
            return cards.iloc[int(day_rng.choice(len(cards), p=card_w))]

        for amount in partition(int(demand), day_rng):
            add("CWD", "010000", "00", "Approved", amount, amount, stamp(day, day_rng), card_row())
            if day_rng.random() < REVERSAL_RATE:
                extra = int(day_rng.choice(LADDER))
                extra_when = stamp(day, day_rng)
                extra_card = card_row()
                add("CWD", "010000", "00", "Approved", extra, extra, extra_when, extra_card)
                origin = rows[-1]
                add(
                    "REV", "010000", "00", "Reversal", extra, -extra,
                    extra_when + pd.Timedelta(seconds=int(day_rng.integers(20, 90))),
                    extra_card, origin["txn_id"],
                )
                rows[-1]["notes_500"] = -origin["notes_500"]
                rows[-1]["notes_200"] = -origin["notes_200"]
                rows[-1]["notes_100"] = -origin["notes_100"]
            for txn_type, proc, code, text, rate, uses_amount in EXTRA:
                if day_rng.random() >= rate:
                    continue
                req = int(day_rng.choice(LADDER)) if uses_amount else 0
                add(txn_type, proc, code, text, req, 0, stamp(day, day_rng), card_row())

    log = pd.DataFrame(rows).sort_values(["txn_ts", "txn_id"]).reset_index(drop=True)
    daily = log.groupby("business_date", as_index=False)["cash_effect"].sum()
    check = atm.assign(business_date=atm["date"].dt.strftime("%Y-%m-%d"))[["business_date", "demand"]]
    merged = check.merge(daily, on="business_date", how="left")
    gap = int((merged["demand"] - merged["cash_effect"]).abs().max())
    if gap != 0:
        raise SystemExit(f"Reconciliation failed, max daily gap {gap}")
    note_value = 500 * log["notes_500"] + 200 * log["notes_200"] + 100 * log["notes_100"]
    if int((note_value - log["cash_effect"]).abs().max()) != 0:
        raise SystemExit("Note counts do not match cash effect")

    log.to_csv(OUT, index=False)
    pd.DataFrame([
        ("txn_id", "Synthetic id", "assigned"),
        ("txn_ts", "Switch response time", "hour profile from this ATM's February 2025 cash-out"),
        ("business_date", "ATM business date", "observed"),
        ("atm_id", "ATM id", "observed"),
        ("terminal_id", "Numeric id in the panel", "observed"),
        ("location", "Site text", "observed"),
        ("mti", "0210 response, or 0430 reversal", "assigned"),
        ("stan", "Trace number, restarts each day", "assigned"),
        ("rrn", "Retrieval reference", "assigned"),
        ("proc_code", "010000 withdrawal, 310000 balance, 380000 mini statement", "assigned"),
        ("txn_type", "CWD, INQ, MINI, REV", "assigned"),
        ("response_code", "00 approved, 05, 51, 55, 82, 91", "declines are assumed rates"),
        ("response_text", "Meaning of the response code", "assigned"),
        ("requested_amount", "Amount requested", "approved rows partition observed cash-out"),
        ("dispensed_amount", "Amount paid, zero on a decline", "daily total is observed"),
        ("cash_effect", "Signed rupees leaving the ATM", "sums to observed daily cash-out"),
        ("currency", "INR", "assigned"),
        ("notes_500", "Count of ₹500 notes", "split aimed at the February hopper mix"),
        ("notes_200", "Count of ₹200 notes", "split aimed at the February hopper mix"),
        ("notes_100", "Count of ₹100 notes", "split aimed at the February hopper mix"),
        ("card_token", "Stand-in for a masked PAN", "assigned, not a real card"),
        ("card_type", "debit, credit, or prepaid", "assumed mix"),
        ("on_us", "Issuer is the ATM owner", "assumed"),
        ("account_type", "SAV or CUR", "assumed"),
        ("acquirer_id", "Fictional acquirer ACQ-SYN", "assigned"),
        ("issuer_id", "Fictional issuer", "assigned"),
        ("mcc", "6011 ATM", "assigned"),
        ("reversal_of", "txn_id reversed", "assigned"),
        ("source", "mock_switch_extract", "label"),
    ], columns=["field", "meaning", "basis"]).to_csv(FIELDS, index=False)

    approved = log[(log.txn_type == "CWD") & (log.response_code == "00")]
    paid = approved.dispensed_amount.sum()
    NOTES.write_text(json.dumps({
        "atm_id": ATM_ID,
        "location": location,
        "rows": int(len(log)),
        "days": int(atm["date"].nunique()),
        "approved_withdrawals": int(len(approved)),
        "reversals": int((log.txn_type == "REV").sum()),
        "other_rows": int(len(log) - len(approved) - (log.txn_type == "REV").sum()),
        "cash_effect_equals_observed_demand": True,
        "max_daily_gap_inr": gap,
        "note_mix_value_share_of_approved": {
            "500": round(float((500 * approved.notes_500).sum() / paid), 3),
            "200": round(float((200 * approved.notes_200).sum() / paid), 3),
            "100": round(float((100 * approved.notes_100).sum() / paid), 3),
        },
        "not_in_this_file": [
            "Customer age, gender, income",
            "Full PAN",
            "CIT route, vendor, cassette capacity, downtime",
        ],
    }, indent=2))
    print(f"wrote {OUT.name} rows={len(log)} approved={len(approved)} gap={gap}")


if __name__ == "__main__":
    main()
