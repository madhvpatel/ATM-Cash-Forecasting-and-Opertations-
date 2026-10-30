from __future__ import annotations

import io
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd


ARCHIVE_PATTERN = "ATM_CashFiles_15Min_XLSX_*.tar.gz"
FOLDER_PATTERN = "ATM_Raw_15Min_XLSX_CashFiles_PART*"
_TIMESTAMP_PATTERN = re.compile(r"KVB_Cashfile_(\d{14})\.xlsx$")
_INTEGER_ATM_ID = re.compile(r"(\d+)$")
_XML_NAMESPACE = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


@dataclass(frozen=True)
class RawCashAudit:
    archives: int
    snapshot_files: int
    snapshot_rows: int
    days: int
    terminals: int
    first_snapshot: str
    last_snapshot: str
    expected_snapshots: int
    missing_snapshots: int
    duplicate_snapshots: int
    zero_end_cash_rows: int
    total_cash_out: int

    def as_dict(self) -> dict[str, int | str]:
        return self.__dict__.copy()


def discover_raw_cash_archives(root: str | Path) -> list[Path]:
    """Return the complete ordered archive set from all raw-data parts."""
    root_path = Path(root)
    # Accept the project root, data directory, or raw directory. Prefer the
    # organized location so old copies cannot accidentally double the dataset.
    for candidate in (root_path / "data" / "raw", root_path / "raw"):
        if candidate.is_dir():
            root_path = candidate
            break
    archives = sorted(
        archive
        for folder in root_path.glob(FOLDER_PATTERN)
        if folder.is_dir()
        for archive in folder.glob(ARCHIVE_PATTERN)
    )
    if not archives:
        raise FileNotFoundError(
            f"No {ARCHIVE_PATTERN} archives found below {root_path}"
        )
    return archives


def _cell_value(cell: ElementTree.Element) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        node = cell.find(f"{_XML_NAMESPACE}is/{_XML_NAMESPACE}t")
    else:
        node = cell.find(f"{_XML_NAMESPACE}v")
    return "" if node is None or node.text is None else node.text


def _parse_cash_report_xml(xml: bytes) -> list[dict[str, object]]:
    """Read the small generated worksheet without an openpyxl dependency."""
    root = ElementTree.fromstring(xml)
    sheet_data = root.find(f"{_XML_NAMESPACE}sheetData")
    if sheet_data is None:
        raise ValueError("Cash Report sheet has no sheetData element")

    parsed: list[dict[str, object]] = []
    for row in list(sheet_data)[1:]:
        cells = {
            re.match(r"[A-Z]+", cell.attrib["r"]).group(): _cell_value(cell)
            for cell in row
        }
        parsed.append(
            {
                "atm_code": cells["B"],
                "city": cells["C"],
                "location": cells["D"],
                "cash_out": int(cells["E"]),
                "end_cash": int(cells["F"]),
            }
        )
    if not parsed:
        raise ValueError("Cash Report sheet contains no ATM rows")
    return parsed


def _read_workbook(workbook: bytes) -> list[dict[str, object]]:
    with zipfile.ZipFile(io.BytesIO(workbook)) as xlsx:
        return _parse_cash_report_xml(xlsx.read("xl/worksheets/sheet1.xml"))


def _terminal_id(atm_code: str) -> int:
    match = _INTEGER_ATM_ID.search(atm_code)
    if match is None:
        raise ValueError(f"ATM code has no numeric identifier: {atm_code}")
    return int(match.group(1))


def load_daily_cash_out(
    root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, RawCashAudit]:
    """Aggregate 15-minute cash-out observations to ATM-day withdrawal demand.

    The experiment follows the project's field-observation assumption that
    ``Cash_Out`` and cash withdrawal demand are operationally interchangeable.
    """
    archives = discover_raw_cash_archives(root)
    daily: dict[tuple[str, pd.Timestamp], int] = {}
    terminal_rows: dict[str, dict[str, object]] = {}
    seen_snapshots: set[pd.Timestamp] = set()
    duplicate_snapshots = 0
    snapshot_files = 0
    snapshot_rows = 0
    zero_end_cash_rows = 0

    for archive in archives:
        with tarfile.open(archive, "r:gz") as monthly:
            for member in monthly:
                if not member.isfile() or not member.name.endswith(".xlsx"):
                    continue
                timestamp_match = _TIMESTAMP_PATTERN.search(member.name)
                if timestamp_match is None:
                    raise ValueError(f"Unexpected workbook name: {member.name}")
                timestamp = pd.to_datetime(
                    timestamp_match.group(1), format="%Y%m%d%H%M%S"
                )
                if timestamp in seen_snapshots:
                    duplicate_snapshots += 1
                seen_snapshots.add(timestamp)

                extracted = monthly.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Could not extract {member.name}")
                rows = _read_workbook(extracted.read())
                snapshot_files += 1
                snapshot_rows += len(rows)
                day = timestamp.normalize()

                for row in rows:
                    atm_code = str(row["atm_code"])
                    terminal_rows.setdefault(
                        atm_code,
                        {
                            "terminal_id": _terminal_id(atm_code),
                            "terminal_code": atm_code,
                            "city": row["city"],
                            "location": row["location"],
                            "site_type": "Cash file ATM",
                        },
                    )
                    daily[(atm_code, day)] = daily.get((atm_code, day), 0) + int(
                        row["cash_out"]
                    )
                    zero_end_cash_rows += int(row["end_cash"] == 0)

    if not seen_snapshots:
        raise ValueError("The archives contain no XLSX snapshots")
    first_snapshot = min(seen_snapshots)
    last_snapshot = max(seen_snapshots)
    expected = pd.date_range(first_snapshot, last_snapshot, freq="15min")
    missing_snapshots = len(expected.difference(pd.DatetimeIndex(seen_snapshots)))

    transactions = pd.DataFrame(
        [
            {
                "terminal_id": _terminal_id(atm_code),
                "transaction_date": day,
                # The aggregate target has no defensible denomination split.
                # ₹100 is the common rounding unit across the source cassettes.
                "denomination": 100,
                "withdrawal_requested_amount": cash_out,
                "amount_dispensed": cash_out,
                "net_demand": cash_out,
                "is_recycler": False,
                "target_semantics": "cash_out_equals_withdrawal_demand",
            }
            for (atm_code, day), cash_out in daily.items()
        ]
    ).sort_values(["terminal_id", "transaction_date"], ignore_index=True)
    terminals = pd.DataFrame(terminal_rows.values()).sort_values(
        "terminal_id", ignore_index=True
    )
    audit = RawCashAudit(
        archives=len(archives),
        snapshot_files=snapshot_files,
        snapshot_rows=snapshot_rows,
        days=transactions["transaction_date"].nunique(),
        terminals=transactions["terminal_id"].nunique(),
        first_snapshot=first_snapshot.isoformat(),
        last_snapshot=last_snapshot.isoformat(),
        expected_snapshots=len(expected),
        missing_snapshots=missing_snapshots,
        duplicate_snapshots=duplicate_snapshots,
        zero_end_cash_rows=zero_end_cash_rows,
        total_cash_out=int(transactions["withdrawal_requested_amount"].sum()),
    )
    return transactions, terminals, audit
