"""Content identities shared by evaluation, training and artifact publication."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import sklearn


def json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    normalized = frame.reindex(sorted(frame.columns), axis=1).copy()
    for column in normalized:
        if column.endswith("date"):
            normalized[column] = pd.to_datetime(normalized[column], errors="raise").dt.strftime("%Y-%m-%d")
    records = json.loads(normalized.to_json(orient="records", date_format="iso", double_precision=15))
    records.sort(key=lambda row: json.dumps(row, sort_keys=True))
    return json_hash({"columns": list(normalized.columns), "rows": records})


def training_identity(transactions: pd.DataFrame, metadata: pd.DataFrame | None, seed: int) -> dict:
    from .model import GLOBAL_FEATURE_COLUMNS, new_quantile_regressor

    columns = ["terminal_id", "transaction_date", "denomination", "withdrawal_requested_amount",
               "amount_dispensed", "withdrawal_short_amount", "is_recycler", "terminal_uptime_pct"]
    data = transactions.reindex(columns=columns).copy()
    data["is_recycler"] = data["is_recycler"].map(lambda value: 0 if pd.isna(value) else int(value))
    profile = None if metadata is None else metadata.reindex(columns=["terminal_id", "site_type"])
    root = Path(__file__).parent
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in ("model.py", "features.py", "calendar.py", "demand_evaluation.py", "demand_artifact.py", "provenance.py")}
    estimator = new_quantile_regressor(.5, seed)
    recipe = {"target": "withdrawal_requested_amount", "features": GLOBAL_FEATURE_COLUMNS,
              "seed": seed, "backend": type(estimator).__module__ + "." + type(estimator).__name__,
              "parameters": estimator.get_params(), "sklearn": sklearn.__version__, "sources": sources}
    if type(estimator).__module__.startswith("lightgbm"):
        import lightgbm
        recipe["lightgbm"] = lightgbm.__version__
    return {"dataset_sha256": frame_hash(data), "metadata_sha256": frame_hash(profile) if profile is not None else json_hash(None),
            "recipe_sha256": json_hash(recipe), "recipe": recipe}


def evaluation_evidence(transactions, metadata, seed, metrics, folds) -> dict:
    return {"schema_version": 1, **training_identity(transactions, metadata, seed),
            "metrics_sha256": frame_hash(metrics), "evaluation_cadence": "rolling_one_day_ahead",
            "folds": [{"name": f.name, "train_end": str(f.train_end.date()),
                       "test_start": str(f.test_start.date()), "test_end": str(f.test_end.date())} for f in folds]}
