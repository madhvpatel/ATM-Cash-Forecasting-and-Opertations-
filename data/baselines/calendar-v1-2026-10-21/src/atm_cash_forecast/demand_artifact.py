from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from .demand_evaluation import build_true_demand_features
from .features import future_feature_row
from .model import GLOBAL_FEATURE_COLUMNS, QUANTILES, new_global_quantile_model
from .provenance import frame_hash, json_hash, training_identity


ARTIFACT_SCHEMA_VERSION = 2
TARGET_COLUMN = "withdrawal_requested_amount"
PROMOTION_GATES = {
    "maximum_wape": 0.20,
    "maximum_absolute_bias_pct": 0.05,
    "p80_coverage_range": (0.75, 0.85),
    "p95_coverage_range": (0.90, 0.98),
    "minimum_wape_improvement_vs_last_week": 0.01,
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    hashes = pd.util.hash_pandas_object(normalized, index=True).values
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def add_operational_quality_fields(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add explicit censoring flags without inventing unavailable telemetry."""
    required = {
        "withdrawal_requested_amount",
        "amount_dispensed",
        "withdrawal_short_amount",
    }
    missing = sorted(required - set(transactions.columns))
    if missing:
        raise ValueError(f"Missing demand-quality columns: {missing}")
    enriched = transactions.copy()
    requested = pd.to_numeric(enriched["withdrawal_requested_amount"], errors="coerce")
    dispensed = pd.to_numeric(enriched["amount_dispensed"], errors="coerce")
    short = pd.to_numeric(enriched["withdrawal_short_amount"], errors="coerce")
    enriched["is_partial_dispense"] = (dispensed > 0) & (dispensed < requested)
    enriched["is_cashout"] = (requested > 0) & (dispensed == 0)
    enriched["is_censored"] = dispensed < requested
    enriched["has_failed_or_short_withdrawal"] = short > 0
    if "terminal_uptime_pct" not in enriched.columns:
        enriched["terminal_uptime_pct"] = pd.Series(pd.NA, index=enriched.index)
    enriched["availability_telemetry_observed"] = enriched[
        "terminal_uptime_pct"
    ].notna()
    return enriched


def audit_demand_data(transactions: pd.DataFrame) -> dict[str, Any]:
    enriched = add_operational_quality_fields(transactions)
    requested = pd.to_numeric(enriched["withdrawal_requested_amount"], errors="coerce")
    dispensed = pd.to_numeric(enriched["amount_dispensed"], errors="coerce")
    short = pd.to_numeric(enriched["withdrawal_short_amount"], errors="coerce")
    reconciles = requested.eq(dispensed + short)
    required_columns = [
        "terminal_id",
        "transaction_date",
        "denomination",
        "withdrawal_requested_amount",
        "amount_dispensed",
        "withdrawal_short_amount",
    ]
    null_counts = {
        column: int(enriched[column].isna().sum()) for column in required_columns
    }
    errors: list[str] = []
    if not bool(reconciles.all()):
        errors.append("Requested amounts must equal dispensed plus short amounts")
    for column in ("terminal_id", "denomination", "withdrawal_requested_amount", "amount_dispensed", "withdrawal_short_amount"):
        values = pd.to_numeric(enriched[column], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values) | (values < 0) | (values % 1 != 0)
        invalid |= enriched[column].map(lambda value: isinstance(value, (bool, np.bool_)))
        invalid |= ~enriched[column].map(lambda value: isinstance(value, (int, float, np.integer, np.floating)))
        if column in {"terminal_id", "denomination"}:
            invalid |= values <= 0
        if invalid.any():
            errors.append(f"Invalid nonnegative integer values: {column}")
    denomination = pd.to_numeric(enriched["denomination"], errors="coerce").replace(0, np.nan)
    for column in ("withdrawal_requested_amount", "amount_dispensed", "withdrawal_short_amount"):
        if (pd.to_numeric(enriched[column], errors="coerce") % denomination != 0).any():
            errors.append(f"Amount is not a denomination multiple: {column}")
    dates = pd.to_datetime(enriched["transaction_date"], errors="coerce")
    if dates.isna().any() or enriched.empty:
        errors.append("Empty history or invalid transaction dates")
    if enriched.duplicated(["terminal_id", "transaction_date", "denomination"]).any():
        errors.append("Duplicate daily ATM/denomination observations")
    if not errors:
        for key, group in enriched.assign(transaction_date=dates).groupby(["terminal_id", "denomination"]):
            observed = pd.DatetimeIndex(group.transaction_date.sort_values())
            if not observed.equals(pd.date_range(observed.min(), dates.max(), freq="D")):
                errors.append(f"Missing days or stale history: {key}")
    uptime = pd.to_numeric(enriched["terminal_uptime_pct"], errors="coerce")
    observed_uptime = enriched["availability_telemetry_observed"]
    valid_uptime = uptime.notna() & np.isfinite(uptime) & uptime.between(0, 100)
    valid_uptime &= ~enriched["terminal_uptime_pct"].map(lambda value: isinstance(value, (bool, np.bool_)))
    if (observed_uptime & ~valid_uptime).any():
        errors.append("terminal_uptime_pct must be finite and between 0 and 100")
    true_demand_ready = not errors and not any(null_counts.values()) and bool(reconciles.all())
    availability_ready = not enriched.empty and bool(valid_uptime.all())
    return {
        "rows": len(enriched),
        "terminal_count": int(enriched["terminal_id"].nunique()),
        "denomination_count": int(enriched["denomination"].nunique()),
        "date_start": str(dates.min().date()) if dates.notna().any() else None,
        "date_end": str(dates.max().date()) if dates.notna().any() else None,
        "required_null_counts": null_counts,
        "requested_dispensed_short_reconciliation_rate": round(
            float(reconciles.mean()), 6
        ),
        "censored_rows": int(enriched["is_censored"].sum()),
        "partial_dispense_rows": int(enriched["is_partial_dispense"].sum()),
        "cashout_rows": int(enriched["is_cashout"].sum()),
        "availability_telemetry_coverage": round(
            float(enriched["availability_telemetry_observed"].mean()), 6
        ),
        "ready_for_true_demand_training": true_demand_ready,
        "ready_for_live_operational_use": true_demand_ready and availability_ready,
        "validation_errors": errors,
        "readiness_scope": "data eligibility only; not model, security or deployment approval",
    }


def evaluate_promotion(metrics: pd.DataFrame) -> dict[str, Any]:
    overall = metrics.loc[metrics["fold"] == "ALL"].set_index("model_name")
    required_models = {"global_quantile_gbm", "same_weekday_last_week"}
    missing = sorted(required_models - set(overall.index))
    if missing:
        raise ValueError(f"Promotion metrics missing models: {missing}")
    candidate = overall.loc["global_quantile_gbm"]
    baseline = overall.loc["same_weekday_last_week"]
    if not overall.index.is_unique:
        raise ValueError("Duplicate overall model metrics")
    values = [float(candidate[key]) for key in ("wape", "bias_pct", "p80_coverage", "p95_coverage")] + [float(baseline["wape"])]
    if not all(np.isfinite(value) for value in values) or values[0] < 0 or values[-1] < 0:
        raise ValueError("Promotion metrics must be finite with nonnegative WAPE")
    improvement = (values[-1] - values[0]) / values[-1] if values[-1] > 0 else 0.0
    checks = {
        "wape": float(candidate["wape"]) <= PROMOTION_GATES["maximum_wape"],
        "bias": abs(float(candidate["bias_pct"]))
        <= PROMOTION_GATES["maximum_absolute_bias_pct"],
        "p80_coverage": PROMOTION_GATES["p80_coverage_range"][0]
        <= float(candidate["p80_coverage"])
        <= PROMOTION_GATES["p80_coverage_range"][1],
        "p95_coverage": PROMOTION_GATES["p95_coverage_range"][0]
        <= float(candidate["p95_coverage"])
        <= PROMOTION_GATES["p95_coverage_range"][1],
        "baseline_improvement": improvement
        >= PROMOTION_GATES["minimum_wape_improvement_vs_last_week"],
    }
    return {
        "passed": all(checks.values()),
        "status": "offline_approved_for_shadow" if all(checks.values()) else "rejected",
        "checks": checks,
        "gates": PROMOTION_GATES,
        "candidate_metrics": candidate.to_dict(),
        "baseline_metrics": baseline.to_dict(),
        "wape_improvement_vs_last_week": round(improvement, 4),
    }


@dataclass
class GlobalDemandArtifact:
    model_version: str
    created_at: str
    target_column: str
    feature_columns: list[str]
    quantile_models: dict[float, object]
    histories: dict[tuple[int, int], list[float]]
    terminal_profiles: dict[int, dict[str, object]]
    training_start: str
    training_end: str
    dataset_sha256: str
    random_state: int
    metrics: dict[str, Any]
    promotion: dict[str, Any]
    training_identity: dict[str, Any] = field(default_factory=dict)
    evaluation_evidence: dict[str, Any] = field(default_factory=dict)

    def _raw_future_row(
        self,
        terminal_id: int,
        denomination: int,
        day: pd.Timestamp,
        history: list[float],
    ) -> pd.DataFrame:
        profile = self.terminal_profiles[terminal_id]
        row = {
            **future_feature_row(day, history),
            "is_recycler": int(profile["is_recycler"]),
            "terminal_id": terminal_id,
            "denomination": denomination,
            "site_type": str(profile["site_type"]),
        }
        return pd.DataFrame([row])[self.feature_columns]

    def forecast(
        self,
        start_date: str | pd.Timestamp,
        days: int,
        terminal_ids: list[int] | None = None,
    ) -> pd.DataFrame:
        if days < 1:
            raise ValueError("days must be positive")
        requested_terminals = (
            sorted(self.terminal_profiles) if terminal_ids is None else terminal_ids
        )
        unknown = sorted(set(requested_terminals) - set(self.terminal_profiles))
        if unknown:
            raise ValueError(f"Artifact does not know terminal IDs: {unknown}")
        start = pd.Timestamp(start_date).normalize()
        first_supported = pd.Timestamp(self.training_end) + pd.Timedelta(days=1)
        if start < first_supported:
            raise ValueError(
                f"start_date must be on or after {first_supported.date()}"
            )
        final_date = start + pd.Timedelta(days=days - 1)
        rows: list[dict[str, object]] = []

        for terminal_id, denomination in sorted(self.histories):
            if terminal_id not in requested_terminals:
                continue
            history = self.histories[(terminal_id, denomination)].copy()
            for forecast_date in pd.date_range(first_supported, final_date, freq="D"):
                features = self._raw_future_row(
                    terminal_id, denomination, forecast_date, history
                )
                raw = [
                    max(0.0, float(self.quantile_models[q].predict(features)[0]))
                    for q in QUANTILES
                ]
                ordered = np.maximum.accumulate(raw)
                rounded = [
                    int(round(value / denomination) * denomination)
                    for value in ordered
                ]
                history.append(float(rounded[0]))
                if forecast_date < start:
                    continue
                rows.append(
                    {
                        "model_version": self.model_version,
                        "forecast_date": forecast_date.date(),
                        "terminal_id": terminal_id,
                        "denomination": denomination,
                        "p50_amount": rounded[0],
                        "p80_amount": rounded[1],
                        "p95_amount": rounded[2],
                    }
                )
        return pd.DataFrame(rows).sort_values(
            ["terminal_id", "forecast_date", "denomination"]
        ).reset_index(drop=True)

    def save(self, model_root: str | Path) -> tuple[Path, Path]:
        root = Path(model_root)
        root.mkdir(parents=True, exist_ok=True)
        final_dir = root / self.model_version
        if final_dir.exists():
            raise FileExistsError(f"Immutable artifact already exists: {self.model_version}")
        with tempfile.TemporaryDirectory(prefix=".artifact-", dir=root) as staging:
            staged = Path(staging) / "bundle"
            staged.mkdir()
            self._write_bundle(staged)
            staged.rename(final_dir)
        return final_dir / "model.joblib", final_dir / "manifest.json"

    def _write_bundle(self, version_dir: Path) -> None:
        artifact_path = version_dir / "model.joblib"
        manifest_path = version_dir / "manifest.json"
        # Persist plain state rather than the dataclass instance itself. When this
        # module is executed with ``python -m``, pickling the instance would bind
        # its class to ``__main__`` and make it unloadable from the serving process.
        joblib.dump(
            {
                "artifact_type": type(self).__name__,
                "state": self.__dict__,
            },
            artifact_path,
        )
        manifest = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_version": self.model_version,
            "created_at": self.created_at,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "training_start": self.training_start,
            "training_end": self.training_end,
            "dataset_sha256": self.dataset_sha256,
            "artifact_sha256": _file_sha256(artifact_path),
            "artifact_file": artifact_path.name,
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "promotion": self.promotion,
            "metrics": self.metrics,
            "training_identity": self.training_identity,
            "evaluation_evidence": self.evaluation_evidence,
            "trusted_load_only": True,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, version_dir: str | Path) -> "GlobalDemandArtifact":
        directory = Path(version_dir)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["artifact_schema_version"] not in {1, ARTIFACT_SCHEMA_VERSION}:
            raise ValueError("Unsupported artifact schema version")
        if manifest["artifact_file"] != "model.joblib":
            raise ValueError("Invalid artifact filename")
        artifact_path = directory / "model.joblib"
        if _file_sha256(artifact_path) != manifest["artifact_sha256"]:
            raise ValueError("Model artifact checksum mismatch")
        stored = joblib.load(artifact_path)
        if isinstance(stored, cls):
            artifact = stored
        elif (
            isinstance(stored, dict)
            and stored.get("artifact_type") == cls.__name__
            and isinstance(stored.get("state"), dict)
        ):
            artifact = cls(**stored["state"])
        else:
            raise TypeError("Artifact contains an unexpected object type")
        if artifact.model_version != manifest["model_version"]:
            raise ValueError("Model version does not match manifest")
        if artifact.feature_columns != manifest["feature_columns"]:
            raise ValueError("Feature schema does not match manifest")
        if manifest["artifact_schema_version"] == 1:
            artifact.training_identity = {}
            artifact.evaluation_evidence = {}
            artifact.promotion = {**artifact.promotion, "status": "legacy_unverified", "passed": False}
        else:
            for name in ("training_identity", "evaluation_evidence", "metrics", "promotion"):
                if json_hash(getattr(artifact, name)) != json_hash(manifest.get(name)):
                    raise ValueError(f"Artifact {name} does not match manifest")
        return artifact


def train_global_artifact(
    transactions: pd.DataFrame,
    terminal_metadata: pd.DataFrame | None,
    metrics: pd.DataFrame,
    random_state: int = 42,
    *,
    evidence: dict[str, Any] | None = None,
    require_promotion: bool = True,
) -> GlobalDemandArtifact:
    audit = audit_demand_data(transactions)
    if not audit["ready_for_true_demand_training"]:
        raise ValueError(f"Demand data failed quality gates: {audit['validation_errors']}")
    identity = training_identity(transactions, terminal_metadata, random_state)
    evidence = evidence or metrics.attrs.get("evaluation_evidence")
    if not evidence:
        raise ValueError("Bound evaluation evidence is required; rerun the benchmark")
    for name in ("dataset_sha256", "metadata_sha256", "recipe_sha256"):
        if evidence.get(name) != identity[name]:
            raise ValueError(f"Evaluation evidence mismatch: {name}")
    if evidence.get("metrics_sha256") != frame_hash(metrics) or not evidence.get("folds"):
        raise ValueError("Evaluation metrics or folds do not match bound evidence")
    promotion = evaluate_promotion(metrics)
    promotion["evaluation_cadence"] = evidence["evaluation_cadence"]
    promotion["data_quality"] = audit
    if not promotion["passed"] and require_promotion:
        failures = [name for name, passed in promotion["checks"].items() if not passed]
        raise ValueError(f"Candidate failed promotion gates: {failures}")
    features = build_true_demand_features(
        transactions, terminal_metadata=terminal_metadata
    )
    quantile_models: dict[float, object] = {}
    for quantile in QUANTILES:
        model = new_global_quantile_model(quantile, random_state)
        model.fit(features[GLOBAL_FEATURE_COLUMNS], features["net_demand"])
        quantile_models[quantile] = model

    raw = transactions.copy()
    raw["transaction_date"] = pd.to_datetime(raw["transaction_date"])
    daily = (
        raw.groupby(["terminal_id", "transaction_date", "denomination"], as_index=False)[
            TARGET_COLUMN
        ]
        .sum()
        .sort_values(["terminal_id", "denomination", "transaction_date"])
    )
    histories = {
        (int(terminal_id), int(denomination)): group[TARGET_COLUMN]
        .astype(float)
        .tolist()
        for (terminal_id, denomination), group in daily.groupby(
            ["terminal_id", "denomination"]
        )
    }
    profile_rows = features[
        ["terminal_id", "site_type", "is_recycler"]
    ].drop_duplicates("terminal_id")
    terminal_profiles = {
        int(row.terminal_id): {
            "site_type": str(row.site_type),
            "is_recycler": int(row.is_recycler),
        }
        for row in profile_rows.itertuples(index=False)
    }
    dataset_hash = identity["dataset_sha256"]
    training_end = str(daily["transaction_date"].max().date())
    model_version = f"demand-global-{training_end}-{json_hash(identity)[:24]}"
    candidate_metrics = promotion["candidate_metrics"]
    return GlobalDemandArtifact(
        model_version=model_version,
        created_at=datetime.now().astimezone().isoformat(),
        target_column=TARGET_COLUMN,
        feature_columns=GLOBAL_FEATURE_COLUMNS.copy(),
        quantile_models=quantile_models,
        histories=histories,
        terminal_profiles=terminal_profiles,
        training_start=str(daily["transaction_date"].min().date()),
        training_end=training_end,
        dataset_sha256=dataset_hash,
        random_state=random_state,
        metrics=candidate_metrics,
        promotion=promotion,
        training_identity=identity,
        evaluation_evidence=evidence,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train, promote, save, reload, and shadow-score the global demand model"
    )
    parser.add_argument("--input", default="outputs/network_daily_transactions.csv")
    parser.add_argument(
        "--terminal-metadata", default="outputs/postgres_csv/dim_terminal.csv"
    )
    parser.add_argument(
        "--metrics", default="outputs/demand_evaluation/demand_walk_forward_metrics.csv"
    )
    parser.add_argument("--output-dir", default="outputs/demand_model")
    parser.add_argument("--horizon", type=int, default=7)
    args = parser.parse_args()

    transactions = pd.read_csv(args.input, parse_dates=["transaction_date"])
    metadata_path = Path(args.terminal_metadata)
    terminal_metadata = pd.read_csv(metadata_path) if metadata_path.exists() else None
    metrics = pd.read_csv(args.metrics)
    evidence_path = Path(args.metrics).with_name("demand_walk_forward_evidence.json")
    if not evidence_path.exists():
        raise ValueError("Missing evaluation evidence; rerun atm-demand-benchmark")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    enriched = add_operational_quality_fields(transactions)
    audit = audit_demand_data(enriched)
    enriched.to_csv(output_path / "demand_training_dataset.csv", index=False)
    (output_path / "data_quality_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    artifact = train_global_artifact(
        enriched, terminal_metadata=terminal_metadata, metrics=metrics, evidence=evidence
    )
    artifact_path, manifest_path = artifact.save(output_path / "models")
    loaded = GlobalDemandArtifact.load(artifact_path.parent)
    shadow = loaded.forecast(
        pd.Timestamp(loaded.training_end) + pd.Timedelta(days=1), args.horizon
    )
    shadow.to_csv(output_path / "shadow_forecast.csv", index=False)
    result = {
        "model_version": loaded.model_version,
        "promotion_status": loaded.promotion["status"],
        "artifact": str(artifact_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "shadow_forecast": str((output_path / "shadow_forecast.csv").resolve()),
        "shadow_rows": len(shadow),
        "data_quality": audit,
    }
    (output_path / "training_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
