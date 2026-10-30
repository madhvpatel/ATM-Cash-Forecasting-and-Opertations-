from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, build_training_features
from .model import (
    GLOBAL_FEATURE_COLUMNS,
    QUANTILES,
    QuantileForecaster,
    new_global_quantile_model,
)


# A trained model owns its feature schema. Users select the forecast scope, not
# which inputs the model receives.
RUNTIME_CONFIGURABLE_MODEL_FEATURES: tuple[str, ...] = ()
BENCHMARK_MODELS = (
    "same_weekday_last_week",
    "two_week_same_weekday_average",
    "local_quantile_gbm",
    "global_quantile_gbm",
)


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_true_demand_features(
    transactions: pd.DataFrame,
    target_column: str = "withdrawal_requested_amount",
    terminal_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build leakage-safe features against requested rather than fulfilled demand.

    Requested withdrawals are the preferred target because dispensed cash is
    censored when a terminal is empty or unable to serve the full request.
    """
    if target_column not in transactions.columns:
        raise ValueError(f"Missing true-demand target column: {target_column}")
    prepared = transactions.copy()
    prepared["net_demand"] = prepared[target_column].clip(lower=0)
    features = build_training_features(prepared)
    if "is_recycler" in transactions.columns:
        recycler_profile = (
            transactions.groupby("terminal_id", as_index=False)["is_recycler"].first()
        )
        features = features.merge(recycler_profile, on="terminal_id", how="left")
    else:
        features["is_recycler"] = False
    features["is_recycler"] = features["is_recycler"].fillna(False).astype(int)

    if terminal_metadata is not None and "site_type" in terminal_metadata.columns:
        profile = terminal_metadata[["terminal_id", "site_type"]].drop_duplicates(
            "terminal_id"
        )
        features = features.merge(profile, on="terminal_id", how="left")
    else:
        features["site_type"] = "Unknown"
    features["site_type"] = features["site_type"].fillna("Unknown").astype(str)
    return features


def make_walk_forward_folds(
    features: pd.DataFrame,
    fold_count: int = 3,
    test_days: int = 30,
    minimum_training_days: int = 180,
) -> list[WalkForwardFold]:
    """Create consecutive out-of-time folds ending at the latest observation."""
    if fold_count < 1 or test_days < 1:
        raise ValueError("fold_count and test_days must be positive")
    dates = pd.to_datetime(features["transaction_date"])
    latest = dates.max().normalize()
    earliest_test = latest - pd.Timedelta(days=fold_count * test_days - 1)
    folds: list[WalkForwardFold] = []

    for index in range(fold_count):
        test_start = earliest_test + pd.Timedelta(days=index * test_days)
        test_end = test_start + pd.Timedelta(days=test_days - 1)
        train_dates = dates.loc[dates < test_start]
        if train_dates.empty or train_dates.nunique() < minimum_training_days:
            raise ValueError(
                f"Fold {index + 1} has fewer than {minimum_training_days} training days"
            )
        folds.append(
            WalkForwardFold(
                name=f"fold_{index + 1}",
                train_end=train_dates.max().normalize(),
                test_start=test_start,
                test_end=test_end,
            )
        )
    return folds


def _ordered_rounded_predictions(
    raw_predictions: list[np.ndarray], denominations: pd.Series
) -> np.ndarray:
    ordered = np.maximum.accumulate(np.column_stack(raw_predictions), axis=1)
    units = denominations.astype(float).to_numpy()[:, np.newaxis]
    return np.rint(np.maximum(0.0, ordered) / units).astype(int) * units.astype(int)


def _prediction_frame(
    test_rows: pd.DataFrame,
    model_name: str,
    fold: WalkForwardFold,
    predictions: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": model_name,
            "fold": fold.name,
            "train_end": fold.train_end.date(),
            "forecast_date": test_rows["transaction_date"].dt.date,
            "terminal_id": test_rows["terminal_id"].astype(int).to_numpy(),
            "denomination": test_rows["denomination"].astype(int).to_numpy(),
            "actual_amount": test_rows["net_demand"].astype(int).to_numpy(),
            "p50_amount": predictions[:, 0].astype(int),
            "p80_amount": predictions[:, 1].astype(int),
            "p95_amount": predictions[:, 2].astype(int),
        }
    )


def _seasonal_baseline(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    fold: WalkForwardFold,
    model_name: str,
    base_columns: tuple[str, ...],
) -> pd.DataFrame:
    train_base = train_rows.loc[:, list(base_columns)].mean(axis=1)
    test_base = test_rows.loc[:, list(base_columns)].mean(axis=1)
    calibration = train_rows.loc[:, ["terminal_id", "denomination"]].copy()
    calibration["residual"] = train_rows["net_demand"] - train_base
    residual_quantiles = (
        calibration.groupby(["terminal_id", "denomination"])["residual"]
        .quantile([0.80, 0.95])
        .unstack()
    )

    keys = pd.MultiIndex.from_frame(test_rows[["terminal_id", "denomination"]])
    q80_adjustment = residual_quantiles[0.80].reindex(keys).fillna(0).to_numpy()
    q95_adjustment = residual_quantiles[0.95].reindex(keys).fillna(0).to_numpy()
    predictions = _ordered_rounded_predictions(
        [
            test_base.to_numpy(),
            test_base.to_numpy() + q80_adjustment,
            test_base.to_numpy() + q95_adjustment,
        ],
        test_rows["denomination"],
    )
    return _prediction_frame(test_rows, model_name, fold, predictions)


def _local_model_predictions(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    fold: WalkForwardFold,
    random_state: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    factory = QuantileForecaster(random_state=random_state)
    for (terminal_id, denomination), group_train in train_rows.groupby(
        ["terminal_id", "denomination"]
    ):
        group_test = test_rows.loc[
            (test_rows["terminal_id"] == terminal_id)
            & (test_rows["denomination"] == denomination)
        ]
        if group_test.empty:
            continue
        models = []
        for quantile in QUANTILES:
            model = factory._new_model(quantile)
            model.fit(group_train[FEATURE_COLUMNS], group_train["net_demand"])
            models.append(model)
        predictions = _ordered_rounded_predictions(
            [model.predict(group_test[FEATURE_COLUMNS]) for model in models],
            group_test["denomination"],
        )
        parts.append(
            _prediction_frame(
                group_test, "local_quantile_gbm", fold, predictions
            )
        )
    return pd.concat(parts, ignore_index=True)


def _global_model_predictions(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    fold: WalkForwardFold,
    random_state: int,
) -> pd.DataFrame:
    models = []
    for quantile in QUANTILES:
        model = new_global_quantile_model(quantile, random_state)
        model.fit(train_rows[GLOBAL_FEATURE_COLUMNS], train_rows["net_demand"])
        models.append(model)
    predictions = _ordered_rounded_predictions(
        [model.predict(test_rows[GLOBAL_FEATURE_COLUMNS]) for model in models],
        test_rows["denomination"],
    )
    return _prediction_frame(test_rows, "global_quantile_gbm", fold, predictions)


def _pinball_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    error = actual - predicted
    return float(np.maximum(quantile * error, (quantile - 1) * error).mean())


def score_demand_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Score each model/fold plus an aggregate row for each model."""
    groups: list[tuple[str, str, pd.DataFrame]] = []
    for (model_name, fold), group in predictions.groupby(["model_name", "fold"]):
        groups.append((str(model_name), str(fold), group))
    for model_name, group in predictions.groupby("model_name"):
        groups.append((str(model_name), "ALL", group))

    rows: list[dict[str, object]] = []
    for model_name, fold, group in groups:
        actual = group["actual_amount"].astype(float).to_numpy()
        p50 = group["p50_amount"].astype(float).to_numpy()
        error = p50 - actual
        actual_total = float(np.abs(actual).sum())
        rows.append(
            {
                "model_name": model_name,
                "fold": fold,
                "observations": len(group),
                "mae": round(float(np.abs(error).mean()), 2),
                "wape": round(float(np.abs(error).sum() / actual_total), 4)
                if actual_total
                else 0.0,
                "bias_amount": round(float(error.mean()), 2),
                "bias_pct": round(float(error.sum() / actual_total), 4)
                if actual_total
                else 0.0,
                "p50_coverage": round(float((actual <= p50).mean()), 4),
                "p80_coverage": round(
                    float((actual <= group["p80_amount"].to_numpy()).mean()), 4
                ),
                "p95_coverage": round(
                    float((actual <= group["p95_amount"].to_numpy()).mean()), 4
                ),
                "p50_pinball": round(_pinball_loss(actual, p50, 0.50), 2),
                "p80_pinball": round(
                    _pinball_loss(actual, group["p80_amount"].to_numpy(), 0.80), 2
                ),
                "p95_pinball": round(
                    _pinball_loss(actual, group["p95_amount"].to_numpy(), 0.95), 2
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_name", "fold"]).reset_index(drop=True)


def run_walk_forward_benchmark(
    transactions: pd.DataFrame,
    terminal_metadata: pd.DataFrame | None = None,
    fold_count: int = 3,
    test_days: int = 30,
    minimum_training_days: int = 180,
    random_state: int = 42,
    models: tuple[str, ...] = BENCHMARK_MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark seasonal baselines, local GBMs and one pooled global GBM."""
    unknown = set(models) - set(BENCHMARK_MODELS)
    if unknown:
        raise ValueError(f"Unknown benchmark models: {sorted(unknown)}")
    features = build_true_demand_features(
        transactions, terminal_metadata=terminal_metadata
    )
    folds = make_walk_forward_folds(
        features,
        fold_count=fold_count,
        test_days=test_days,
        minimum_training_days=minimum_training_days,
    )
    parts: list[pd.DataFrame] = []

    for fold in folds:
        train_rows = features.loc[
            features["transaction_date"] <= fold.train_end
        ].copy()
        test_rows = features.loc[
            features["transaction_date"].between(fold.test_start, fold.test_end)
        ].copy()
        if test_rows.empty:
            raise ValueError(f"No test observations for {fold.name}")

        if "same_weekday_last_week" in models:
            parts.append(
                _seasonal_baseline(
                    train_rows,
                    test_rows,
                    fold,
                    "same_weekday_last_week",
                    ("lag_7",),
                )
            )
        if "two_week_same_weekday_average" in models:
            parts.append(
                _seasonal_baseline(
                    train_rows,
                    test_rows,
                    fold,
                    "two_week_same_weekday_average",
                    ("lag_7", "lag_14"),
                )
            )
        if "local_quantile_gbm" in models:
            parts.append(
                _local_model_predictions(train_rows, test_rows, fold, random_state)
            )
        if "global_quantile_gbm" in models:
            parts.append(
                _global_model_predictions(train_rows, test_rows, fold, random_state)
            )

    predictions = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["model_name", "fold", "terminal_id", "forecast_date", "denomination"])
        .reset_index(drop=True)
    )
    metrics = score_demand_predictions(predictions)
    from .provenance import evaluation_evidence
    metrics.attrs["evaluation_evidence"] = evaluation_evidence(
        transactions, terminal_metadata, random_state, metrics, folds
    )
    return predictions, metrics


def write_benchmark_outputs(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, object]:
    evidence = metrics.attrs.get("evaluation_evidence")
    if evidence is None:
        raise ValueError("Benchmark metrics lack bound evaluation evidence")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path / "demand_walk_forward_predictions.csv", index=False)
    metrics.to_csv(output_path / "demand_walk_forward_metrics.csv", index=False)
    (output_path / "demand_walk_forward_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    overall = metrics.loc[metrics["fold"] == "ALL"].sort_values("wape")
    winner = overall.iloc[0]
    summary = {
        "target": "true requested withdrawal amount",
        "evaluation_cadence": "rolling one-day-ahead forecasts",
        "runtime_configurable_model_features": len(
            RUNTIME_CONFIGURABLE_MODEL_FEATURES
        ),
        "feature_schema": GLOBAL_FEATURE_COLUMNS,
        "winner": str(winner["model_name"]),
        "winner_wape": float(winner["wape"]),
        "winner_bias_pct": float(winner["bias_pct"]),
        "models": overall.to_dict(orient="records"),
    }
    (output_path / "demand_walk_forward_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the demand-only walk-forward benchmark")
    parser.add_argument(
        "--input",
        default="outputs/network_daily_transactions.csv",
        help="Daily transaction CSV produced by the POC",
    )
    parser.add_argument("--output-dir", default="outputs/demand_evaluation")
    parser.add_argument(
        "--terminal-metadata",
        default="outputs/postgres_csv/dim_terminal.csv",
        help="Optional terminal metadata CSV containing site_type",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=30)
    args = parser.parse_args()

    transactions = pd.read_csv(args.input, parse_dates=["transaction_date"])
    metadata_path = Path(args.terminal_metadata)
    terminal_metadata = (
        pd.read_csv(metadata_path) if metadata_path.exists() else None
    )
    predictions, metrics = run_walk_forward_benchmark(
        transactions,
        terminal_metadata=terminal_metadata,
        fold_count=args.folds,
        test_days=args.test_days,
    )
    summary = write_benchmark_outputs(predictions, metrics, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
