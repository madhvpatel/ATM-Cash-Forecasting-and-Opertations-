from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from lightgbm import LGBMRegressor
except (ImportError, OSError):  # macOS can have the wheel but not its libomp runtime.
    LGBMRegressor = None

from .features import FEATURE_COLUMNS, build_training_features, future_feature_row


QUANTILES = (0.50, 0.80, 0.95)
GLOBAL_CATEGORICAL_FEATURES = ["terminal_id", "denomination", "site_type"]
GLOBAL_NUMERIC_FEATURES = FEATURE_COLUMNS + ["is_recycler"]
GLOBAL_FEATURE_COLUMNS = GLOBAL_NUMERIC_FEATURES + GLOBAL_CATEGORICAL_FEATURES


def new_quantile_regressor(quantile: float, random_state: int = 42) -> object:
    if LGBMRegressor is not None:
        return LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=220,
            learning_rate=0.035,
            num_leaves=24,
            min_child_samples=18,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            verbosity=-1,
        )
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=quantile,
        n_estimators=180,
        learning_rate=0.04,
        max_depth=3,
        min_samples_leaf=12,
        subsample=0.9,
        random_state=random_state,
    )


def new_global_quantile_model(quantile: float, random_state: int = 42) -> Pipeline:
    """Return one train/serve-equivalent global quantile pipeline."""
    preprocessing = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                GLOBAL_CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", GLOBAL_NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("regressor", new_quantile_regressor(quantile, random_state)),
        ]
    )


@dataclass
class QuantileForecaster:
    random_state: int = 42
    models: dict[tuple[int, int, float], object] = field(default_factory=dict)
    histories: dict[tuple[int, int], list[float]] = field(default_factory=dict)
    backtest_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    evaluation_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    evaluation_models: dict[tuple[int, int, float], object] = field(default_factory=dict)
    historical_daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    evaluation_train_end: pd.Timestamp | None = None

    @property
    def model_name(self) -> str:
        return "LightGBM Quantile" if LGBMRegressor is not None else "scikit-learn Gradient Boosting Quantile"

    def _new_model(self, quantile: float) -> object:
        return new_quantile_regressor(quantile, self.random_state)

    @staticmethod
    def _ordered_predictions(predictions: list[np.ndarray], denomination: int) -> np.ndarray:
        ordered = np.maximum.accumulate(np.column_stack(predictions), axis=1)
        return np.rint(ordered / denomination).astype(int) * denomination

    def fit(self, transactions: pd.DataFrame, holdout_days: int = 90) -> "QuantileForecaster":
        """Evaluate on the last holdout window, then refit using all history."""
        training = build_training_features(transactions)
        demand_column = "net_demand" if "net_demand" in transactions.columns else "amount_dispensed"
        raw_daily = (
            transactions.groupby(
                ["terminal_id", "transaction_date", "denomination"], as_index=False
            )[demand_column]
            .sum()
            .sort_values(["terminal_id", "transaction_date", "denomination"])
        )
        raw_daily["transaction_date"] = pd.to_datetime(raw_daily["transaction_date"])
        holdout_start = training["transaction_date"].max() - pd.Timedelta(days=holdout_days - 1)
        self.historical_daily = raw_daily.rename(columns={demand_column: "demand"})
        self.evaluation_train_end = holdout_start - pd.Timedelta(days=1)
        backtest_parts: list[pd.DataFrame] = []

        for (terminal_id, denomination), rows in training.groupby(
            ["terminal_id", "denomination"]
        ):
            terminal_id = int(terminal_id)
            denomination = int(denomination)
            train_rows = rows.loc[rows["transaction_date"] < holdout_start]
            test_rows = rows.loc[rows["transaction_date"] >= holdout_start]
            if train_rows.empty or test_rows.empty:
                raise ValueError("Not enough history for the requested holdout period")

            evaluation_models: list[object] = []
            for quantile in QUANTILES:
                model = self._new_model(quantile)
                model.fit(train_rows[FEATURE_COLUMNS], train_rows["net_demand"])
                evaluation_models.append(model)
                self.evaluation_models[(terminal_id, denomination, quantile)] = model

            ordered = self._ordered_predictions(
                [model.predict(test_rows[FEATURE_COLUMNS]) for model in evaluation_models],
                denomination,
            )
            backtest_parts.append(
                pd.DataFrame(
                    {
                        "terminal_id": terminal_id,
                        "forecast_date": test_rows["transaction_date"].dt.date,
                        "denomination": denomination,
                        "actual_amount": test_rows["net_demand"].astype(int).to_numpy(),
                        "p50_amount": ordered[:, 0],
                        "p80_amount": ordered[:, 1],
                        "p95_amount": ordered[:, 2],
                    }
                )
            )

            for quantile in QUANTILES:
                final_model = self._new_model(quantile)
                final_model.fit(rows[FEATURE_COLUMNS], rows["net_demand"])
                self.models[(terminal_id, denomination, quantile)] = final_model

            history = raw_daily.loc[
                (raw_daily["terminal_id"] == terminal_id)
                & (raw_daily["denomination"] == denomination),
                demand_column,
            ]
            self.histories[(terminal_id, denomination)] = history.astype(float).tolist()

        self.backtest_predictions = (
            pd.concat(backtest_parts, ignore_index=True)
            .sort_values(["terminal_id", "forecast_date", "denomination"])
            .reset_index(drop=True)
        )
        self.evaluation_metrics = evaluate_predictions(self.backtest_predictions)
        return self

    def forecast_at_origins(self, origins: list[pd.Timestamp], days: int) -> pd.DataFrame:
        """Recursive forecasts from each decision morning, with no future actuals.

        Evaluation models are frozen before the holdout. Each origin may consume
        actual observations strictly before that origin; subsequent horizon steps
        consume recursive median predictions only. Origins are batched per model.
        """
        if days < 1 or not self.evaluation_models or self.evaluation_train_end is None:
            raise ValueError("Fit evaluation models and request a positive horizon")
        origins = sorted({pd.Timestamp(day).normalize() for day in origins})
        if not origins or origins[0] <= self.evaluation_train_end:
            raise ValueError("Forecast origins must follow evaluation training")
        rows = []
        for (terminal_id, denomination), daily in self.historical_daily.groupby(
            ["terminal_id", "denomination"]
        ):
            histories = [daily.loc[daily.transaction_date < origin, "demand"].astype(float).tolist()
                         for origin in origins]
            for offset in range(days):
                dates = [origin + pd.Timedelta(days=offset) for origin in origins]
                features = pd.DataFrame([future_feature_row(day, history)
                                         for day, history in zip(dates, histories)])[FEATURE_COLUMNS]
                predictions = [np.maximum(0, self.evaluation_models[(int(terminal_id), int(denomination), q)].predict(features))
                               for q in QUANTILES]
                values = self._ordered_predictions(predictions, int(denomination))
                for i, (origin, day) in enumerate(zip(origins, dates)):
                    histories[i].append(float(values[i, 0]))
                    rows.append({"forecast_origin": origin.date(), "forecast_date": day.date(),
                                 "horizon_step": offset + 1, "terminal_id": int(terminal_id),
                                 "denomination": int(denomination), "p50_amount": int(values[i, 0]),
                                 "p80_amount": int(values[i, 1]), "p95_amount": int(values[i, 2])})
        return pd.DataFrame(rows)

    def forecast(self, start_date: pd.Timestamp, days: int) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("Call fit before forecast")
        rows: list[dict[str, object]] = []

        for (terminal_id, denomination), original_history in self.histories.items():
            history = original_history.copy()
            for offset in range(days):
                forecast_date = start_date.normalize() + pd.Timedelta(days=offset)
                features = pd.DataFrame([future_feature_row(forecast_date, history)])[FEATURE_COLUMNS]
                predictions = [
                    max(
                        0.0,
                        float(
                            self.models[(terminal_id, denomination, quantile)].predict(features)[0]
                        ),
                    )
                    for quantile in QUANTILES
                ]
                ordered = np.maximum.accumulate(predictions)
                rounded = [int(round(value / denomination) * denomination) for value in ordered]
                rows.append(
                    {
                        "terminal_id": terminal_id,
                        "forecast_date": forecast_date.date(),
                        "denomination": denomination,
                        "p50_amount": rounded[0],
                        "p80_amount": rounded[1],
                        "p95_amount": rounded[2],
                    }
                )
                history.append(float(rounded[0]))
        return pd.DataFrame(rows).sort_values(
            ["terminal_id", "forecast_date", "denomination"]
        ).reset_index(drop=True)


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return denomination-level and overall MAE, WAPE, and quantile coverage."""
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, str, pd.DataFrame]] = []
    for (terminal_id, denomination), group in predictions.groupby(
        ["terminal_id", "denomination"]
    ):
        groups.append((str(int(terminal_id)), str(int(denomination)), group))
    for terminal_id, group in predictions.groupby("terminal_id"):
        groups.append((str(int(terminal_id)), "ALL", group))
    for denomination, group in predictions.groupby("denomination"):
        groups.append(("ALL", str(int(denomination)), group))
    groups.append(("ALL", "ALL", predictions))

    for terminal_label, denomination_label, group in groups:
        absolute_error = (group["actual_amount"] - group["p50_amount"]).abs()
        actual_total = float(group["actual_amount"].abs().sum())
        rows.append(
            {
                "terminal_id": terminal_label,
                "denomination": denomination_label,
                "observations": len(group),
                "mae": round(float(absolute_error.mean()), 2),
                "wape": round(float(absolute_error.sum() / actual_total), 4) if actual_total else 0.0,
                "p50_coverage": round(float((group["actual_amount"] <= group["p50_amount"]).mean()), 4),
                "p80_coverage": round(float((group["actual_amount"] <= group["p80_amount"]).mean()), 4),
                "p95_coverage": round(float((group["actual_amount"] <= group["p95_amount"]).mean()), 4),
            }
        )
    return pd.DataFrame(rows)
