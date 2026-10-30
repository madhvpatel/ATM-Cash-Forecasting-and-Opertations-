"""Demand Forecaster for ATM Replenishment.

Implements:
1. LightGBM Calendar + Holidays Model (History + Calendar + Payday/RBI Holidays)
2. Pure Exogenous Model (Calendar + Holidays only, robust against lag noise/shocks)
3. Naive Baseline (Previous 7-Day Copy)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb

from replenishment_engine.config import ReplenishmentConfig


# Feature definitions
CALENDAR_FEATURES = ["dow", "dom", "month", "is_weekend", "is_2_4_sat", "is_month_end"]
HOLIDAY_FEATURES = ["is_payday", "is_regional_holiday", "pre_holiday"]
HISTORY_FEATURES = [
    "horizon", "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_std_7", "roll_mean_14", "roll_std_14", "roll_mean_28", "roll_std_28"
]


class ReplenishmentForecaster:
    """Forecaster producing 7-day demand trajectories and prediction intervals."""

    def __init__(
        self,
        config: Optional[ReplenishmentConfig] = None,
        use_uncensored_target: bool = True,
    ):
        self.config = config or ReplenishmentConfig()
        self.use_uncensored_target = use_uncensored_target
        self.target_col = "uncensored_demand" if use_uncensored_target else "demand"

        self.lgb_params = {
            "objective": "regression_l1",  # Minimizes MAE / directly targets WAPE
            "n_estimators": 50,
            "max_depth": 5,
            "learning_rate": 0.08,
            "num_leaves": 25,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            "force_col_wise": True,
        }

    def _extract_calendar_series(self, group: pd.DataFrame) -> Tuple[np.ndarray, ...]:
        dom = group["dom"].to_numpy(dtype=int)
        dow = group["dow"].to_numpy(dtype=int)
        month = group["month"].to_numpy(dtype=int)
        days_in_month = group["date"].dt.days_in_month.to_numpy(dtype=int)
        is_month_end = (days_in_month - dom <= 2).astype(int)
        is_weekend = (dow >= 5).astype(int)
        is_2_4_sat = group["is_2_4_sat"].to_numpy(dtype=int) if "is_2_4_sat" in group.columns else np.zeros(len(group), dtype=int)
        is_payday = group["is_payday"].to_numpy(dtype=int) if "is_payday" in group.columns else np.zeros(len(group), dtype=int)
        is_regional_holiday = group["is_regional_holiday"].to_numpy(dtype=int) if "is_regional_holiday" in group.columns else np.zeros(len(group), dtype=int)
        pre_holiday = group["pre_holiday"].to_numpy(dtype=int) if "pre_holiday" in group.columns else np.zeros(len(group), dtype=int)

        return (
            dow, dom, month, is_weekend, is_2_4_sat, is_month_end,
            is_payday, is_regional_holiday, pre_holiday
        )

    def build_terminal_dataset(
        self,
        terminal_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Construct multi-step forecasting features for a single terminal."""
        group = terminal_data.sort_values("date").reset_index(drop=True)
        demand = group[self.target_col].to_numpy(dtype=float)
        dates = group["date"].to_numpy()
        n = len(group)
        horizon = self.config.horizon_days

        (
            dow, dom, month, is_weekend, is_2_4_sat, is_month_end,
            is_payday, is_regional_holiday, pre_holiday
        ) = self._extract_calendar_series(group)

        rows = []
        # Need at least 28 days of history for rolling statistics
        for i in range(28, n - horizon):
            cutoff_date = dates[i]
            history = demand[: i + 1]

            lag_1 = history[-1]
            lag_7 = history[-7] if i >= 6 else history[-1]
            lag_14 = history[-14] if i >= 13 else history[-1]
            lag_28 = history[-28] if i >= 27 else history[-1]

            roll_mean_7 = np.mean(history[-7:])
            roll_std_7 = np.std(history[-7:])
            roll_mean_14 = np.mean(history[-14:])
            roll_std_14 = np.std(history[-14:])
            roll_mean_28 = np.mean(history[-28:])
            roll_std_28 = np.std(history[-28:])

            for h in range(1, horizon + 1):
                target_idx = i + h
                rows.append({
                    "cutoff_date": cutoff_date,
                    "target_date": dates[target_idx],
                    "horizon": h,
                    "target": demand[target_idx],
                    "lag_1": lag_1,
                    "lag_7": lag_7,
                    "lag_14": lag_14,
                    "lag_28": lag_28,
                    "roll_mean_7": roll_mean_7,
                    "roll_std_7": roll_std_7,
                    "roll_mean_14": roll_mean_14,
                    "roll_std_14": roll_std_14,
                    "roll_mean_28": roll_mean_28,
                    "roll_std_28": roll_std_28,
                    "dow": dow[target_idx],
                    "dom": dom[target_idx],
                    "month": month[target_idx],
                    "is_weekend": is_weekend[target_idx],
                    "is_2_4_sat": is_2_4_sat[target_idx],
                    "is_month_end": is_month_end[target_idx],
                    "is_payday": is_payday[target_idx],
                    "is_regional_holiday": is_regional_holiday[target_idx],
                    "pre_holiday": pre_holiday[target_idx],
                })

        return pd.DataFrame(rows)

    def forecast_7day(
        self,
        df_feat: pd.DataFrame,
        cutoff_date: pd.Timestamp,
        model_type: str = "lgb_calendar_holidays",
    ) -> Dict[str, any]:
        """Generate 7-day point forecasts and cumulative standard error.

        Args:
            df_feat: Feature DataFrame from build_terminal_dataset.
            cutoff_date: Cutoff origin date.
            model_type: One of 'lgb_calendar_holidays', 'pure_exogenous', 'naive_last_week'.

        Returns:
            Dict with 'daily_preds', 'sum_pred', 'sigma_7', 'dates'.
        """
        train = df_feat[df_feat["cutoff_date"] < cutoff_date].copy()
        test = df_feat[df_feat["cutoff_date"] == cutoff_date].copy()

        if len(test) != self.config.horizon_days:
            raise ValueError(f"Expected {self.config.horizon_days} test rows for cutoff {cutoff_date}, got {len(test)}")

        if model_type == "naive_last_week":
            # Just copy lag_7 (which is the corresponding day last week)
            preds = test["lag_7"].to_numpy(dtype=float)
            # Historical residual standard error on train
            residuals = train["target"].to_numpy() - train["lag_7"].to_numpy()
            sigma_daily = float(np.std(residuals))
            sigma_7 = float(sigma_daily * np.sqrt(self.config.horizon_days))

        elif model_type == "pure_exogenous":
            feat_cols = ["horizon"] + CALENDAR_FEATURES + HOLIDAY_FEATURES
            model = lgb.LGBMRegressor(**self.lgb_params)
            model.fit(train[feat_cols], train["target"])
            preds = model.predict(test[feat_cols])
            train_preds = model.predict(train[feat_cols])
            residuals = train["target"].to_numpy() - train_preds
            sigma_daily = float(np.std(residuals))
            sigma_7 = float(sigma_daily * np.sqrt(self.config.horizon_days))

        elif model_type in ["strategy_a_volume_normalized", "volume_normalized"]:
            base_vol = float(np.median(train["target"].to_numpy()))
            base_vol = max(1.0, base_vol)
            train_ratio = train["target"] / base_vol
            feat_cols = ["horizon"] + CALENDAR_FEATURES + HOLIDAY_FEATURES
            model = lgb.LGBMRegressor(**self.lgb_params)
            model.fit(train[feat_cols], train_ratio)
            preds = model.predict(test[feat_cols]) * base_vol
            train_preds = model.predict(train[feat_cols]) * base_vol
            residuals = train["target"].to_numpy() - train_preds
            sigma_daily = float(np.std(residuals))
            sigma_7 = float(sigma_daily * np.sqrt(self.config.horizon_days))

        elif model_type in ["strategy_c_telemetry_adaptive", "telemetry_adaptive"]:
            feat_cols = ["roll_mean_7", "roll_std_7", "roll_mean_28", "lag_7"] + CALENDAR_FEATURES + HOLIDAY_FEATURES
            # Ensure features are present
            available_cols = [c for c in feat_cols if c in train.columns]
            model = lgb.LGBMRegressor(**self.lgb_params)
            model.fit(train[available_cols], train["target"])
            preds = model.predict(test[available_cols])
            train_preds = model.predict(train[available_cols])
            residuals = train["target"].to_numpy() - train_preds
            sigma_daily = float(np.std(residuals))
            sigma_7 = float(sigma_daily * np.sqrt(self.config.horizon_days))

        elif model_type == "lgb_calendar_holidays":
            feat_cols = HISTORY_FEATURES + CALENDAR_FEATURES + HOLIDAY_FEATURES
            model = lgb.LGBMRegressor(**self.lgb_params)
            model.fit(train[feat_cols], train["target"])
            preds = model.predict(test[feat_cols])
            train_preds = model.predict(train[feat_cols])
            residuals = train["target"].to_numpy() - train_preds
            sigma_daily = float(np.std(residuals))
            sigma_7 = float(sigma_daily * np.sqrt(self.config.horizon_days))

        else:
            raise ValueError(f"Unknown model_type '{model_type}'")

        # Clip predictions at 0 (non-negative cash withdrawal)
        preds = np.maximum(preds, 0.0)

        return {
            "model_type": model_type,
            "cutoff_date": cutoff_date,
            "daily_preds": preds.tolist(),
            "sum_pred": float(np.sum(preds)),
            "sigma_7": sigma_7,
            "actual_daily": test["target"].to_numpy(dtype=float).tolist(),
            "actual_sum": float(np.sum(test["target"].to_numpy(dtype=float))),
            "dates": [str(pd.to_datetime(d).date()) for d in test["target_date"]],
        }
