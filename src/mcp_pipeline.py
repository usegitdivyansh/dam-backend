"""
config system,
feature engineering,
leakage auditing,
modeling,
evaluation,
plotting,
orchestration,
experiment running,

all in one file."""

import copy
import json
import re
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import HuberRegressor, LogisticRegression, QuantileRegressor
from sklearn.metrics import mean_absolute_error, r2_score, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import (
    Config,
    MARKET_CUTOFFS,
    MARKET_SPECS,
    MarketPolicy,
    MarketSpec,
    config_from_dict,
    infer_market_column,
    infer_renewable_columns,
    parse_args,
    parse_int_list,
    parse_market_columns,
    parse_str_list,
    resolve_market_columns,
    resolve_market_policy,
)
from dataclasses import dataclass

try:
    from .evaluation.plots import (
        evaluate_price_buckets,
        build_calibration_table,
        plot_calibration_curve,
        compute_rolling_metrics,
        hard_switch_predictions,
        cost_weighted_mae,
        spike_weighted_penalty,
        underprediction_penalty,
        compute_daily_procurement_costs,
        plot_cumulative_monetary_error,
        compute_daily_aggregate_errors,
        summarize_daily_aggregate_errors,
        plot_daily_totals,
        plot_regime_confusion_matrix,
        plot_regime_distribution_over_time,
        plot_spike_timeline,
    )
except ImportError:
    from evaluation.plots import (
        evaluate_price_buckets,
        build_calibration_table,
        plot_calibration_curve,
        compute_rolling_metrics,
        hard_switch_predictions,
        cost_weighted_mae,
        spike_weighted_penalty,
        underprediction_penalty,
        compute_daily_procurement_costs,
        plot_cumulative_monetary_error,
        compute_daily_aggregate_errors,
        summarize_daily_aggregate_errors,
        plot_daily_totals,
        plot_regime_confusion_matrix,
        plot_regime_distribution_over_time,
        plot_spike_timeline,
    )

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


@dataclass(frozen=True)
class FeatureTemporalMeta:
    latest_offset_steps: int | None
    reason: str
    deterministic: bool = False


FEATURE_GROUP_TEMPORAL = "temporal"
FEATURE_GROUP_TARGET_LAG = "target_lag"
FEATURE_GROUP_TARGET_ROLL = "target_roll"
FEATURE_GROUP_MARKET_STATE = "market_state"
FEATURE_GROUP_RENEWABLE = "renewable"
FEATURE_GROUP_WEATHER = "weather"
FEATURE_GROUP_CROSS_LAG = "cross_market_lag"
FEATURE_GROUP_CROSS_SPREAD = "cross_market_spread"
FEATURE_GROUP_CROSS_VOL = "cross_market_vol"

MARKET_POLICIES = {
    "DAM": MarketPolicy(
        allowed_groups={
            FEATURE_GROUP_TEMPORAL,
            FEATURE_GROUP_TARGET_LAG,
            FEATURE_GROUP_TARGET_ROLL,
            FEATURE_GROUP_MARKET_STATE,
            FEATURE_GROUP_WEATHER,
            FEATURE_GROUP_CROSS_LAG,
            FEATURE_GROUP_CROSS_SPREAD,
            FEATURE_GROUP_CROSS_VOL,
        },
        allowed_cross_markets={"GDAM", "RTM"},
    ),
    "GDAM": MarketPolicy(
        allowed_groups={
            FEATURE_GROUP_TEMPORAL,
            FEATURE_GROUP_TARGET_LAG,
            FEATURE_GROUP_TARGET_ROLL,
            FEATURE_GROUP_MARKET_STATE,
            FEATURE_GROUP_RENEWABLE,
            FEATURE_GROUP_WEATHER,
            FEATURE_GROUP_CROSS_LAG,
            FEATURE_GROUP_CROSS_SPREAD,
            FEATURE_GROUP_CROSS_VOL,
        },
        allowed_cross_markets={"DAM", "RTM"},
    ),
    "RTM": MarketPolicy(
        allowed_groups={
            FEATURE_GROUP_TEMPORAL,
            FEATURE_GROUP_TARGET_LAG,
            FEATURE_GROUP_TARGET_ROLL,
            FEATURE_GROUP_WEATHER,
            FEATURE_GROUP_CROSS_LAG,
            FEATURE_GROUP_CROSS_SPREAD,
            FEATURE_GROUP_CROSS_VOL,
        },
        allowed_cross_markets={"DAM", "GDAM"},
    ),
}


def sanitize_feature_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower())
    return cleaned.strip("_")


def build_cutoff_audit_tables(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    feature_cols: list[str],
    datetime_col: str,
) -> dict[str, pd.DataFrame]:
    before = add_delivery_block_columns(df_before, datetime_col)
    after = add_delivery_block_columns(df_after, datetime_col)

    before_rows = len(before)
    after_rows = len(after)
    dropped_rows = before_rows - after_rows
    retained_rows = after_rows
    retention_rate = retained_rows / before_rows if before_rows else 0.0

    summary = pd.DataFrame(
        [
            {"metric": "rows_before_masking", "value": before_rows},
            {"metric": "rows_after_masking", "value": after_rows},
            {"metric": "rows_dropped", "value": dropped_rows},
            {"metric": "retention_rate", "value": retention_rate},
            {
                "metric": "all_delivery_blocks_represented",
                "value": bool(after["delivery_block"].nunique() == 96) if after_rows else False,
            },
        ]
    )

    block_counts = (
        after.groupby(["delivery_block", "delivery_hour"], dropna=False)
        .size()
        .reset_index(name="retained_rows")
        .sort_values(["delivery_block"])
    )
    expected_blocks = pd.DataFrame({"delivery_block": range(1, 97)})
    expected_blocks["delivery_hour"] = (expected_blocks["delivery_block"] - 1) // 4
    block_counts = expected_blocks.merge(block_counts, on=["delivery_block", "delivery_hour"], how="left")
    block_counts["retained_rows"] = block_counts["retained_rows"].fillna(0).astype(int)
    block_counts["retained"] = block_counts["retained_rows"] > 0

    feature_availability = []
    for col in feature_cols:
        if col not in after.columns:
            continue
        available_rows = int(after[col].notna().sum())
        feature_availability.append(
            {
                "feature_name": col,
                "available_rows": available_rows,
                "availability_rate": available_rows / after_rows if after_rows else 0.0,
            }
        )
    feature_availability_df = pd.DataFrame(feature_availability).sort_values(
        ["availability_rate", "feature_name"], ascending=[True, True]
    )

    return {
        "summary": summary,
        "retained_rows_by_block": block_counts,
        "feature_availability": feature_availability_df,
    }


def add_delivery_block_columns(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    df = df.copy()
    timestamps = pd.to_datetime(df[datetime_col])
    df["delivery_hour"] = timestamps.dt.hour
    quarter = (timestamps.dt.minute // 15).astype(int)
    df["delivery_block"] = df["delivery_hour"] * 4 + quarter + 1
    return df


def infer_step_size(datetime_series: pd.Series) -> pd.Timedelta:
    series = pd.to_datetime(datetime_series).sort_values()
    diffs = series.diff().dropna()
    if diffs.empty:
        return pd.Timedelta(minutes=15)
    median = diffs.median()
    if pd.isna(median) or median <= pd.Timedelta(0):
        return pd.Timedelta(minutes=15)
    return median


def compute_issue_timestamps(
    delivery_ts: pd.Series,
    target_market: str,
    cutoff_hour: int,
    cutoff_minute: int,
    step: pd.Timedelta,
    lead_steps: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    delivery_ts = pd.to_datetime(delivery_ts)
    if target_market in {"DAM", "GDAM"}:
        delivery_day = delivery_ts.dt.floor("D")
        cutoff_ts = (
            delivery_day
            - pd.Timedelta(days=1)
            + pd.Timedelta(hours=cutoff_hour, minutes=cutoff_minute)
        )
        issue_ts = cutoff_ts.copy()
    else:
        issue_ts = delivery_ts - lead_steps * step
        cutoff_ts = issue_ts.copy()

    issue_offset = ((delivery_ts - issue_ts) / step).round().astype(int)
    issue_offset = issue_offset.clip(lower=0)
    return issue_ts, cutoff_ts, issue_offset


def apply_cutoff_mask(
    df: pd.DataFrame,
    feature_temporal: dict[str, FeatureTemporalMeta],
    issue_offset_steps: pd.Series,
) -> None:
    for name in feature_temporal:
        meta = feature_temporal.get(name)
        if meta is None or meta.deterministic:
            continue
        if meta.latest_offset_steps is None:
            df[name] = np.nan
            continue
        unsafe_mask = issue_offset_steps > meta.latest_offset_steps
        if unsafe_mask.any() and name in df.columns:
            df.loc[unsafe_mask, name] = np.nan


def build_feature_validation(
    feature_temporal: dict[str, FeatureTemporalMeta],
    delivery_ts: pd.Series,
    issue_ts: pd.Series,
    issue_offset_steps: pd.Series,
    step: pd.Timedelta,
    df: pd.DataFrame,
) -> pd.DataFrame:
    if delivery_ts.empty:
        return pd.DataFrame(
            columns=[
                "feature_name",
                "effective_latest_timestamp",
                "latest_source_timestamp",
                "forecast_issue_timestamp",
                "feature_offset_steps",
                "issue_offset_steps",
                "leakage_safe",
                "violation_count",
                "reason",
            ]
        )

    issue_offset = int(issue_offset_steps.max()) if len(issue_offset_steps) else 0
    latest_delivery = pd.to_datetime(delivery_ts.iloc[-1])
    latest_issue = pd.to_datetime(issue_ts.iloc[-1]) if len(issue_ts) else pd.NaT

    rows: list[dict[str, object]] = []
    for name, meta in feature_temporal.items():
        feature_offset = meta.latest_offset_steps
        if feature_offset is None:
            effective = "t-unknown"
            latest_source = pd.NaT
        elif feature_offset == 0:
            effective = "t"
            latest_source = pd.NaT
        else:
            effective = f"t-{feature_offset}"
            latest_source = latest_delivery - feature_offset * step

        leakage_safe = meta.deterministic or (
            feature_offset is not None and feature_offset <= issue_offset
        )
        if feature_offset is None:
            violation_count = int(issue_offset_steps.notna().sum())
        else:
            violation_count = int((issue_offset_steps > feature_offset).sum())

        rows.append(
            {
                "feature_name": name,
                "effective_latest_timestamp": effective,
                "latest_source_timestamp": latest_source,
                "forecast_issue_timestamp": latest_issue,
                "feature_offset_steps": feature_offset,
                "issue_offset_steps": issue_offset,
                "leakage_safe": bool(leakage_safe),
                "violation_count": violation_count if not leakage_safe else 0,
                "reason": meta.reason,
            }
        )

    return pd.DataFrame(rows)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    # SMAPE: symmetric mean absolute percentage error (in percent)
    num = np.abs(y_pred - y_true)
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    return float(np.mean(2.0 * num / denom) * 100)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }


def build_sample_weights(y_values: np.ndarray) -> np.ndarray:
    threshold = np.percentile(y_values, 90)
    return np.where(y_values > threshold, 3.0, 1.0)


def build_sample_weights_configured(
    y_values: np.ndarray,
    quantile: float,
    high_weight: float,
) -> np.ndarray:
    threshold = np.percentile(y_values, quantile * 100.0)
    return np.where(y_values > threshold, high_weight, 1.0)


def transform_target(y_values: np.ndarray, target_transform: str) -> np.ndarray:
    if target_transform == "none":
        return y_values
    if target_transform == "log1p":
        if np.any(y_values < 0):
            raise ValueError("log1p target requested but negative values found.")
        return np.log1p(y_values)
    if target_transform == "asinh":
        return np.arcsinh(y_values)
    raise ValueError(f"Unsupported target_transform: {target_transform}")


def inverse_transform_target(y_values: np.ndarray, target_transform: str) -> np.ndarray:
    if target_transform == "none":
        return y_values
    if target_transform == "log1p":
        return np.expm1(y_values)
    if target_transform == "asinh":
        return np.sinh(y_values)
    raise ValueError(f"Unsupported target_transform: {target_transform}")


def fit_with_weights(model: object, X: pd.DataFrame, y: pd.Series, weights: np.ndarray) -> object:
    if isinstance(model, Pipeline):
        step_name = model.steps[-1][0]
        try:
            return model.fit(X, y, **{f"{step_name}__sample_weight": weights})
        except TypeError:
            return model.fit(X, y)
    try:
        return model.fit(X, y, sample_weight=weights)
    except TypeError:
        return model.fit(X, y)


def resolve_backend(preferred: str) -> str:
    if preferred == "lightgbm" and HAS_LGBM:
        return "lightgbm"
    if preferred == "xgboost" and HAS_XGB:
        return "xgboost"
    if HAS_LGBM:
        return "lightgbm"
    if HAS_XGB:
        return "xgboost"
    raise ImportError("Install lightgbm or xgboost to run the GBM model.")


def resolve_quantile_backend(preferred: str) -> str:
    if preferred == "lightgbm" and HAS_LGBM:
        return "lightgbm"
    return "sklearn"


def normalize_lags(lags: list[int], lead_steps: int) -> list[int]:
    normalized = []
    for lag in lags:
        try:
            lag_value = int(lag)
        except (TypeError, ValueError):
            continue
        if lag_value > 0:
            normalized.append(lag_value)
    return sorted(set(normalized))


def feature_engineering(
    df: pd.DataFrame,
    datetime_col: str,
    target_market: str,
    target_col: str,
    market_columns: dict[str, str],
    lead_steps: int,
    target_lags: list[int],
    rolling_windows: list[int],
    cross_market_lags: list[int],
    renewable_cols: list[str],
    buy_col: str,
    sell_col: str,
    solar_col: str,
    weather_cols: list,
    market_policy: MarketPolicy,
    auction_cutoff_hour: int,
    auction_cutoff_minute: int,
    dropna_target: bool = True,
) -> tuple[pd.DataFrame, list, dict[str, str], pd.DataFrame]:
    required = [datetime_col, target_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)
    delivery_ts = df[datetime_col]
    step = infer_step_size(delivery_ts)
    issue_ts, cutoff_ts, issue_offset_steps = compute_issue_timestamps(
        delivery_ts,
        target_market=target_market,
        cutoff_hour=auction_cutoff_hour,
        cutoff_minute=auction_cutoff_minute,
        step=step,
        lead_steps=lead_steps,
    )
    df["delivery_timestamp"] = delivery_ts
    df["forecast_issue_timestamp"] = issue_ts
    df["auction_cutoff_timestamp"] = cutoff_ts
    df["issue_offset_steps"] = issue_offset_steps

    df["hour"] = df[datetime_col].dt.hour
    df["weekday"] = df[datetime_col].dt.weekday
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    target_lags = normalize_lags(target_lags, lead_steps)
    cross_market_lags = normalize_lags(cross_market_lags, lead_steps)

    safe_cols: dict[str, str] = {}
    feature_groups: dict[str, str] = {}
    feature_temporal: dict[str, FeatureTemporalMeta] = {}

    def register_feature(
        name: str,
        group: str,
        latest_offset_steps: int | None,
        reason: str,
        deterministic: bool = False,
    ) -> None:
        feature_groups[name] = group
        feature_temporal[name] = FeatureTemporalMeta(
            latest_offset_steps=latest_offset_steps,
            reason=reason,
            deterministic=deterministic,
        )

    def add_safe(col: str) -> None:
        safe_name = f"{col}_safe"
        df[safe_name] = df[col].shift(lead_steps)
        safe_cols[col] = safe_name

    if not renewable_cols:
        renewable_cols = infer_renewable_columns(df.columns.tolist())
    renewable_cols = [col for col in renewable_cols if col in df.columns]

    for col in [target_col, buy_col, sell_col, solar_col, *renewable_cols]:
        if col and col in df.columns:
            add_safe(col)

    weather_features = [col for col in weather_cols if col in df.columns]
    for col in weather_features:
        add_safe(col)

    cross_market_cols = {
        market: col
        for market, col in market_columns.items()
        if market != target_market
        and market in market_policy.allowed_cross_markets
        and col in df.columns
    }
    for col in cross_market_cols.values():
        add_safe(col)

    for lag in target_lags:
        df[f"price_lag_{lag}"] = df[target_col].shift(lag)
        register_feature(
            f"price_lag_{lag}",
            FEATURE_GROUP_TARGET_LAG,
            latest_offset_steps=lag,
            reason=f"Target lag {lag}",
        )

    safe_price = df[safe_cols[target_col]]
    for window in rolling_windows:
        df[f"price_rolling_mean_{window}"] = safe_price.rolling(window).mean()
        df[f"price_rolling_std_{window}"] = safe_price.rolling(window).std()
        register_feature(
            f"price_rolling_mean_{window}",
            FEATURE_GROUP_TARGET_ROLL,
            latest_offset_steps=lead_steps,
            reason="Rolling mean on lead-shifted target",
        )
        register_feature(
            f"price_rolling_std_{window}",
            FEATURE_GROUP_TARGET_ROLL,
            latest_offset_steps=lead_steps,
            reason="Rolling std on lead-shifted target",
        )

    if buy_col in safe_cols and sell_col in safe_cols:
        safe_sell = df[safe_cols[sell_col]].replace(0, np.nan)
        df["demand_supply_ratio"] = df[safe_cols[buy_col]] / (safe_sell + 1e-6)
        register_feature(
            "demand_supply_ratio",
            FEATURE_GROUP_MARKET_STATE,
            latest_offset_steps=lead_steps,
            reason="Ratio of lead-shifted buy/sell",
        )

    if solar_col in safe_cols:
        df["solar_hour_interaction"] = df[safe_cols[solar_col]] * df["hour_sin"]
        register_feature(
            "solar_hour_interaction",
            FEATURE_GROUP_RENEWABLE,
            latest_offset_steps=lead_steps,
            reason="Lead-shifted solar with calendar interaction",
        )

    cross_market_feature_cols = []
    for market, col in cross_market_cols.items():
        for lag in cross_market_lags:
            feature_name = f"{market.lower()}_price_lag_{lag}"
            df[feature_name] = df[col].shift(lag)
            cross_market_feature_cols.append(feature_name)
            register_feature(
                feature_name,
                FEATURE_GROUP_CROSS_LAG,
                latest_offset_steps=lag,
                reason=f"Cross-market lag {lag}",
            )

    spread_feature_cols = []
    spread_roll_feature_cols = []
    cross_market_vol_cols = []
    explicit_cross_feature_cols = []
    interaction_feature_cols = []
    rolling_stat_feature_cols = []

    allowed_markets = {target_market} | set(market_policy.allowed_cross_markets)
    safe_market_cols = {
        market: safe_cols[col]
        for market, col in market_columns.items()
        if market in allowed_markets and col in safe_cols
    }

    required_markets = ["DAM", "GDAM", "RTM"]
    required_safe_cols = {
        market: safe_cols[col]
        for market, col in market_columns.items()
        if market in required_markets and col in safe_cols
    }
    for market, safe_col in required_safe_cols.items():
        for lag in [1, 4, 96]:
            feature_name = f"{market}_MCP_lag_{lag}"
            df[feature_name] = df[safe_col].shift(lag)
            explicit_cross_feature_cols.append(feature_name)
            register_feature(
                feature_name,
                FEATURE_GROUP_CROSS_LAG,
                latest_offset_steps=lead_steps + lag,
                reason=f"Lead-shifted market lag {lag}",
            )

    spread_specs = [
        ("DAM", "GDAM", "DAM_GDAM_spread"),
        ("RTM", "GDAM", "RTM_GDAM_spread"),
        ("RTM", "DAM", "RTM_DAM_spread"),
    ]
    for market_a, market_b, spread_name in spread_specs:
        if market_a in required_safe_cols and market_b in required_safe_cols:
            df[spread_name] = (
                df[required_safe_cols[market_a]]
                - df[required_safe_cols[market_b]]
            )
            spread_feature_cols.append(spread_name)
            register_feature(
                spread_name,
                FEATURE_GROUP_CROSS_SPREAD,
                latest_offset_steps=lead_steps,
                reason="Spread of lead-shifted markets",
            )

    safe_price = df[safe_cols[target_col]]
    df["rolling_mean_96"] = safe_price.rolling(96).mean()
    df["rolling_std_96"] = safe_price.rolling(96).std()
    df["rolling_volatility"] = safe_price.pct_change().rolling(96).std()
    rolling_stat_feature_cols.extend(
        ["rolling_mean_96", "rolling_std_96", "rolling_volatility"]
    )
    register_feature(
        "rolling_mean_96",
        FEATURE_GROUP_TARGET_ROLL,
        latest_offset_steps=lead_steps,
        reason="Rolling mean on lead-shifted target",
    )
    register_feature(
        "rolling_std_96",
        FEATURE_GROUP_TARGET_ROLL,
        latest_offset_steps=lead_steps,
        reason="Rolling std on lead-shifted target",
    )
    register_feature(
        "rolling_volatility",
        FEATURE_GROUP_TARGET_ROLL,
        latest_offset_steps=lead_steps,
        reason="Rolling volatility on lead-shifted target",
    )

    ratio_specs = [
        ("GDAM", "DAM", "GDAM_to_DAM_ratio"),
        ("RTM", "DAM", "RTM_to_DAM_ratio"),
        ("GDAM", "RTM", "GDAM_to_RTM_ratio"),
    ]
    for numerator, denominator, ratio_name in ratio_specs:
        if numerator in required_safe_cols and denominator in required_safe_cols:
            df[ratio_name] = (
                df[required_safe_cols[numerator]]
                / (df[required_safe_cols[denominator]] + 1e-6)
            )
            interaction_feature_cols.append(ratio_name)
            register_feature(
                ratio_name,
                FEATURE_GROUP_CROSS_SPREAD,
                latest_offset_steps=lead_steps,
                reason="Ratio of lead-shifted markets",
            )
    markets = sorted(safe_market_cols.keys())
    for idx, market_a in enumerate(markets):
        for market_b in markets[idx + 1 :]:
            safe_a = df[safe_market_cols[market_a]]
            safe_b = df[safe_market_cols[market_b]]
            spread_name = f"{market_a.lower()}_{market_b.lower()}_spread"
            df[spread_name] = safe_a - safe_b
            spread_feature_cols.append(spread_name)
            register_feature(
                spread_name,
                FEATURE_GROUP_CROSS_SPREAD,
                latest_offset_steps=lead_steps,
                reason="Spread of lead-shifted markets",
            )
            for window in rolling_windows:
                mean_name = f"{spread_name}_roll_mean_{window}"
                std_name = f"{spread_name}_roll_std_{window}"
                df[mean_name] = df[spread_name].rolling(window).mean()
                df[std_name] = df[spread_name].rolling(window).std()
                spread_roll_feature_cols.extend([mean_name, std_name])
                register_feature(
                    mean_name,
                    FEATURE_GROUP_CROSS_SPREAD,
                    latest_offset_steps=lead_steps,
                    reason="Rolling mean on lead-shifted spread",
                )
                register_feature(
                    std_name,
                    FEATURE_GROUP_CROSS_SPREAD,
                    latest_offset_steps=lead_steps,
                    reason="Rolling std on lead-shifted spread",
                )

    for market, safe_col in safe_market_cols.items():
        for window in rolling_windows:
            vol_name = f"{market.lower()}_vol_{window}"
            df[vol_name] = df[safe_col].rolling(window).std()
            cross_market_vol_cols.append(vol_name)
            register_feature(
                vol_name,
                FEATURE_GROUP_CROSS_VOL,
                latest_offset_steps=lead_steps,
                reason="Rolling volatility on lead-shifted market",
            )

    renewable_feature_cols = []
    if renewable_cols:
        safe_renewables = [safe_cols[col] for col in renewable_cols if col in safe_cols]
        total_renewable = df[safe_renewables].sum(axis=1)
        df["renewable_total"] = total_renewable
        renewable_feature_cols.append("renewable_total")
        register_feature(
            "renewable_total",
            FEATURE_GROUP_RENEWABLE,
            latest_offset_steps=lead_steps,
            reason="Sum of lead-shifted renewables",
        )
        for col in renewable_cols:
            safe_col = safe_cols.get(col)
            if safe_col:
                ratio_name = f"{sanitize_feature_name(col)}_mix_ratio"
                df[ratio_name] = df[safe_col] / (total_renewable + 1e-6)
                renewable_feature_cols.append(ratio_name)
                register_feature(
                    ratio_name,
                    FEATURE_GROUP_RENEWABLE,
                    latest_offset_steps=lead_steps,
                    reason="Ratio of lead-shifted renewables",
                )
        if sell_col in safe_cols:
            df["renewable_supply_ratio"] = total_renewable / (df[safe_cols[sell_col]] + 1e-6)
            renewable_feature_cols.append("renewable_supply_ratio")
            register_feature(
                "renewable_supply_ratio",
                FEATURE_GROUP_RENEWABLE,
                latest_offset_steps=lead_steps,
                reason="Ratio of lead-shifted renewables and supply",
            )

    imbalance_feature_cols = []
    if buy_col in safe_cols and sell_col in safe_cols:
        df["market_imbalance"] = df[safe_cols[buy_col]] - df[safe_cols[sell_col]]
        df["market_imbalance_abs"] = df["market_imbalance"].abs()
        imbalance_feature_cols.extend(["market_imbalance", "market_imbalance_abs"])
        register_feature(
            "market_imbalance",
            FEATURE_GROUP_MARKET_STATE,
            latest_offset_steps=lead_steps,
            reason="Difference of lead-shifted buy/sell",
        )
        register_feature(
            "market_imbalance_abs",
            FEATURE_GROUP_MARKET_STATE,
            latest_offset_steps=lead_steps,
            reason="Abs difference of lead-shifted buy/sell",
        )
        for window in rolling_windows:
            mean_name = f"market_imbalance_roll_mean_{window}"
            std_name = f"market_imbalance_roll_std_{window}"
            df[mean_name] = df["market_imbalance"].rolling(window).mean()
            df[std_name] = df["market_imbalance"].rolling(window).std()
            imbalance_feature_cols.extend([mean_name, std_name])
            register_feature(
                mean_name,
                FEATURE_GROUP_MARKET_STATE,
                latest_offset_steps=lead_steps,
                reason="Rolling mean on lead-shifted imbalance",
            )
            register_feature(
                std_name,
                FEATURE_GROUP_MARKET_STATE,
                latest_offset_steps=lead_steps,
                reason="Rolling std on lead-shifted imbalance",
            )
            ratio_mean = f"demand_supply_ratio_roll_mean_{window}"
            ratio_std = f"demand_supply_ratio_roll_std_{window}"
            df[ratio_mean] = df["demand_supply_ratio"].rolling(window).mean()
            df[ratio_std] = df["demand_supply_ratio"].rolling(window).std()
            imbalance_feature_cols.extend([ratio_mean, ratio_std])
            register_feature(
                ratio_mean,
                FEATURE_GROUP_MARKET_STATE,
                latest_offset_steps=lead_steps,
                reason="Rolling mean on lead-shifted ratio",
            )
            register_feature(
                ratio_std,
                FEATURE_GROUP_MARKET_STATE,
                latest_offset_steps=lead_steps,
                reason="Rolling std on lead-shifted ratio",
            )

    feature_cols = [
        "hour",
        "weekday",
        "hour_sin",
        "hour_cos",
        *[f"price_lag_{lag}" for lag in target_lags],
        *[f"price_rolling_mean_{window}" for window in rolling_windows],
        *[f"price_rolling_std_{window}" for window in rolling_windows],
    ]
    for name in ["hour", "weekday", "hour_sin", "hour_cos"]:
        register_feature(
            name,
            FEATURE_GROUP_TEMPORAL,
            latest_offset_steps=0,
            reason="Calendar feature from target timestamp",
            deterministic=True,
        )

    if buy_col in safe_cols:
        feature_cols.append(safe_cols[buy_col])
        register_feature(
            safe_cols[buy_col],
            FEATURE_GROUP_MARKET_STATE,
            latest_offset_steps=lead_steps,
            reason="Lead-shifted buy volume",
        )
    if sell_col in safe_cols:
        feature_cols.append(safe_cols[sell_col])
        register_feature(
            safe_cols[sell_col],
            FEATURE_GROUP_MARKET_STATE,
            latest_offset_steps=lead_steps,
            reason="Lead-shifted sell volume",
        )
    if "demand_supply_ratio" in df.columns:
        feature_cols.append("demand_supply_ratio")
    if "solar_hour_interaction" in df.columns:
        feature_cols.append("solar_hour_interaction")

    weather_safe = [safe_cols[col] for col in weather_features]
    feature_cols.extend(weather_safe)
    for col in weather_safe:
        register_feature(
            col,
            FEATURE_GROUP_WEATHER,
            latest_offset_steps=lead_steps,
            reason="Lead-shifted weather",
        )
    feature_cols.extend(cross_market_feature_cols)
    feature_cols.extend(explicit_cross_feature_cols)
    feature_cols.extend(spread_feature_cols)
    feature_cols.extend(spread_roll_feature_cols)
    feature_cols.extend(cross_market_vol_cols)
    feature_cols.extend(rolling_stat_feature_cols)
    feature_cols.extend(interaction_feature_cols)
    feature_cols.extend(renewable_feature_cols)
    feature_cols.extend(imbalance_feature_cols)

    feature_cols = list(dict.fromkeys(feature_cols))
    if target_col in feature_cols:
        feature_cols.remove(target_col)

    feature_cols = [
        col
        for col in feature_cols
        if feature_groups.get(col, FEATURE_GROUP_TARGET_LAG)
        in market_policy.allowed_groups
    ]

    apply_cutoff_mask(df, feature_temporal, issue_offset_steps)
    validation_df = build_feature_validation(
        feature_temporal,
        delivery_ts=delivery_ts,
        issue_ts=issue_ts,
        issue_offset_steps=issue_offset_steps,
        step=step,
        df=df,
    )
    unknown = set(
        validation_df.loc[validation_df["feature_offset_steps"].isna(), "feature_name"].tolist()
    )
    feature_cols = [col for col in feature_cols if col not in unknown]

    if dropna_target:
        df = df.dropna(subset=[target_col]).reset_index(drop=True)
    return df, feature_cols, feature_groups, validation_df


def time_train_test_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def build_huber_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("huber", HuberRegressor(max_iter=1000)),
        ]
    )


def build_gbm_model(backend: str, params: dict | None = None) -> object:
    params = params or {}
    if backend == "lightgbm":
        base = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
        }
        base.update(params)
        return lgb.LGBMRegressor(objective="regression", **base)
    if backend == "xgboost":
        base = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
        }
        base.update(params)
        return xgb.XGBRegressor(objective="reg:squarederror", **base)
    raise ValueError(f"Unsupported backend: {backend}")


def build_quantile_model(backend: str, quantile: float) -> object:
    if backend == "lightgbm":
        return lgb.LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            # Small alpha improves convergence stability for QuantileRegressor.
            ("quantile", QuantileRegressor(quantile=quantile, alpha=1e-4)),
        ]
    )


def build_regime_classifier() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def build_spike_classifier() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000)),
        ]
    )


def validate_classifier_nan_robustness(
    X: pd.DataFrame,
    y: pd.Series,
    target_market: str,
    output_dir: Path,
) -> None:
    if target_market not in {"DAM", "GDAM"}:
        return

    checks: list[dict[str, object]] = []

    regime_labels = (y > y.quantile(0.9)).astype(int)
    if regime_labels.nunique() < 2 and len(regime_labels) >= 2:
        regime_labels = regime_labels.copy()
        regime_labels.iloc[-1] = 1 - int(regime_labels.iloc[-1])

    regime_ok = True
    regime_error = ""
    try:
        regime_clf = build_regime_classifier()
        regime_clf.fit(X, regime_labels)
        regime_clf.predict_proba(X.iloc[: min(len(X), 32)])
    except Exception as exc:  # pragma: no cover - defensive logging path
        regime_ok = False
        regime_error = str(exc)
    checks.append(
        {
            "check": "regime_classifier_fit_with_masked_features",
            "passed": regime_ok,
            "error": regime_error,
        }
    )

    spike_labels = (y > y.quantile(0.92)).astype(int)
    if spike_labels.nunique() < 2 and len(spike_labels) >= 2:
        spike_labels = spike_labels.copy()
        spike_labels.iloc[-1] = 1 - int(spike_labels.iloc[-1])

    spike_ok = True
    spike_error = ""
    try:
        spike_clf = build_spike_classifier()
        spike_clf.fit(X, spike_labels)
        spike_clf.predict_proba(X.iloc[: min(len(X), 32)])
    except Exception as exc:  # pragma: no cover - defensive logging path
        spike_ok = False
        spike_error = str(exc)
    checks.append(
        {
            "check": "spike_classifier_fit_with_masked_features",
            "passed": spike_ok,
            "error": spike_error,
        }
    )

    check_df = pd.DataFrame(checks)
    check_df.to_csv(output_dir / "classifier_nan_robustness.csv", index=False)
    if not bool(check_df["passed"].all()):
        failures = check_df.loc[~check_df["passed"], "error"].tolist()
        raise RuntimeError(
            "Classifier NaN-robustness validation failed after cutoff masking: "
            + " | ".join(failures)
        )



def compute_dynamic_labels(
    y_series: pd.Series,
    datetime_series: pd.Series | None = None,
    strategy: str = "global",
    q_med: float = 0.90,
    q_ext: float = 0.98,
    classes: int = 3,
    rolling_window_days: int = 30,
    rolling_min_periods: int | None = None,
    zscore_threshold: float = 1.28,
    vol_multiplier: float = 2.0,
    baseline_lag: int = 1,
    bidirectional: bool = False,
    valley_quantile: float | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if not bidirectional and q_ext <= q_med:
        raise ValueError("regime_extreme_quantile must be greater than regime_quantile")
    if bidirectional:
        if valley_quantile is None:
            valley_quantile = 1.0 - q_med
        if not 0.0 < valley_quantile < 1.0:
            raise ValueError("regime_valley_quantile must be between 0 and 1")
        if not 0.0 < q_med < 1.0:
            raise ValueError("regime_quantile must be between 0 and 1")
        if valley_quantile >= q_med:
            raise ValueError("regime_valley_quantile must be less than regime_quantile")
    y_series = pd.Series(y_series.values, index=y_series.index)
    prices = y_series.values
    labels = np.ones(len(prices), dtype=int) if bidirectional else np.zeros(len(prices), dtype=int)

    def estimate_rows_per_day(series: pd.Series | None) -> int:
        if series is None:
            return 1
        dt = pd.to_datetime(series)
        diffs = dt.sort_values().diff().dropna()
        if diffs.empty:
            return 1
        median = diffs.median()
        if pd.isna(median) or median <= pd.Timedelta(0):
            return 1
        rows = int(round(pd.Timedelta(days=1) / median))
        return max(1, rows)

    rows_per_day = estimate_rows_per_day(datetime_series)
    window_size = max(1, int(rolling_window_days * rows_per_day))
    min_periods = (
        rolling_min_periods
        if rolling_min_periods is not None
        else max(1, int(window_size * 0.25))
    )

    if strategy == "rolling":
        strategy = "rolling_quantile"

    # Use history-only rolling windows; never backward-fill to avoid leakage.
    history = y_series.shift(1)

    if strategy == "global":
        if bidirectional:
            t_low = np.percentile(prices, valley_quantile * 100)
            t_high = np.percentile(prices, q_med * 100)
            thresh_low = pd.Series(t_low, index=y_series.index)
            thresh_high = pd.Series(t_high, index=y_series.index)
        else:
            tmed = np.percentile(prices, q_med * 100)
            text = np.percentile(prices, q_ext * 100)
            thresh_med = pd.Series(tmed, index=y_series.index)
            thresh_ext = pd.Series(text, index=y_series.index)
    elif strategy == "daily":
        if datetime_series is None:
            raise ValueError("datetime_series is required for daily thresholds")
        dates = pd.to_datetime(datetime_series).dt.date
        df = pd.DataFrame({"price": prices, "date": dates}, index=y_series.index)
        if bidirectional:
            thresh_low = df.groupby("date")["price"].transform(
                lambda s: s.quantile(valley_quantile)
            )
            thresh_high = df.groupby("date")["price"].transform(lambda s: s.quantile(q_med))
        else:
            thresh_med = df.groupby("date")["price"].transform(lambda s: s.quantile(q_med))
            thresh_ext = df.groupby("date")["price"].transform(lambda s: s.quantile(q_ext))
            tmed = thresh_med.values
            text = thresh_ext.values
    elif strategy == "rolling_quantile":
        roll = history.rolling(window=window_size, min_periods=min_periods)
        if bidirectional:
            thresh_low = roll.quantile(valley_quantile)
            thresh_high = roll.quantile(q_med)
        else:
            thresh_med = roll.quantile(q_med)
            thresh_ext = roll.quantile(q_ext)
            tmed = thresh_med.values
            text = thresh_ext.values
    elif strategy == "zscore":
        roll = history.rolling(window=window_size, min_periods=min_periods)
        mean_s = roll.mean()
        std_s = roll.std()
        if bidirectional:
            thresh_low = mean_s - zscore_threshold * std_s
            thresh_high = mean_s + zscore_threshold * std_s
        else:
            z_ext = max(zscore_threshold * 1.6, zscore_threshold + 0.5)
            thresh_med = mean_s + zscore_threshold * std_s
            thresh_ext = mean_s + z_ext * std_s
            tmed = thresh_med.values
            text = thresh_ext.values
    elif strategy == "vol_adj":
        roll = history.rolling(window=window_size, min_periods=min_periods)
        std_s = roll.std()
        baseline = y_series.shift(max(1, int(baseline_lag)))
        if bidirectional:
            thresh_low = baseline - vol_multiplier * std_s
            thresh_high = baseline + vol_multiplier * std_s
        else:
            vol_ext = max(vol_multiplier * 1.6, vol_multiplier + 0.5)
            thresh_med = baseline + vol_multiplier * std_s
            thresh_ext = baseline + vol_ext * std_s
            tmed = thresh_med.values
            text = thresh_ext.values
    elif strategy == "hybrid":
        roll = history.rolling(window=window_size, min_periods=min_periods)
        quant_med = roll.quantile(q_med)
        quant_ext = roll.quantile(q_ext)
        quant_low = roll.quantile(valley_quantile) if bidirectional else None
        mean_s = roll.mean()
        std_s = roll.std()
        z_ext = max(zscore_threshold * 1.6, zscore_threshold + 0.5)
        z_med = mean_s + zscore_threshold * std_s
        z_high = mean_s + z_ext * std_s
        z_low = mean_s - zscore_threshold * std_s if bidirectional else None
        baseline = y_series.shift(max(1, int(baseline_lag)))
        vol_ext = max(vol_multiplier * 1.6, vol_multiplier + 0.5)
        vol_med = baseline + vol_multiplier * std_s
        vol_high = baseline + vol_ext * std_s
        if bidirectional:
            vol_low = baseline - vol_multiplier * std_s
            thresh_low = pd.concat([quant_low, z_low, vol_low], axis=1).max(axis=1)
            thresh_high = pd.concat([quant_med, z_med, vol_med], axis=1).min(axis=1)
        else:
            thresh_med = pd.concat([quant_med, z_med, vol_med], axis=1).min(axis=1)
            thresh_ext = pd.concat([quant_ext, z_high, vol_high], axis=1).min(axis=1)
            tmed = thresh_med.values
            text = thresh_ext.values
    else:
        raise ValueError(f"Unknown regime strategy: {strategy}")

    if bidirectional:
        if isinstance(thresh_low, pd.Series) and isinstance(thresh_high, pd.Series):
            aligned = thresh_low.notna() & thresh_high.notna()
            low_vals = thresh_low[aligned].values
            high_vals = thresh_high[aligned].values
            thresh_low = thresh_low.copy()
            thresh_high = thresh_high.copy()
            thresh_low[aligned] = np.minimum(low_vals, high_vals)
            thresh_high[aligned] = np.maximum(low_vals, high_vals)
        t_low = thresh_low.values
        t_high = thresh_high.values
        valid_low = ~np.isnan(t_low)
        valid_high = ~np.isnan(t_high)
        labels[valid_low & (prices < t_low)] = 0
        labels[valid_high & (prices > t_high)] = 2
        return pd.Series(labels, index=y_series.index), thresh_low, thresh_high

    if isinstance(thresh_med, pd.Series) and isinstance(thresh_ext, pd.Series):
        monotonic_mask = thresh_med.notna() & thresh_ext.notna()
        thresh_ext = thresh_ext.copy()
        thresh_ext[monotonic_mask] = np.maximum(
            thresh_ext[monotonic_mask], thresh_med[monotonic_mask]
        )
        tmed = thresh_med.values
        text = thresh_ext.values

    valid_med = ~np.isnan(tmed)
    labels[valid_med & (prices > tmed)] = 1
    if classes > 2:
        valid_ext = ~np.isnan(text)
        labels[valid_ext & (prices > text)] = 2

    return pd.Series(labels, index=y_series.index), thresh_med, thresh_ext


def resolve_regime_label_names(regime_bidirectional: bool, regime_classes: int) -> list[str]:
    if regime_bidirectional:
        return ["Valley", "Normal", "Spike"]
    return ["Normal", "Spike"] if regime_classes == 2 else ["Normal", "Medium", "Extreme"]



def blend_predictions(spike_prob: np.ndarray, normal_pred: np.ndarray, spike_pred: np.ndarray) -> np.ndarray:
    return (1.0 - spike_prob) * normal_pred + spike_prob * spike_pred


def compute_regime_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spike_prob: np.ndarray,
    spike_threshold: float,
    prob_threshold: float,
    valley_threshold: np.ndarray | None = None,
    pred_labels: np.ndarray | None = None,
    bidirectional: bool = False,
) -> dict:
    def subset_metrics(name: str, mask: np.ndarray) -> dict:
        if mask.sum() == 0:
            return {
                f"mae_{name}": np.nan,
                f"smape_{name}": np.nan,
                f"r2_{name}": np.nan,
                f"count_{name}": int(mask.sum()),
            }
        mae = float(mean_absolute_error(y_true[mask], y_pred[mask]))
        smape_val = float(smape(y_true[mask], y_pred[mask]))
        r2_val = float(r2_score(y_true[mask], y_pred[mask])) if mask.sum() > 1 else np.nan
        return {
            f"mae_{name}": mae,
            f"smape_{name}": smape_val,
            f"r2_{name}": r2_val,
            f"count_{name}": int(mask.sum()),
        }

    threshold_high = np.asarray(spike_threshold, dtype=float)
    threshold_high = np.where(np.isnan(threshold_high), np.inf, threshold_high)
    true_spike = y_true > threshold_high

    if valley_threshold is not None:
        threshold_low = np.asarray(valley_threshold, dtype=float)
        threshold_low = np.where(np.isnan(threshold_low), -np.inf, threshold_low)
        true_valley = y_true < threshold_low
        true_normal = ~(true_spike | true_valley)
    else:
        true_valley = None
        true_normal = ~true_spike

    pred_spike = spike_prob > prob_threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_spike.astype(int),
        pred_spike.astype(int),
        average="binary",
        zero_division=0,
    )

    spike_rate = float(true_spike.mean())
    pred_spike_rate = float(pred_spike.mean())
    spike_prob_mean = float(np.mean(spike_prob))
    spike_prob_p90 = float(np.percentile(spike_prob, 90))
    spike_prob_p99 = float(np.percentile(spike_prob, 99))

    metrics = {
        "spike_rate": spike_rate,
        "pred_spike_rate": pred_spike_rate,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "spike_prob_mean": spike_prob_mean,
        "spike_prob_p90": spike_prob_p90,
        "spike_prob_p99": spike_prob_p99,
    }
    metrics.update(subset_metrics("spike", true_spike))
    metrics.update(subset_metrics("normal", true_normal))

    if true_valley is not None:
        metrics.update(subset_metrics("valley", true_valley))
        if pred_labels is not None and bidirectional:
            pred_valley = pred_labels == 0
            v_prec, v_rec, v_f1, _ = precision_recall_fscore_support(
                true_valley.astype(int),
                pred_valley.astype(int),
                average="binary",
                zero_division=0,
            )
            metrics["valley_precision"] = float(v_prec)
            metrics["valley_recall"] = float(v_rec)
            metrics["valley_f1"] = float(v_f1)
    return metrics


def train_regime_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    datetime_series: pd.Series | None,
    backend: str,
    model_params: dict,
    target_transform: str,
    weight_quantile: float,
    weight_multiplier: float,
    regime_quantile: float,
    regime_extreme_quantile: float,
    min_spike_samples: int,
    regime_spike_weight: float,
    regime_strategy: str,
    regime_classes: int,
    rolling_window_days: int,
    regime_zscore_threshold: float,
    regime_vol_multiplier: float,
    regime_baseline_lag: int,
    regime_bidirectional: bool,
    regime_valley_quantile: float | None,
) -> tuple[Pipeline, object, object, object, pd.Series, pd.Series, bool]:
    regime_labels, thresh_low, thresh_high = compute_dynamic_labels(
        y_train,
        datetime_series=datetime_series,
        strategy=regime_strategy,
        q_med=regime_quantile,
        q_ext=regime_extreme_quantile,
        classes=regime_classes,
        rolling_window_days=rolling_window_days,
        zscore_threshold=regime_zscore_threshold,
        vol_multiplier=regime_vol_multiplier,
        baseline_lag=regime_baseline_lag,
        bidirectional=regime_bidirectional,
        valley_quantile=regime_valley_quantile,
    )

    if pd.Series(regime_labels).nunique() < 2:
        classifier = DummyClassifier(strategy="most_frequent")
        classifier.fit(X_train, regime_labels)
    else:
        classifier = build_regime_classifier()
        classifier.fit(X_train, regime_labels)

    low_mask = regime_labels == 0
    mid_mask = regime_labels == 1
    high_mask = regime_labels == 2

    low_model = build_gbm_model(backend, model_params)
    mid_model = build_gbm_model(backend, model_params)
    high_model = build_gbm_model(backend, model_params)
    global_model = build_gbm_model(backend, model_params)

    def fit_subset(
        model: object,
        X_sub: pd.DataFrame,
        y_sub: pd.Series,
        weights_override: np.ndarray | None = None,
    ) -> None:
        if len(y_sub) == 0:
            return
        y_trans = pd.Series(
            transform_target(y_sub.values, target_transform),
            index=y_sub.index,
        )
        if weights_override is None:
            weights = build_sample_weights_configured(
                y_sub.values,
                quantile=weight_quantile,
                high_weight=weight_multiplier,
            )
        else:
            weights = weights_override
        fit_with_weights(model, X_sub, y_trans, weights)

    global_weights = build_sample_weights_configured(
        y_train.values,
        quantile=weight_quantile,
        high_weight=weight_multiplier,
    )
    fit_subset(global_model, X_train, y_train, global_weights)

    spike_fallback = False
    if regime_bidirectional:
        if mid_mask.sum() < min_spike_samples:
            mid_model = copy.deepcopy(global_model)
        else:
            fit_subset(mid_model, X_train[mid_mask], y_train[mid_mask])

        if low_mask.sum() < min_spike_samples:
            low_model = copy.deepcopy(mid_model)
            spike_fallback = True
        else:
            low_weights = np.where(low_mask, regime_spike_weight, 1.0)
            fit_subset(low_model, X_train, y_train, low_weights)

        if high_mask.sum() < min_spike_samples:
            high_model = copy.deepcopy(mid_model)
            spike_fallback = True
        else:
            high_weights = np.where(high_mask, regime_spike_weight, 1.0)
            fit_subset(high_model, X_train, y_train, high_weights)
    else:
        if low_mask.sum() < min_spike_samples:
            low_model = copy.deepcopy(global_model)
        else:
            fit_subset(low_model, X_train[low_mask], y_train[low_mask])

        if mid_mask.sum() < min_spike_samples:
            mid_model = copy.deepcopy(low_model)
            spike_fallback = True
        else:
            mid_weights = np.where(mid_mask, regime_spike_weight, 1.0)
            fit_subset(mid_model, X_train, y_train, mid_weights)

        if high_mask.sum() < min_spike_samples:
            high_model = copy.deepcopy(mid_model)
        else:
            high_weights = np.where(high_mask, regime_spike_weight * 2.0, 1.0)
            fit_subset(high_model, X_train, y_train, high_weights)

    return (
        classifier,
        low_model,
        mid_model,
        high_model,
        thresh_low,
        thresh_high,
        spike_fallback,
    )


def predict_regime(
    X: pd.DataFrame,
    classifier: object,
    low_model: object,
    mid_model: object,
    high_model: object,
    prob_threshold: float,
    target_transform: str,
    regime_bidirectional: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = classifier.predict_proba(X)
    classes = getattr(classifier, "classes_", np.array([0]))

    pred_low = inverse_transform_target(low_model.predict(X), target_transform)
    pred_mid = inverse_transform_target(mid_model.predict(X), target_transform)
    pred_high = inverse_transform_target(high_model.predict(X), target_transform)

    def class_prob(target: int) -> np.ndarray:
        if probs.shape[1] == 1:
            if target in classes and classes[0] == target:
                return np.ones_like(pred_low)
            return np.zeros_like(pred_low)
        if target not in classes:
            return np.zeros_like(pred_low)
        idx = int(np.where(classes == target)[0][0])
        return probs[:, idx]

    if regime_bidirectional:
        low_prob = class_prob(0)
        mid_prob = class_prob(1)
        spike_prob = class_prob(2)
        preds = low_prob * pred_low + mid_prob * pred_mid + spike_prob * pred_high
    else:
        normal_prob = class_prob(0)
        med_prob = class_prob(1)
        ext_prob = class_prob(2)
        spike_prob = med_prob + ext_prob
        preds = normal_prob * pred_low + med_prob * pred_mid + ext_prob * pred_high

    use_spike = spike_prob > prob_threshold
    return preds, spike_prob, use_spike


def time_series_cv_metrics(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    model_name: str,
    target_transform: str,
    weight_quantile: float,
    weight_multiplier: float,
    debug_fold: int | None = None,
) -> pd.DataFrame:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        model_clone = clone(model)
        y_train_raw = y.iloc[train_idx]
        y_val_raw = y.iloc[val_idx]
        weights = build_sample_weights_configured(
            y_train_raw.values,
            quantile=weight_quantile,
            high_weight=weight_multiplier,
        )
        y_train_trans = pd.Series(
            transform_target(y_train_raw.values, target_transform),
            index=y_train_raw.index,
        )
        fit_with_weights(model_clone, X.iloc[train_idx], y_train_trans, weights)
        preds_trans = model_clone.predict(X.iloc[val_idx])
        preds = inverse_transform_target(preds_trans, target_transform)
        metrics = evaluate_predictions(y_val_raw.values, preds)
        metrics["fold"] = fold
        metrics["model"] = model_name
        rows.append(metrics)

        if debug_fold is not None and fold == debug_fold:
            print("\nFold debug:")
            print(y_train_raw.describe())
            print("Weights summary:")
            print(pd.Series(weights).describe())
    return pd.DataFrame(rows)


def time_series_cv_regime_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    backend: str,
    model_params: dict,
    n_splits: int,
    target_transform: str,
    weight_quantile: float,
    weight_multiplier: float,
    regime_quantile: float,
    regime_extreme_quantile: float,
    regime_prob_threshold: float,
    regime_min_samples: int,
    regime_spike_weight: float,
    regime_strategy: str,
    regime_classes: int,
    rolling_window_days: int,
    regime_zscore_threshold: float,
    regime_vol_multiplier: float,
    regime_baseline_lag: int,
    regime_bidirectional: bool,
    regime_valley_quantile: float | None,
    debug_fold: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    diag_rows = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        classifier, low_model, mid_model, high_model, thresh_low, thresh_high, spike_fallback = train_regime_models(
            X_train,
            y_train,
            timestamps.iloc[train_idx],
            backend,
            model_params,
            target_transform,
            weight_quantile,
            weight_multiplier,
            regime_quantile,
            regime_extreme_quantile,
            regime_min_samples,
            regime_spike_weight,
            regime_strategy,
            regime_classes,
            rolling_window_days,
            regime_zscore_threshold,
            regime_vol_multiplier,
            regime_baseline_lag,
            regime_bidirectional,
            regime_valley_quantile,
        )

        preds, spike_prob, use_spike = predict_regime(
            X_val,
            classifier,
            low_model,
            mid_model,
            high_model,
            regime_prob_threshold,
            target_transform,
            regime_bidirectional,
        )
        metrics = evaluate_predictions(y_val.values, preds)
        metrics["fold"] = fold
        metrics["model"] = "Regime"
        rows.append(metrics)

        _, thresh_low_val, thresh_high_val = compute_dynamic_labels(
            y_val,
            datetime_series=timestamps.iloc[val_idx],
            strategy=regime_strategy,
            q_med=regime_quantile,
            q_ext=regime_extreme_quantile,
            classes=regime_classes,
            rolling_window_days=rolling_window_days,
            zscore_threshold=regime_zscore_threshold,
            vol_multiplier=regime_vol_multiplier,
            baseline_lag=regime_baseline_lag,
            bidirectional=regime_bidirectional,
            valley_quantile=regime_valley_quantile,
        )
        pred_labels = classifier.predict(X_val)
        diag = compute_regime_diagnostics(
            y_val.values,
            preds,
            spike_prob,
            thresh_high_val.values,
            regime_prob_threshold,
            valley_threshold=thresh_low_val.values if regime_bidirectional else None,
            pred_labels=pred_labels,
            bidirectional=regime_bidirectional,
        )
        diag["fold"] = fold
        diag["spike_fallback"] = spike_fallback
        diag_rows.append(diag)

        if debug_fold is not None and fold == debug_fold:
            print("\nRegime fold debug:")
            print(y_train.describe())
            print("Spike prob summary:")
            print(pd.Series(spike_prob).describe())
            print("Spike model usage:", int(use_spike.sum()))

    return pd.DataFrame(rows), pd.DataFrame(diag_rows)


def tune_gbm_model(
    backend: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    target_transform: str,
    weight_quantile: float,
    weight_multiplier: float,
) -> dict:
    if backend == "lightgbm":
        grid = {
            "n_estimators": [200, 400],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [31, 63],
        }
    else:
        grid = {
            "n_estimators": [200, 400],
            "learning_rate": [0.05, 0.1],
            "max_depth": [4, 6],
        }

    best_params = {}
    best_score = float("inf")

    for params in ParameterGrid(grid):
        model = build_gbm_model(backend, params)
        cv_df = time_series_cv_metrics(
            model,
            X,
            y,
            n_splits,
            "GBM",
            target_transform,
            weight_quantile,
            weight_multiplier,
            debug_fold=None,
        )
        mae_mean = cv_df["mae"].mean()
        if mae_mean < best_score:
            best_score = mae_mean
            best_params = params

    return best_params


def train_quantile_models(
    backend: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    quantiles: list,
    target_transform: str,
    weight_quantile: float,
    weight_multiplier: float,
) -> dict:
    models = {}
    weights = build_sample_weights_configured(
        y_train.values,
        quantile=weight_quantile,
        high_weight=weight_multiplier,
    )
    y_train_trans = pd.Series(
        transform_target(y_train.values, target_transform),
        index=y_train.index,
    )
    for q in quantiles:
        model = build_quantile_model(backend, q)
        fit_with_weights(model, X_train, y_train_trans, weights)
        models[q] = model
    return models


def _get_package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def save_deployment_artifacts(
    config: Config,
    feature_cols: list[str],
    target_col: str,
    market_columns: dict[str, str],
    gbm_backend: str,
    gbm_params: dict,
    quantile_backend: str,
    huber_model: object,
    gbm_model: object,
    classifier: object,
    low_model: object,
    mid_model: object,
    high_model: object,
    quantile_models: dict,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    deploy_dir = repo_root / "deploy"
    models_dir = deploy_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_files = {
        "regime_classifier": "regime_classifier.joblib",
        "regime_low": "regime_low_model.joblib",
        "regime_mid": "regime_mid_model.joblib",
        "regime_high": "regime_high_model.joblib",
        "p10": "p10_model.joblib",
        "p50": "p50_model.joblib",
        "p90": "p90_model.joblib",
        "huber": "huber_model.joblib",
        "gbm": "gbm_model.joblib",
    }

    joblib.dump(classifier, models_dir / model_files["regime_classifier"])
    joblib.dump(low_model, models_dir / model_files["regime_low"])
    joblib.dump(mid_model, models_dir / model_files["regime_mid"])
    joblib.dump(high_model, models_dir / model_files["regime_high"])
    joblib.dump(quantile_models[0.1], models_dir / model_files["p10"])
    joblib.dump(quantile_models[0.5], models_dir / model_files["p50"])
    joblib.dump(quantile_models[0.9], models_dir / model_files["p90"])
    joblib.dump(huber_model, models_dir / model_files["huber"])
    joblib.dump(gbm_model, models_dir / model_files["gbm"])

    feature_payload = {
        "feature_list": feature_cols,
        "target_column": target_col,
        "market_name": config.target_market,
        "datetime_column": config.datetime_col,
    }
    deploy_dir.mkdir(parents=True, exist_ok=True)
    (deploy_dir / "feature_list.json").write_text(
        json.dumps(feature_payload, indent=2),
        encoding="utf-8",
    )

    required_raw_columns = _dedupe_keep_order(
        [
            config.datetime_col,
            target_col,
            config.buy_col,
            config.sell_col,
            config.solar_col,
            *config.renewable_cols,
            *config.weather_cols,
            *list(market_columns.values()),
        ]
    )

    model_metadata = {
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": {
            "regime_classifier": type(classifier).__name__,
            "regime_low": type(low_model).__name__,
            "regime_mid": type(mid_model).__name__,
            "regime_high": type(high_model).__name__,
            "p10": type(quantile_models[0.1]).__name__,
            "p50": type(quantile_models[0.5]).__name__,
            "p90": type(quantile_models[0.9]).__name__,
            "huber": type(huber_model).__name__,
            "gbm": type(gbm_model).__name__,
        },
        "selected_hyperparameters": {
            "gbm_backend": gbm_backend,
            "gbm_params": gbm_params,
            "quantile_backend": quantile_backend,
            "regime_strategy": config.regime_strategy,
            "regime_classes": config.regime_classes,
            "target_transform": config.target_transform,
            "weight_quantile": config.weight_quantile,
            "weight_multiplier": config.weight_multiplier,
            "regime_prob_threshold": config.regime_prob_threshold,
        },
        "library_versions": {
            "scikit_learn": _get_package_version("scikit-learn"),
            "xgboost": _get_package_version("xgboost"),
            "lightgbm": _get_package_version("lightgbm"),
        },
        "feature_config": {
            "target_market": config.target_market,
            "datetime_col": config.datetime_col,
            "target_col": target_col,
            "market_columns": market_columns,
            "lead_steps": config.lead_steps,
            "target_lags": config.target_lags,
            "rolling_windows": config.rolling_windows,
            "cross_market_lags": config.cross_market_lags,
            "renewable_cols": config.renewable_cols,
            "buy_col": config.buy_col,
            "sell_col": config.sell_col,
            "solar_col": config.solar_col,
            "weather_cols": config.weather_cols,
            "auction_cutoff_hour": config.auction_cutoff_hour,
            "auction_cutoff_minute": config.auction_cutoff_minute,
            "target_transform": config.target_transform,
            "regime_prob_threshold": config.regime_prob_threshold,
            "regime_bidirectional": config.regime_bidirectional,
            "regime_classes": config.regime_classes,
        },
        "required_raw_columns": required_raw_columns,
        "model_files": model_files,
        "feature_count": len(feature_cols),
    }
    (deploy_dir / "model_metadata.json").write_text(
        json.dumps(model_metadata, indent=2),
        encoding="utf-8",
    )


def train_spike_hybrid(
    X_train: pd.DataFrame,
    train_df: pd.DataFrame,
    X_test: pd.DataFrame,
    base_preds: np.ndarray,
    target_col: str,
    spike_baseline_lag: int,
    threshold: float,
    adjustment: float,
) -> tuple[np.ndarray, Pipeline]:
    baseline_col = f"price_lag_{spike_baseline_lag}"
    if baseline_col not in train_df.columns:
        raise ValueError(
            f"Missing baseline lag feature '{baseline_col}' for spike hybrid."
        )
    spike_label = (
        train_df[target_col] > train_df[baseline_col] + threshold
    ).astype(int)

    clf = build_spike_classifier()
    clf.fit(X_train, spike_label)
    spike_prob = clf.predict_proba(X_test)[:, 1]

    hybrid_preds = base_preds * (1 + spike_prob * adjustment)
    return hybrid_preds, clf


def plot_results(
    output_dir: Path,
    test_times: pd.Series,
    y_true: np.ndarray,
    preds: dict,
    p10: np.ndarray | None,
    p50: np.ndarray | None,
    p90: np.ndarray | None,
    plot_points: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = min(plot_points, len(y_true))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_times.iloc[:n], y_true[:n], label="Actual", color="black", linewidth=2)
    for name, series in preds.items():
        ax.plot(test_times.iloc[:n], series[:n], label=name)
    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_pred.png")
    plt.close(fig)

    if "GBM" in preds:
        resid = y_true - preds["GBM"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(preds["GBM"], resid, alpha=0.3, s=12)
        ax.axhline(0, linestyle="--", color="black")
        ax.set_title("Residuals vs Prediction (GBM)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        fig.tight_layout()
        fig.savefig(output_dir / "residuals.png")
        plt.close(fig)

    if p10 is not None and p50 is not None and p90 is not None:
        lower = np.minimum(p10, p90)
        upper = np.maximum(p10, p90)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(test_times.iloc[:n], y_true[:n], label="Actual", color="black", linewidth=2)
        ax.plot(test_times.iloc[:n], p50[:n], label="P50", linestyle="--")
        ax.fill_between(test_times.iloc[:n], lower[:n], upper[:n], alpha=0.2, label="P10-P90")
        ax.set_title("Quantile Bands")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_dir / "quantile_bands.png")
        plt.close(fig)


def run_regime_strategy_comparison(
    output_dir: Path,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_times: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_times: pd.Series,
    backend: str,
    model_params: dict,
    target_transform: str,
    weight_quantile: float,
    weight_multiplier: float,
    regime_quantile: float,
    regime_extreme_quantile: float,
    regime_min_samples: int,
    regime_spike_weight: float,
    regime_classes: int,
    rolling_window_days: int,
    regime_zscore_threshold: float,
    regime_vol_multiplier: float,
    regime_baseline_lag: int,
    regime_bidirectional: bool,
    regime_valley_quantile: float | None,
) -> pd.DataFrame:
    strategies = ["rolling_quantile", "zscore", "vol_adj", "hybrid"]
    label_names = resolve_regime_label_names(regime_bidirectional, regime_classes)
    rows = []

    for strategy in strategies:
        (
            clf,
            low_model,
            mid_model,
            high_model,
            _,
            _,
            fallback,
        ) = train_regime_models(
            X_train,
            y_train,
            train_times,
            backend,
            model_params,
            target_transform,
            weight_quantile,
            weight_multiplier,
            regime_quantile,
            regime_extreme_quantile,
            regime_min_samples,
            regime_spike_weight,
            strategy,
            regime_classes,
            rolling_window_days,
            regime_zscore_threshold,
            regime_vol_multiplier,
            regime_baseline_lag,
            regime_bidirectional,
            regime_valley_quantile,
        )
        preds, spike_prob, _ = predict_regime(
            X_test,
            clf,
            low_model,
            mid_model,
            high_model,
            prob_threshold=0.0,
            target_transform=target_transform,
            regime_bidirectional=regime_bidirectional,
        )
        true_labels, thresh_low, thresh_high = compute_dynamic_labels(
            y_test,
            datetime_series=test_times,
            strategy=strategy,
            q_med=regime_quantile,
            q_ext=regime_extreme_quantile,
            classes=regime_classes,
            rolling_window_days=rolling_window_days,
            zscore_threshold=regime_zscore_threshold,
            vol_multiplier=regime_vol_multiplier,
            baseline_lag=regime_baseline_lag,
            bidirectional=regime_bidirectional,
            valley_quantile=regime_valley_quantile,
        )
        pred_labels = clf.predict(X_test)

        metrics = evaluate_predictions(y_test.values, preds)
        if regime_bidirectional:
            spike_rate = float((true_labels.values == 2).mean())
            pred_spike_rate = float((pred_labels == 2).mean())
            valley_rate = float((true_labels.values == 0).mean())
            pred_valley_rate = float((pred_labels == 0).mean())
            metrics.update(
                {
                    "strategy": strategy,
                    "spike_rate": spike_rate,
                    "pred_spike_rate": pred_spike_rate,
                    "valley_rate": valley_rate,
                    "pred_valley_rate": pred_valley_rate,
                    "spike_prob_mean": float(np.mean(spike_prob)),
                    "spike_fallback": bool(fallback),
                }
            )
        else:
            metrics.update(
                {
                    "strategy": strategy,
                    "spike_rate": float((true_labels.values > 0).mean()),
                    "pred_spike_rate": float((pred_labels > 0).mean()),
                    "extreme_rate": float((true_labels.values == 2).mean())
                    if regime_classes > 2
                    else 0.0,
                    "pred_extreme_rate": float((pred_labels == 2).mean())
                    if regime_classes > 2
                    else 0.0,
                    "spike_prob_mean": float(np.mean(spike_prob)),
                    "spike_fallback": bool(fallback),
                }
            )
        rows.append(metrics)

        strat_dir = output_dir / f"regime_strategy_{strategy}"
        plot_regime_confusion_matrix(
            strat_dir,
            true_labels.values,
            pred_labels,
            label_names=label_names,
            filename="confusion_matrix.png",
        )
        plot_regime_distribution_over_time(
            strat_dir,
            test_times,
            true_labels.values,
            label_names=label_names,
            filename="distribution_over_time.png",
        )
        plot_spike_timeline(
            strat_dir,
            test_times,
            y_test.values,
            true_labels.values,
            thresh_low.values,
            thresh_high.values,
            label_names=label_names,
            filename="spike_timeline_true.png",
        )
        plot_spike_timeline(
            strat_dir,
            test_times,
            y_test.values,
            pred_labels,
            thresh_low.values,
            thresh_high.values,
            label_names=label_names,
            filename="spike_timeline_pred.png",
        )

    return pd.DataFrame(rows)


def run_pipeline(config: Config) -> None:
    df = pd.read_csv(config.input_csv)
    df_raw = df.copy()
    market_policy = resolve_market_policy(config.target_market)
    market_columns = resolve_market_columns(
        df.columns.tolist(),
        config.target_market,
        config.target_col,
        config.market_columns,
    )
    target_col = market_columns[config.target_market]
    df, feature_cols, feature_groups, validation_df = feature_engineering(
        df,
        datetime_col=config.datetime_col,
        target_market=config.target_market,
        target_col=target_col,
        market_columns=market_columns,
        lead_steps=config.lead_steps,
        target_lags=config.target_lags,
        rolling_windows=config.rolling_windows,
        cross_market_lags=config.cross_market_lags,
        renewable_cols=config.renewable_cols,
        buy_col=config.buy_col,
        sell_col=config.sell_col,
        solar_col=config.solar_col,
        weather_cols=config.weather_cols,
        market_policy=market_policy,
        auction_cutoff_hour=config.auction_cutoff_hour,
        auction_cutoff_minute=config.auction_cutoff_minute,
    )

    if getattr(config, "feature_include", None):
        selected_features = set(config.feature_include)
        missing_features = [feature for feature in config.feature_include if feature not in feature_cols]
        if missing_features:
            raise ValueError(
                f"Requested feature(s) not found in engineered output: {', '.join(missing_features)}"
            )
        feature_cols = [feature for feature in feature_cols if feature in selected_features]

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "processed_dataset.csv", index=False)
    audit_tables = build_cutoff_audit_tables(
        df_before=df_raw,
        df_after=df,
        feature_cols=feature_cols,
        datetime_col=config.datetime_col,
    )

    train_df, test_df = time_train_test_split(df, config.test_size)

    feature_catalog = pd.DataFrame(
        [
            {
                "feature": name,
                "group": feature_groups.get(name, FEATURE_GROUP_TARGET_LAG),
                "allowed": name in feature_cols,
            }
            for name in sorted(feature_groups.keys())
        ]
    )
    feature_catalog.to_csv(output_dir / "feature_catalog.csv", index=False)
    validation_df.to_csv(output_dir / "feature_validation.csv", index=False)
    audit_tables["summary"].to_csv(output_dir / "cutoff_audit_summary.csv", index=False)
    audit_tables["retained_rows_by_block"].to_csv(output_dir / "retained_rows_by_block.csv", index=False)
    audit_tables["feature_availability"].to_csv(output_dir / "feature_availability_after_masking.csv", index=False)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    validate_classifier_nan_robustness(
        X_train,
        y_train,
        target_market=config.target_market,
        output_dir=output_dir,
    )

    huber_model = build_huber_model()
    cv_huber = time_series_cv_metrics(
        huber_model,
        X_train,
        y_train,
        config.n_splits,
        "Huber",
        config.target_transform,
        config.weight_quantile,
        config.weight_multiplier,
        config.debug_fold,
    )

    gbm_backend = resolve_backend(config.gbm_backend)
    quantile_backend = resolve_quantile_backend(config.quantile_backend)
    gbm_params = (
        tune_gbm_model(
            gbm_backend,
            X_train,
            y_train,
            config.n_splits,
            config.target_transform,
            config.weight_quantile,
            config.weight_multiplier,
        )
        if config.tune_gbm
        else {}
    )
    gbm_model = build_gbm_model(gbm_backend, gbm_params)
    gbm_model_unweighted = build_gbm_model(gbm_backend, gbm_params)
    cv_gbm = time_series_cv_metrics(
        gbm_model,
        X_train,
        y_train,
        config.n_splits,
        "GBM",
        config.target_transform,
        config.weight_quantile,
        config.weight_multiplier,
        config.debug_fold,
    )

    cv_p50 = time_series_cv_metrics(
        build_quantile_model(quantile_backend, 0.5),
        X_train,
        y_train,
        config.n_splits,
        "P50",
        config.target_transform,
        config.weight_quantile,
        config.weight_multiplier,
        config.debug_fold,
    )

    cv_p10 = time_series_cv_metrics(
        build_quantile_model(quantile_backend, 0.1),
        X_train,
        y_train,
        config.n_splits,
        "P10",
        config.target_transform,
        config.weight_quantile,
        config.weight_multiplier,
        config.debug_fold,
    )

    cv_regime, regime_diag_cv = time_series_cv_regime_metrics(
        X_train,
        y_train,
        train_df[config.datetime_col],
        gbm_backend,
        gbm_params,
        config.n_splits,
        config.target_transform,
        config.weight_quantile,
        config.weight_multiplier,
        config.regime_quantile,
        config.regime_extreme_quantile,
        config.regime_prob_threshold,
        config.regime_min_samples,
        config.regime_spike_weight,
        config.regime_strategy,
        config.regime_classes,
        config.rolling_window_days,
        config.regime_zscore_threshold,
        config.regime_vol_multiplier,
        config.regime_baseline_lag,
        config.regime_bidirectional,
        config.regime_valley_quantile,
        config.debug_fold,
    )

    weights = build_sample_weights_configured(
        y_train.values,
        quantile=config.weight_quantile,
        high_weight=config.weight_multiplier,
    )
    y_train_trans = pd.Series(
        transform_target(y_train.values, config.target_transform),
        index=y_train.index,
    )
    fit_with_weights(huber_model, X_train, y_train_trans, weights)
    fit_with_weights(gbm_model, X_train, y_train_trans, weights)
    gbm_model_unweighted.fit(X_train, y_train_trans)

    preds_huber = inverse_transform_target(huber_model.predict(X_test), config.target_transform)
    preds_gbm = inverse_transform_target(gbm_model.predict(X_test), config.target_transform)
    preds_gbm_unweighted = inverse_transform_target(
        gbm_model_unweighted.predict(X_test),
        config.target_transform,
    )

    preds = {
        "Huber": preds_huber,
        "GBM": preds_gbm,
        "GBM_Unweighted": preds_gbm_unweighted,
    }

    hybrid_preds = None
    if config.enable_spike_hybrid:
        hybrid_preds, _ = train_spike_hybrid(
            X_train,
            train_df,
            X_test,
            preds_gbm,
            target_col,
            config.spike_baseline_lag,
            config.spike_threshold,
            config.spike_adjustment,
        )
        preds["Hybrid"] = hybrid_preds

    (
        classifier,
        low_model,
        mid_model,
        high_model,
        _,
        _,
        regime_fallback,
    ) = train_regime_models(
        X_train,
        y_train,
        train_df[config.datetime_col],
        gbm_backend,
        gbm_params,
        config.target_transform,
        config.weight_quantile,
        config.weight_multiplier,
        config.regime_quantile,
        config.regime_extreme_quantile,
        config.regime_min_samples,
        config.regime_spike_weight,
        config.regime_strategy,
        config.regime_classes,
        config.rolling_window_days,
        config.regime_zscore_threshold,
        config.regime_vol_multiplier,
        config.regime_baseline_lag,
        config.regime_bidirectional,
        config.regime_valley_quantile,
    )
    regime_preds, regime_prob, regime_use_spike = predict_regime(
        X_test,
        classifier,
        low_model,
        mid_model,
        high_model,
        config.regime_prob_threshold,
        config.target_transform,
        config.regime_bidirectional,
    )
    preds["Regime"] = regime_preds
    pred_low = inverse_transform_target(low_model.predict(X_test), config.target_transform)
    pred_mid = inverse_transform_target(mid_model.predict(X_test), config.target_transform)
    pred_high = inverse_transform_target(high_model.predict(X_test), config.target_transform)
    regime_label_pred = classifier.predict(X_test)
    regime_hard_preds = np.select(
        [regime_label_pred == 0, regime_label_pred == 1, regime_label_pred == 2],
        [pred_low, pred_mid, pred_high],
        default=pred_low,
    )
    preds["Regime_Hard"] = regime_hard_preds
    regime_label_true, regime_thresh_low, regime_thresh_high = compute_dynamic_labels(
        y_test,
        datetime_series=test_df[config.datetime_col],
        strategy=config.regime_strategy,
        q_med=config.regime_quantile,
        q_ext=config.regime_extreme_quantile,
        classes=config.regime_classes,
        rolling_window_days=config.rolling_window_days,
        zscore_threshold=config.regime_zscore_threshold,
        vol_multiplier=config.regime_vol_multiplier,
        baseline_lag=config.regime_baseline_lag,
        bidirectional=config.regime_bidirectional,
        valley_quantile=config.regime_valley_quantile,
    )
    regime_holdout_diag = compute_regime_diagnostics(
        y_test.values,
        regime_preds,
        regime_prob,
        regime_thresh_high.values,
        config.regime_prob_threshold,
        valley_threshold=regime_thresh_low.values if config.regime_bidirectional else None,
        pred_labels=regime_label_pred,
        bidirectional=config.regime_bidirectional,
    )
    label_names = resolve_regime_label_names(config.regime_bidirectional, config.regime_classes)
    plot_regime_confusion_matrix(
        output_dir,
        regime_label_true.values,
        regime_label_pred,
        label_names=label_names,
        filename=f"regime_confusion_matrix_{config.regime_strategy}.png",
    )
    plot_regime_distribution_over_time(
        output_dir,
        test_df[config.datetime_col],
        regime_label_true.values,
        label_names=label_names,
        filename=f"regime_distribution_{config.regime_strategy}.png",
    )
    plot_spike_timeline(
        output_dir,
        test_df[config.datetime_col],
        y_test.values,
        regime_label_true.values,
        regime_thresh_low.values,
        regime_thresh_high.values,
        label_names=label_names,
        filename=f"spike_timeline_true_{config.regime_strategy}.png",
    )
    plot_spike_timeline(
        output_dir,
        test_df[config.datetime_col],
        y_test.values,
        regime_label_pred,
        regime_thresh_low.values,
        regime_thresh_high.values,
        label_names=label_names,
        filename=f"spike_timeline_pred_{config.regime_strategy}.png",
    )

    quantile_models = train_quantile_models(
        quantile_backend,
        X_train,
        y_train,
        [0.1, 0.5, 0.9],
        config.target_transform,
        config.weight_quantile,
        config.weight_multiplier,
    )
    p10 = inverse_transform_target(quantile_models[0.1].predict(X_test), config.target_transform)
    p50 = inverse_transform_target(quantile_models[0.5].predict(X_test), config.target_transform)
    p90 = inverse_transform_target(quantile_models[0.9].predict(X_test), config.target_transform)
    preds["P10"] = p10
    preds["P50"] = p50

    model_snapshot_dir = output_dir / "trained_models"
    model_snapshot_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(huber_model, model_snapshot_dir / "huber_model.joblib")
    joblib.dump(gbm_model, model_snapshot_dir / "gbm_model.joblib")
    joblib.dump(gbm_model_unweighted, model_snapshot_dir / "gbm_unweighted_model.joblib")
    joblib.dump(classifier, model_snapshot_dir / "regime_classifier.joblib")
    joblib.dump(low_model, model_snapshot_dir / "regime_low_model.joblib")
    joblib.dump(mid_model, model_snapshot_dir / "regime_mid_model.joblib")
    joblib.dump(high_model, model_snapshot_dir / "regime_high_model.joblib")
    joblib.dump(quantile_models[0.1], model_snapshot_dir / "p10_model.joblib")
    joblib.dump(quantile_models[0.5], model_snapshot_dir / "p50_model.joblib")
    joblib.dump(quantile_models[0.9], model_snapshot_dir / "p90_model.joblib")

    holdout_rows = []
    for name, series in preds.items():
        metrics = evaluate_predictions(y_test.values, series)
        metrics["model"] = name
        metrics["dataset"] = "holdout"
        holdout_rows.append(metrics)

    cv_summary = []
    for name, cv_df in [
        ("Huber", cv_huber),
        ("GBM", cv_gbm),
        ("P10", cv_p10),
        ("P50", cv_p50),
        ("Regime", cv_regime),
    ]:
        cv_summary.append(
            {
                "model": name,
                "dataset": "cv_mean",
                "mae": cv_df["mae"].mean(),
                "r2": cv_df["r2"].mean(),
                "smape": cv_df["smape"].mean(),
                "mae_std": cv_df["mae"].std(),
                "r2_std": cv_df["r2"].std(),
                "smape_std": cv_df["smape"].std(),
            }
        )

    comparison_table = pd.DataFrame(cv_summary + holdout_rows)

    cv_huber.to_csv(output_dir / "cv_metrics_huber.csv", index=False)
    cv_gbm.to_csv(output_dir / "cv_metrics_gbm.csv", index=False)
    cv_p10.to_csv(output_dir / "cv_metrics_p10.csv", index=False)
    cv_p50.to_csv(output_dir / "cv_metrics_p50.csv", index=False)
    cv_regime.to_csv(output_dir / "cv_metrics_regime.csv", index=False)
    comparison_table.to_csv(output_dir / "comparison_table.csv", index=False)
    if regime_diag_cv is not None:
        regime_diag_cv.to_csv(output_dir / "regime_diagnostics_cv.csv", index=False)
    pd.DataFrame([regime_holdout_diag]).to_csv(
        output_dir / "regime_diagnostics_holdout.csv",
        index=False,
    )

    # Bucketed evaluation on holdout
    q90 = float(np.percentile(y_test.values, 90))
    q95 = float(np.percentile(y_test.values, 95))
    bucket_rows = []
    for name, series in preds.items():
        rows = evaluate_price_buckets(y_test.values, series, q90, q95)
        for row in rows:
            row["model"] = name
            row["dataset"] = "holdout"
            bucket_rows.append(row)
    pd.DataFrame(bucket_rows).to_csv(output_dir / "price_bucket_metrics.csv", index=False)

    # Regime threshold sweep on holdout
    sweep_rows = []
    sweep_diag_rows = []
    for rq in [0.85, 0.9, 0.95]:
        if not config.regime_bidirectional and config.regime_extreme_quantile <= rq:
            raise ValueError(
                "regime_extreme_quantile must be greater than regime_quantile in sweep"
            )
        sweep_valley_quantile = (
            config.regime_valley_quantile
            if config.regime_valley_quantile is not None
            else 1.0 - rq
        )
        (
            clf_s,
            low_s,
            mid_s,
            high_s,
            thresh_low_s,
            _,
            fallback_s,
        ) = train_regime_models(
            X_train,
            y_train,
            train_df[config.datetime_col],
            gbm_backend,
            gbm_params,
            config.target_transform,
            config.weight_quantile,
            config.weight_multiplier,
            rq,
            config.regime_extreme_quantile,
            config.regime_min_samples,
            config.regime_spike_weight,
            config.regime_strategy,
            config.regime_classes,
            config.rolling_window_days,
            config.regime_zscore_threshold,
            config.regime_vol_multiplier,
            config.regime_baseline_lag,
            config.regime_bidirectional,
            sweep_valley_quantile,
        )
        preds_s, prob_s, _ = predict_regime(
            X_test,
            clf_s,
            low_s,
            mid_s,
            high_s,
            config.regime_prob_threshold,
            config.target_transform,
            config.regime_bidirectional,
        )
        metrics = evaluate_predictions(y_test.values, preds_s)
        metrics.update(
            {
                "regime_quantile": rq,
                "regime_strategy": config.regime_strategy,
                "model": "Regime_Soft",
            }
        )
        sweep_rows.append(metrics)
        # compute thresholds aligned to the holdout (test) period before diagnostics
        _, thresh_low_test, thresh_high_test = compute_dynamic_labels(
            y_test,
            datetime_series=test_df[config.datetime_col],
            strategy=config.regime_strategy,
            q_med=rq,
            q_ext=config.regime_extreme_quantile,
            classes=config.regime_classes,
            rolling_window_days=config.rolling_window_days,
            zscore_threshold=config.regime_zscore_threshold,
            vol_multiplier=config.regime_vol_multiplier,
            baseline_lag=config.regime_baseline_lag,
            bidirectional=config.regime_bidirectional,
            valley_quantile=sweep_valley_quantile,
        )
        pred_labels_s = clf_s.predict(X_test)
        diag = compute_regime_diagnostics(
            y_test.values,
            preds_s,
            prob_s,
            thresh_high_test.values,
            config.regime_prob_threshold,
            valley_threshold=thresh_low_test.values if config.regime_bidirectional else None,
            pred_labels=pred_labels_s,
            bidirectional=config.regime_bidirectional,
        )
        diag.update(
            {
                "regime_quantile": rq,
                "regime_strategy": config.regime_strategy,
                "spike_fallback": fallback_s,
            }
        )
        sweep_diag_rows.append(diag)
    pd.DataFrame(sweep_rows).to_csv(output_dir / "regime_threshold_sweep.csv", index=False)
    pd.DataFrame(sweep_diag_rows).to_csv(
        output_dir / "regime_threshold_sweep_diagnostics.csv",
        index=False,
    )

    pred_df = pd.DataFrame(
        {
            config.datetime_col: test_df[config.datetime_col].values,
            "actual": y_test.values,
            "pred_huber": preds_huber,
            "pred_gbm": preds_gbm,
            "pred_gbm_unweighted": preds_gbm_unweighted,
            "pred_p10": p10,
            "pred_p50": p50,
            "pred_p90": p90,
            "uncertainty_width": p90 - p10,
            "pred_regime": regime_preds,
            "pred_regime_hard": regime_hard_preds,
            "regime_spike_prob": regime_prob,
            "regime_use_spike": regime_use_spike.astype(int),
            "regime_label_true": regime_label_true.values,
            "regime_label_pred": regime_label_pred,
            "regime_threshold_med": regime_thresh_low.values,
            "regime_threshold_ext": regime_thresh_high.values,
            "regime_threshold_low": regime_thresh_low.values,
            "regime_threshold_high": regime_thresh_high.values,
        }
    )
    if hybrid_preds is not None:
        pred_df["pred_hybrid"] = hybrid_preds

    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    model_cols = [
        "pred_huber",
        "pred_gbm",
        "pred_gbm_unweighted",
        "pred_p10",
        "pred_p50",
        "pred_p90",
        "pred_regime",
        "pred_regime_hard",
    ]
    if hybrid_preds is not None:
        model_cols.append("pred_hybrid")

    pricing_rows = []
    for name, series in preds.items():
        metrics = evaluate_predictions(y_test.values, series)
        pricing_rows.append(
            {
                "model": name,
                "mae": metrics["mae"],
                "smape": metrics["smape"],
                "cost_weighted_error": cost_weighted_mae(y_test.values, series),
                "spike_weighted_penalty": spike_weighted_penalty(
                    y_test.values,
                    series,
                    spike_quantile=0.9,
                    spike_multiplier=config.spike_penalty_multiplier,
                ),
                "underprediction_penalty": underprediction_penalty(
                    y_test.values,
                    series,
                    spike_quantile=0.9,
                    underprediction_multiplier=config.underprediction_multiplier,
                ),
            }
        )
    pricing_df = pd.DataFrame(pricing_rows)
    pricing_df[["model", "mae", "smape", "cost_weighted_error"]].to_csv(
        output_dir / "pricing_comparison_table.csv",
        index=False,
    )
    pricing_df.to_csv(output_dir / "pricing_metrics.csv", index=False)

    daily_cost_df = compute_daily_procurement_costs(
        pred_df,
        config.datetime_col,
        "actual",
        model_cols,
    )
    daily_cost_df.to_csv(output_dir / "daily_procurement_cost.csv", index=False)
    plot_cumulative_monetary_error(output_dir, daily_cost_df)

    daily_agg_df = compute_daily_aggregate_errors(
        pred_df,
        config.datetime_col,
        "actual",
        model_cols,
    )
    daily_agg_df.to_csv(output_dir / "daily_aggregate_errors.csv", index=False)
    daily_agg_summary = summarize_daily_aggregate_errors(daily_agg_df)
    daily_agg_summary.to_csv(output_dir / "daily_aggregate_summary.csv", index=False)
    plot_daily_totals(output_dir, daily_agg_df)

    model_to_col = {
        "Huber": "pred_huber",
        "GBM": "pred_gbm",
        "GBM_Unweighted": "pred_gbm_unweighted",
        "P10": "pred_p10",
        "P50": "pred_p50",
        "Regime": "pred_regime",
        "Regime_Hard": "pred_regime_hard",
    }
    if hybrid_preds is not None:
        model_to_col["Hybrid"] = "pred_hybrid"

    daily_compare_rows = []
    for name, series in preds.items():
        metrics = evaluate_predictions(y_test.values, series)
        model_col = model_to_col.get(name)
        daily_error = np.nan
        if model_col is not None:
            match = daily_agg_summary.loc[daily_agg_summary["model"] == model_col, "mean_daily_aggregate_error"]
            if not match.empty:
                daily_error = float(match.iloc[0])
        daily_compare_rows.append(
            {
                "model": name,
                "mae": metrics["mae"],
                "smape": metrics["smape"],
                "daily_aggregate_error": daily_error,
            }
        )
    pd.DataFrame(daily_compare_rows).to_csv(
        output_dir / "daily_aggregate_comparison_table.csv",
        index=False,
    )

    # Calibration output
    true_spike = (
        (regime_label_true.values == 2)
        if config.regime_bidirectional
        else (regime_label_true.values > 0)
    ).astype(int)
    calib_df = build_calibration_table(regime_prob, true_spike, config.calibration_bins)
    calib_df.to_csv(output_dir / "calibration_curve.csv", index=False)
    plot_calibration_curve(output_dir, calib_df)

    # Rolling-window evaluation
    rolling_df = compute_rolling_metrics(
        pred_df,
        config.datetime_col,
        model_cols,
        config.rolling_window_days,
        config.rolling_step_days,
    )
    rolling_df.to_csv(output_dir / "rolling_metrics.csv", index=False)

    if config.compare_regime_strategies:
        comparison_df = run_regime_strategy_comparison(
            output_dir,
            X_train,
            y_train,
            train_df[config.datetime_col],
            X_test,
            y_test,
            test_df[config.datetime_col],
            gbm_backend,
            gbm_params,
            config.target_transform,
            config.weight_quantile,
            config.weight_multiplier,
            config.regime_quantile,
            config.regime_extreme_quantile,
            config.regime_min_samples,
            config.regime_spike_weight,
            config.regime_classes,
            config.rolling_window_days,
            config.regime_zscore_threshold,
            config.regime_vol_multiplier,
            config.regime_baseline_lag,
            config.regime_bidirectional,
            config.regime_valley_quantile,
        )
        comparison_df.to_csv(output_dir / "regime_strategy_comparison.csv", index=False)

    # Feature importances
    # Robustly get the fitted classifier inside the pipeline (support different step names)
    clf = None
    # If classifier is a sklearn Pipeline, try to locate the inner estimator
    if hasattr(classifier, "named_steps"):
        for step_name in ("logreg", "clf", "model"):
            if step_name in classifier.named_steps:
                clf = classifier.named_steps[step_name]
                break
        if clf is None:
            # fallback to last step in pipeline
            clf = classifier.steps[-1][1]
    else:
        # classifier is not a Pipeline (e.g., DummyClassifier) — use it directly
        clf = classifier

    # Extract coefficients if available, otherwise create an empty table
    coef = getattr(clf, "coef_", None)
    if coef is None:
        clf_imp = pd.DataFrame({"feature": feature_cols})
        clf_imp["abs_mean_coef"] = 0.0
    else:
        coef = np.atleast_2d(coef)
        clf_imp = pd.DataFrame({"feature": feature_cols})
        for idx in range(coef.shape[0]):
            clf_imp[f"class_{idx}_coef"] = coef[idx]
        clf_imp["abs_mean_coef"] = np.mean(np.abs(coef), axis=0)
        clf_imp = clf_imp.sort_values("abs_mean_coef", ascending=False)
    clf_imp.to_csv(output_dir / "feature_importance_regime_classifier.csv", index=False)

    if config.regime_bidirectional:
        regime_model_specs = [
            ("valley", low_model),
            ("normal", mid_model),
            ("spike", high_model),
        ]
    else:
        regime_model_specs = [
            ("normal", low_model),
            ("medium", mid_model),
            ("extreme", high_model),
        ]

    for name, model in regime_model_specs:
        if hasattr(model, "feature_importances_"):
            pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False).to_csv(
                output_dir / f"feature_importance_regime_{name}_model.csv",
                index=False,
            )

    plot_results(
        output_dir,
        test_df[config.datetime_col],
        y_test.values,
        preds,
        p10,
        p50,
        p90,
        config.plot_points,
    )

    if getattr(config, "save_deployment_artifacts", True):
        save_deployment_artifacts(
            config,
            feature_cols,
            target_col,
            market_columns,
            gbm_backend,
            gbm_params,
            quantile_backend,
            huber_model,
            gbm_model,
            classifier,
            low_model,
            mid_model,
            high_model,
            quantile_models,
        )

    print("Saved outputs to:", output_dir)
    print(comparison_table)
    print("Regime spike usage:", int(regime_use_spike.sum()))
    print("Regime spike fallback:", bool(regime_fallback))
    print("Regime holdout diagnostics:", regime_holdout_diag)


if __name__ == "__main__":
    run_pipeline(parse_args())
