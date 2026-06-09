import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import HuberRegressor, LogisticRegression, QuantileRegressor
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None


DEFAULT_WEATHER_COLS = ["temp", "solar", "cloud", "wind", "humidity", "rain"]


def smape(y_true, y_pred, epsilon=1e-6):
    num = np.abs(y_pred - y_true)
    denom = (np.abs(y_true) + np.abs(y_pred)) + epsilon
    return float(np.mean(2.0 * num / denom) * 100.0)


def build_sample_weights(y_values, quantile=0.9, high_weight=3.0):
    threshold = np.percentile(y_values, quantile * 100.0)
    return np.where(y_values > threshold, high_weight, 1.0)


def transform_target(y_values, use_log_target):
    if not use_log_target:
        return y_values
    if np.any(y_values < 0):
        raise ValueError("Log target requested but negative values found.")
    return np.log1p(y_values)


def inverse_transform_target(y_values, use_log_target):
    if not use_log_target:
        return y_values
    return np.expm1(y_values)


def fit_with_weights(model, X, y, weights):
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


def feature_engineering(
    df,
    datetime_col="datetime",
    price_col="price",
    buy_col="buy_mw",
    sell_col="sell_mw",
    weather_cols=None,
    lags=(96, 192),
    rolling_window=96,
):
    df = df.copy()
    missing = [c for c in [datetime_col, price_col, buy_col, sell_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col)

    df["hour"] = df[datetime_col].dt.hour
    df["weekday"] = df[datetime_col].dt.weekday
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["demand_supply_ratio"] = df[buy_col] / (df[sell_col] + 1e-6)

    for lag in lags:
        df[f"price_lag_{lag}"] = df[price_col].shift(lag)

    # Shift before rolling to avoid leakage from the current price.
    df[f"price_rolling_mean_{rolling_window}"] = (
        df[price_col].shift(1).rolling(rolling_window).mean()
    )
    df[f"price_rolling_std_{rolling_window}"] = (
        df[price_col].shift(1).rolling(rolling_window).std()
    )

    if "solar" in df.columns:
        df["solar_hour_interaction"] = df["solar"] * df["hour_sin"]

    if weather_cols is None:
        weather_cols = [col for col in DEFAULT_WEATHER_COLS if col in df.columns]
    else:
        weather_cols = [col for col in weather_cols if col in df.columns]

    feature_cols = [
        "hour",
        "weekday",
        "hour_sin",
        "hour_cos",
        f"price_lag_{lags[0]}",
        f"price_lag_{lags[1]}",
        f"price_rolling_mean_{rolling_window}",
        f"price_rolling_std_{rolling_window}",
        "demand_supply_ratio",
    ]

    if "solar_hour_interaction" in df.columns:
        feature_cols.append("solar_hour_interaction")

    feature_cols.extend(weather_cols)

    df = df.dropna(subset=feature_cols + [price_col])
    return df, feature_cols


def time_based_split(df, test_size=0.2):
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def build_huber_model():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", HuberRegressor(max_iter=1000)),
        ]
    )


def build_spike_model():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def build_regime_classifier():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def compute_regime_labels(
    y_values,
    strategy,
    rolling_window,
    quantile,
    zscore_threshold,
    vol_multiplier,
    baseline_lag,
):
    y_series = pd.Series(y_values)
    baseline_lag = max(1, int(baseline_lag))
    history = y_series.shift(1)
    min_periods = max(1, int(rolling_window * 0.25))
    roll = history.rolling(rolling_window, min_periods=min_periods)
    quantile_thresh = roll.quantile(quantile)
    mean_s = roll.mean()
    std_s = roll.std()

    z_thresh = mean_s + zscore_threshold * std_s
    baseline = y_series.shift(baseline_lag)
    vol_thresh = baseline + vol_multiplier * std_s

    if strategy == "rolling_quantile":
        threshold = quantile_thresh
    elif strategy == "zscore":
        threshold = z_thresh
    elif strategy == "vol_adj":
        threshold = vol_thresh
    elif strategy == "hybrid":
        threshold = pd.concat(
            [quantile_thresh, z_thresh, vol_thresh],
            axis=1,
        ).min(axis=1)
    else:
        raise ValueError(f"Unknown regime strategy: {strategy}")

    threshold = threshold.ffill().bfill()
    return threshold, (y_series > threshold).astype(int).values


def blend_predictions(spike_prob, normal_pred, spike_pred):
    return (1.0 - spike_prob) * normal_pred + spike_prob * spike_pred


def compute_regime_diagnostics(
    y_true,
    y_pred,
    spike_prob,
    spike_threshold,
    prob_threshold,
    valley_threshold=None,
    pred_labels=None,
    bidirectional=False,
):
    def subset_metrics(name, mask):
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

    threshold_high = np.asarray(spike_threshold)
    threshold_high = np.where(np.isnan(threshold_high), np.inf, threshold_high)
    true_spike = y_true > threshold_high

    if valley_threshold is not None:
        threshold_low = np.asarray(valley_threshold)
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


def plot_regime_confusion_matrix(
    output_dir: Path,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    label_names: list[str] | None = None,
    filename: str = "regime_confusion_matrix.png",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if label_names is None:
        label_names = ["Normal", "Medium", "Extreme"]
    labels = list(range(len(label_names)))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(label_names, rotation=30, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_title("Regime Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def plot_regime_distribution_over_time(
    output_dir: Path,
    timestamps: pd.Series,
    regime_labels: np.ndarray,
    label_names: list[str] | None = None,
    filename: str = "regime_distribution_over_time.png",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if label_names is None:
        label_names = ["Normal", "Medium", "Extreme"]
    df = pd.DataFrame(
        {"datetime": pd.to_datetime(timestamps), "label": regime_labels}
    ).sort_values("datetime")
    df["date"] = df["datetime"].dt.date
    counts = df.groupby(["date", "label"]).size().unstack(fill_value=0)
    proportions = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    for label_id, name in enumerate(label_names):
        if label_id in proportions.columns:
            ax.plot(
                pd.to_datetime(proportions.index),
                proportions[label_id],
                label=name,
            )
    ax.set_ylim(0, 1)
    ax.set_title("Regime Distribution Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Share of observations")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def plot_spike_timeline(
    output_dir: Path,
    timestamps: pd.Series,
    y_true: np.ndarray,
    regime_labels: np.ndarray,
    thresh_low: np.ndarray | None = None,
    thresh_high: np.ndarray | None = None,
    label_names: list[str] | None = None,
    filename: str = "spike_timeline.png",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    times = pd.to_datetime(timestamps)
    label_names = label_names or ["Normal", "Medium", "Extreme"]
    has_valley = any("valley" in name.lower() for name in label_names)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, y_true, color="black", linewidth=1.2, label="Actual")
    if thresh_low is not None:
        ax.plot(
            times,
            thresh_low,
            linestyle="--",
            color="#1f77b4",
            label=f"{label_names[0]} threshold" if has_valley else "Medium threshold",
        )
    if thresh_high is not None:
        ax.plot(
            times,
            thresh_high,
            linestyle="--",
            color="#d62728",
            label=f"{label_names[-1]} threshold" if has_valley else "Extreme threshold",
        )

    if has_valley:
        valley_mask = regime_labels == 0
        spike_mask = regime_labels == 2
        if valley_mask.any():
            ax.scatter(
                times[valley_mask],
                y_true[valley_mask],
                color="#1f77b4",
                s=12,
                label=label_names[0],
            )
        if spike_mask.any():
            ax.scatter(
                times[spike_mask],
                y_true[spike_mask],
                color="#d62728",
                s=12,
                label=label_names[-1],
            )
        ax.set_title("Regime Occurrence Timeline")
    else:
        med_mask = regime_labels == 1
        ext_mask = regime_labels == 2
        if med_mask.any():
            ax.scatter(times[med_mask], y_true[med_mask], color="#ff7f0e", s=12, label=label_names[1])
        if ext_mask.any():
            ext_label = label_names[2] if len(label_names) > 2 else "Spike"
            ax.scatter(times[ext_mask], y_true[ext_mask], color="#d62728", s=12, label=ext_label)
        ax.set_title("Spike Occurrence Timeline")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def build_boosting_model(model_type, params):
    if model_type == "lightgbm":
        return lgb.LGBMRegressor(
            objective="regression",
            random_state=42,
            n_jobs=-1,
            **params,
        )
    if model_type == "xgboost":
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def build_quantile_model(model_type, quantile, params):
    if model_type == "lightgbm":
        return lgb.LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            random_state=42,
            n_jobs=-1,
            **params,
        )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", QuantileRegressor(quantile=quantile, alpha=0.0, solver="highs")),
        ]
    )


def train_regime_models(
    X_train,
    y_train,
    model_type,
    model_params,
    use_log_target,
    weight_quantile,
    weight_multiplier,
    regime_quantile,
    regime_strategy,
    regime_rolling_window,
    regime_zscore_threshold,
    regime_vol_multiplier,
    regime_baseline_lag,
    min_spike_samples,
    regime_spike_weight,
):
    threshold, spike_labels = compute_regime_labels(
        y_train,
        strategy=regime_strategy,
        rolling_window=regime_rolling_window,
        quantile=regime_quantile,
        zscore_threshold=regime_zscore_threshold,
        vol_multiplier=regime_vol_multiplier,
        baseline_lag=regime_baseline_lag,
    )
    classifier = build_regime_classifier()
    classifier.fit(X_train, spike_labels)

    normal_mask = spike_labels == 0
    spike_mask = spike_labels == 1

    normal_model = build_boosting_model(model_type, model_params)
    spike_model = build_boosting_model(model_type, model_params)
    global_model = build_boosting_model(model_type, model_params)

    def fit_subset(model, X_sub, y_sub, weights_override=None):
        y_trans = pd.Series(
            transform_target(y_sub.values, use_log_target),
            index=y_sub.index,
        )
        if weights_override is None:
            weights = build_sample_weights(
                y_sub.values,
                quantile=weight_quantile,
                high_weight=weight_multiplier,
            )
        else:
            weights = weights_override
        fit_with_weights(model, X_sub, y_trans, weights)

    global_weights = build_sample_weights(
        y_train.values,
        quantile=weight_quantile,
        high_weight=weight_multiplier,
    )
    fit_subset(global_model, X_train, y_train, global_weights)

    if normal_mask.sum() == 0:
        normal_model = global_model
    else:
        fit_subset(normal_model, X_train[normal_mask], y_train[normal_mask])

    spike_fallback = False
    if spike_mask.sum() < min_spike_samples:
        normal_model = global_model
        spike_model = global_model
        spike_fallback = True
    else:
        spike_weights = np.where(spike_labels == 1, regime_spike_weight, 1.0)
        fit_subset(spike_model, X_train, y_train, spike_weights)

    return classifier, normal_model, spike_model, threshold, spike_fallback


def predict_regime(
    X,
    classifier,
    normal_model,
    spike_model,
    prob_threshold,
    use_log_target,
):
    spike_prob = classifier.predict_proba(X)[:, 1]
    pred_normal = inverse_transform_target(normal_model.predict(X), use_log_target)
    pred_spike = inverse_transform_target(spike_model.predict(X), use_log_target)
    preds = blend_predictions(spike_prob, pred_normal, pred_spike)
    use_spike = spike_prob > prob_threshold
    return preds, spike_prob, use_spike


def evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "smape": smape(y_true, y_pred),
    }


def cross_validate_model(
    build_model_fn,
    X,
    y,
    n_splits,
    use_log_target,
    weight_quantile,
    weight_multiplier,
    debug_fold,
):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        model = build_model_fn()
        y_train_raw = y.iloc[train_idx]
        y_val_raw = y.iloc[val_idx]
        weights = build_sample_weights(
            y_train_raw.values,
            quantile=weight_quantile,
            high_weight=weight_multiplier,
        )
        y_train_trans = pd.Series(
            transform_target(y_train_raw.values, use_log_target),
            index=y_train_raw.index,
        )
        fit_with_weights(model, X.iloc[train_idx], y_train_trans, weights)
        preds_trans = model.predict(X.iloc[val_idx])
        preds = inverse_transform_target(preds_trans, use_log_target)
        metrics = evaluate(y_val_raw, preds)
        metrics["fold"] = fold
        rows.append(metrics)

        if debug_fold is not None and fold == debug_fold:
            print("\nFold debug:")
            print(y_train_raw.describe())
            print("Weights summary:")
            print(pd.Series(weights).describe())
    return pd.DataFrame(rows)


def cross_validate_regime_model(
    X,
    y,
    model_type,
    model_params,
    n_splits,
    use_log_target,
    weight_quantile,
    weight_multiplier,
    regime_quantile,
    regime_strategy,
    regime_rolling_window,
    regime_zscore_threshold,
    regime_vol_multiplier,
    regime_baseline_lag,
    regime_prob_threshold,
    min_spike_samples,
    regime_spike_weight,
    debug_fold,
):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    diag_rows = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        classifier, normal_model, spike_model, threshold, spike_fallback = train_regime_models(
            X_train,
            y_train,
            model_type,
            model_params,
            use_log_target,
            weight_quantile,
            weight_multiplier,
            regime_quantile,
            regime_strategy,
            regime_rolling_window,
            regime_zscore_threshold,
            regime_vol_multiplier,
            regime_baseline_lag,
            min_spike_samples,
            regime_spike_weight,
        )

        preds, spike_prob, use_spike = predict_regime(
            X_val,
            classifier,
            normal_model,
            spike_model,
            regime_prob_threshold,
            use_log_target,
        )
        metrics = evaluate(y_val, preds)
        metrics["fold"] = fold
        rows.append(metrics)

        diag = compute_regime_diagnostics(
            y_val.values,
            preds,
            spike_prob,
            threshold,
            regime_prob_threshold,
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


def hard_switch_predictions(
    spike_prob: np.ndarray,
    normal_pred: np.ndarray,
    spike_pred: np.ndarray,
    prob_threshold: float,
) -> np.ndarray:
    use_spike = spike_prob > prob_threshold
    return np.where(use_spike, spike_pred, normal_pred)


def evaluate_price_buckets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q90: float,
    q95: float,
) -> list[dict]:
    buckets = [
        ("top_10", y_true >= q90),
        ("top_5", y_true >= q95),
        ("normal", y_true < q90),
    ]
    rows = []
    for name, mask in buckets:
        if mask.sum() == 0:
            metrics = {"mae": np.nan, "r2": np.nan, "smape": np.nan}
        else:
            metrics = {
                "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
                "r2": float(r2_score(y_true[mask], y_pred[mask])),
                "smape": float(smape(y_true[mask], y_pred[mask])),
            }
        metrics["bucket"] = name
        metrics["count"] = int(mask.sum())
        rows.append(metrics)
    return rows


def build_calibration_table(
    spike_prob: np.ndarray,
    true_spike: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(spike_prob, edges, right=True) - 1
    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            rows.append(
                {
                    "bin": b,
                    "count": 0,
                    "predicted": np.nan,
                    "actual": np.nan,
                    "bin_start": edges[b],
                    "bin_end": edges[b + 1],
                }
            )
            continue
        rows.append(
            {
                "bin": b,
                "count": int(mask.sum()),
                "predicted": float(np.mean(spike_prob[mask])),
                "actual": float(np.mean(true_spike[mask])),
                "bin_start": edges[b],
                "bin_end": edges[b + 1],
            }
        )
    return pd.DataFrame(rows)


def plot_calibration_curve(output_dir: Path, calib_df: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.plot(calib_df["predicted"], calib_df["actual"], marker="o", color="#1f77b4")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted spike probability")
    ax.set_ylabel("Observed spike frequency")
    ax.set_title("Spike Probability Calibration")
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_curve.png")
    plt.close(fig)


def compute_rolling_metrics(
    pred_df: pd.DataFrame,
    datetime_col: str,
    model_cols: list[str],
    window_days: int,
    step_days: int,
) -> pd.DataFrame:
    df = pred_df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col)

    start = df[datetime_col].min()
    end = df[datetime_col].max()

    rows = []
    window = pd.Timedelta(days=window_days)
    step = pd.Timedelta(days=step_days)
    current = start

    while current + window <= end:
        window_end = current + window
        mask = (df[datetime_col] >= current) & (df[datetime_col] < window_end)
        if mask.sum() == 0:
            current += step
            continue
        for col in model_cols:
            metrics = {
                "mae": float(mean_absolute_error(df.loc[mask, "actual"].values, df.loc[mask, col].values)),
                "r2": float(r2_score(df.loc[mask, "actual"].values, df.loc[mask, col].values)),
                "smape": float(smape(df.loc[mask, "actual"].values, df.loc[mask, col].values)),
            }
            metrics.update(
                {
                    "model": col,
                    "window_start": current,
                    "window_end": window_end,
                    "count": int(mask.sum()),
                }
            )
            rows.append(metrics)
        current += step

    return pd.DataFrame(rows)


def cost_weighted_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    weighted_error = np.abs(y_true - y_pred) * y_true
    return float(np.mean(weighted_error))


def spike_weighted_penalty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spike_quantile: float,
    spike_multiplier: float,
) -> float:
    threshold = float(np.percentile(y_true, spike_quantile * 100.0))
    weights = np.where(y_true >= threshold, spike_multiplier, 1.0)
    return float(np.mean(np.abs(y_true - y_pred) * weights))


def underprediction_penalty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spike_quantile: float,
    underprediction_multiplier: float,
) -> float:
    threshold = float(np.percentile(y_true, spike_quantile * 100.0))
    high_mask = y_true >= threshold
    under_mask = y_pred < y_true
    weights = np.where(high_mask & under_mask, underprediction_multiplier, 1.0)
    return float(np.mean(np.abs(y_true - y_pred) * weights))


def compute_daily_procurement_costs(
    pred_df: pd.DataFrame,
    datetime_col: str,
    actual_col: str,
    model_cols: list[str],
) -> pd.DataFrame:
    df = pred_df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df["date"] = df[datetime_col].dt.date
    rows = []
    for col in model_cols:
        daily = (
            df.groupby("date")
            .agg(actual_cost=(actual_col, "sum"), predicted_cost=(col, "sum"))
            .reset_index()
        )
        daily["cost_error"] = daily["predicted_cost"] - daily["actual_cost"]
        daily["abs_cost_error"] = daily["cost_error"].abs()
        daily["model"] = col
        rows.append(daily)
    return pd.concat(rows, ignore_index=True)


def plot_cumulative_monetary_error(output_dir: Path, daily_df: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, group in daily_df.groupby("model"):
        group = group.sort_values("date")
        dates = pd.to_datetime(group["date"])
        cumulative_error = group["cost_error"].cumsum()
        ax.plot(dates, cumulative_error, label=model_name)
    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    ax.set_title("Cumulative Monetary Error")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative cost error")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_monetary_error.png")
    plt.close(fig)


def compute_daily_aggregate_errors(
    pred_df: pd.DataFrame,
    datetime_col: str,
    actual_col: str,
    model_cols: list[str],
    eps: float = 1e-6,
) -> pd.DataFrame:
    df = pred_df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df["date"] = df[datetime_col].dt.date
    rows = []
    for col in model_cols:
        daily = (
            df.groupby("date")
            .agg(daily_actual=(actual_col, "sum"), daily_pred=(col, "sum"))
            .reset_index()
        )
        denom = np.maximum(daily["daily_actual"].values, eps)
        daily["daily_aggregate_error"] = np.abs(daily["daily_pred"] - daily["daily_actual"]) / denom
        daily["model"] = col
        rows.append(daily)
    return pd.concat(rows, ignore_index=True)


def summarize_daily_aggregate_errors(daily_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        daily_df.groupby("model")
        .agg(
            mean_daily_aggregate_error=("daily_aggregate_error", "mean"),
            median_daily_aggregate_error=("daily_aggregate_error", "median"),
            worst_daily_aggregate_error=("daily_aggregate_error", "max"),
        )
        .reset_index()
    )
    return summary


def plot_daily_totals(output_dir: Path, daily_df: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    actual = (
        daily_df.drop_duplicates("date")
        .sort_values("date")
        .loc[:, ["date", "daily_actual"]]
    )
    ax.plot(pd.to_datetime(actual["date"]), actual["daily_actual"], label="Actual", color="black")

    for model_name, group in daily_df.groupby("model"):
        group = group.sort_values("date")
        ax.plot(pd.to_datetime(group["date"]), group["daily_pred"], label=model_name)

    ax.set_title("Daily Total: Actual vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily total price")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "daily_totals_actual_vs_predicted.png")
    plt.close(fig)


def candidate_param_grid(model_type):
    if model_type == "lightgbm":
        return [
            {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": -1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            {
                "n_estimators": 300,
                "learning_rate": 0.1,
                "num_leaves": 63,
                "max_depth": -1,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        ]
    return [
        {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
    ]


def tune_boosting_model(
    model_type,
    X,
    y,
    n_splits,
    use_log_target,
    weight_quantile,
    weight_multiplier,
):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_params = None
    best_mae = np.inf

    for params in candidate_param_grid(model_type):
        fold_mae = []
        for train_idx, val_idx in tscv.split(X):
            model = build_boosting_model(model_type, params)
            y_train_raw = y.iloc[train_idx]
            y_val_raw = y.iloc[val_idx]
            weights = build_sample_weights(
                y_train_raw.values,
                quantile=weight_quantile,
                high_weight=weight_multiplier,
            )
            y_train_trans = pd.Series(
                transform_target(y_train_raw.values, use_log_target),
                index=y_train_raw.index,
            )
            fit_with_weights(model, X.iloc[train_idx], y_train_trans, weights)
            preds_trans = model.predict(X.iloc[val_idx])
            preds = inverse_transform_target(preds_trans, use_log_target)
            fold_mae.append(mean_absolute_error(y_val_raw, preds))
        avg_mae = float(np.mean(fold_mae))
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_params = params

    return best_params, best_mae


def select_boosting_backend(preferred="auto"):
    if preferred == "lightgbm":
        if lgb is None:
            raise ImportError("lightgbm is not installed.")
        return "lightgbm"
    if preferred == "xgboost":
        if xgb is None:
            raise ImportError("xgboost is not installed.")
        return "xgboost"

    if lgb is not None:
        return "lightgbm"
    if xgb is not None:
        return "xgboost"
    raise ImportError("Install lightgbm or xgboost to use the boosting model.")


def apply_spike_adjustment(base_pred, spike_prob, spike_uplift):
    return base_pred * (1.0 + spike_uplift * spike_prob)


def train_models(
    train_df,
    test_df,
    feature_cols,
    price_col,
    model_type,
    n_splits,
    spike_threshold,
    spike_uplift,
    use_spike_model,
    use_log_target,
    weight_quantile,
    weight_multiplier,
    debug_fold,
    regime_quantile,
    regime_strategy,
    regime_rolling_window,
    regime_zscore_threshold,
    regime_vol_multiplier,
    regime_baseline_lag,
    regime_prob_threshold,
    regime_min_samples,
    regime_spike_weight,
):
    X_train = train_df[feature_cols]
    y_train = train_df[price_col]
    X_test = test_df[feature_cols]
    y_test = test_df[price_col]

    y_train_trans = pd.Series(
        transform_target(y_train.values, use_log_target),
        index=y_train.index,
    )
    weights = build_sample_weights(
        y_train.values,
        quantile=weight_quantile,
        high_weight=weight_multiplier,
    )

    huber_model = build_huber_model()
    fit_with_weights(huber_model, X_train, y_train_trans, weights)
    huber_pred = inverse_transform_target(huber_model.predict(X_test), use_log_target)

    best_params, best_mae = tune_boosting_model(
        model_type,
        X_train,
        y_train,
        n_splits,
        use_log_target,
        weight_quantile,
        weight_multiplier,
    )
    boost_model = build_boosting_model(model_type, best_params)
    fit_with_weights(boost_model, X_train, y_train_trans, weights)
    boost_pred = inverse_transform_target(boost_model.predict(X_test), use_log_target)

    p50_model = build_quantile_model(model_type, 0.5, best_params)
    p90_model = build_quantile_model(model_type, 0.9, best_params)
    fit_with_weights(p50_model, X_train, y_train_trans, weights)
    fit_with_weights(p90_model, X_train, y_train_trans, weights)
    p50_pred = inverse_transform_target(p50_model.predict(X_test), use_log_target)
    p90_pred = inverse_transform_target(p90_model.predict(X_test), use_log_target)

    predictions = {
        "Huber": huber_pred,
        model_type.upper(): boost_pred,
        "P50": p50_pred,
        "P90": p90_pred,
    }

    models = {
        "Huber": huber_model,
        model_type.upper(): boost_model,
        "P50": p50_model,
        "P90": p90_model,
        "boosting_params": best_params,
        "boosting_cv_mae": best_mae,
    }

    regime_outputs = None
    classifier, normal_model, spike_model, threshold, spike_fallback = train_regime_models(
        X_train,
        y_train,
        model_type,
        best_params,
        use_log_target,
        weight_quantile,
        weight_multiplier,
        regime_quantile,
        regime_strategy,
        regime_rolling_window,
        regime_zscore_threshold,
        regime_vol_multiplier,
        regime_baseline_lag,
        regime_min_samples,
        regime_spike_weight,
    )
    regime_pred, spike_prob, use_spike = predict_regime(
        X_test,
        classifier,
        normal_model,
        spike_model,
        regime_prob_threshold,
        use_log_target,
    )
    predictions["Regime"] = regime_pred
    models["Regime_classifier"] = classifier
    models["Regime_normal"] = normal_model
    models["Regime_spike"] = spike_model
    models["Regime_threshold"] = threshold
    models["Regime_spike_fallback"] = spike_fallback
    regime_outputs = {
        "spike_prob": spike_prob,
        "use_spike": use_spike,
        "threshold": threshold,
        "spike_fallback": spike_fallback,
        "holdout_diag": compute_regime_diagnostics(
            y_test.values,
            regime_pred,
            spike_prob,
            threshold,
            regime_prob_threshold,
        ),
    }

    if use_spike_model:
        lag_col = "price_lag_96"
        if lag_col not in train_df.columns:
            raise ValueError("price_lag_96 is required for spike detection.")

        spike_label = (train_df[price_col] > train_df[lag_col] + spike_threshold).astype(int)
        spike_model = build_spike_model()
        spike_model.fit(X_train, spike_label)
        spike_prob = spike_model.predict_proba(X_test)[:, 1]

        predictions["Huber_Hybrid"] = apply_spike_adjustment(
            huber_pred, spike_prob, spike_uplift
        )
        predictions[f"{model_type.upper()}_Hybrid"] = apply_spike_adjustment(
            boost_pred, spike_prob, spike_uplift
        )
        models["Spike"] = spike_model

    huber_cv = cross_validate_model(
        build_huber_model,
        X_train,
        y_train,
        n_splits,
        use_log_target,
        weight_quantile,
        weight_multiplier,
        debug_fold,
    )
    huber_cv["model"] = "Huber"

    boost_cv = cross_validate_model(
        lambda: build_boosting_model(model_type, best_params),
        X_train,
        y_train,
        n_splits,
        use_log_target,
        weight_quantile,
        weight_multiplier,
        debug_fold,
    )
    boost_cv["model"] = model_type.upper()

    p50_cv = cross_validate_model(
        lambda: build_quantile_model(model_type, 0.5, best_params),
        X_train,
        y_train,
        n_splits,
        use_log_target,
        weight_quantile,
        weight_multiplier,
        debug_fold,
    )
    p50_cv["model"] = "P50"

    regime_cv, regime_diag_cv = cross_validate_regime_model(
        X_train,
        y_train,
        model_type,
        best_params,
        n_splits,
        use_log_target,
        weight_quantile,
        weight_multiplier,
        regime_quantile,
        regime_strategy,
        regime_rolling_window,
        regime_zscore_threshold,
        regime_vol_multiplier,
        regime_baseline_lag,
        regime_prob_threshold,
        regime_min_samples,
        regime_spike_weight,
        debug_fold,
    )
    regime_cv["model"] = "Regime"

    cv_metrics = pd.concat([huber_cv, boost_cv, p50_cv, regime_cv], ignore_index=True)
    return models, predictions, cv_metrics, y_test, regime_outputs, regime_diag_cv


def plot_results(pred_df, output_dir, plot_points=500, boost_label="BOOST"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = pred_df.tail(plot_points) if len(pred_df) > plot_points else pred_df

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df["datetime"], plot_df["actual"], label="Actual", color="black")
    if "Huber" in plot_df.columns:
        plt.plot(plot_df["datetime"], plot_df["Huber"], label="Huber", alpha=0.8)
    if boost_label in plot_df.columns:
        plt.plot(plot_df["datetime"], plot_df[boost_label], label=boost_label, alpha=0.8)
    plt.title("Actual vs Predicted")
    plt.xlabel("Datetime")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png")
    plt.close()

    if boost_label in pred_df.columns:
        residuals = pred_df["actual"] - pred_df[boost_label]
        plt.figure(figsize=(10, 5))
        plt.scatter(pred_df[boost_label], residuals, alpha=0.4)
        plt.axhline(0, linestyle="--", color="black")
        plt.title("Residuals vs Predicted (Boosting)")
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.tight_layout()
        plt.savefig(output_dir / "residuals.png")
        plt.close()

    if "P50" in pred_df.columns and "P90" in pred_df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(plot_df["datetime"], plot_df["actual"], label="Actual", color="black")
        plt.plot(plot_df["datetime"], plot_df["P50"], label="P50", alpha=0.8)
        plt.plot(plot_df["datetime"], plot_df["P90"], label="P90", alpha=0.8)
        plt.fill_between(
            plot_df["datetime"],
            plot_df["P50"],
            plot_df["P90"],
            alpha=0.2,
            label="P50-P90",
        )
        plt.title("Quantile Bands")
        plt.xlabel("Datetime")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "quantile_bands.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Time-series pipeline for MCP prediction")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output-dir", default="outputs", help="Directory for outputs")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test size")
    parser.add_argument("--cv-splits", type=int, default=5, help="TimeSeriesSplit folds")
    parser.add_argument(
        "--weather-cols",
        default=None,
        help="Comma-separated weather columns (default: auto-detect)",
    )
    parser.add_argument(
        "--boosting",
        default="auto",
        choices=["auto", "lightgbm", "xgboost"],
        help="Boosting backend",
    )
    parser.add_argument("--plot-points", type=int, default=500, help="Points to plot")
    parser.add_argument(
        "--use-log-target",
        action="store_true",
        help="Train on log1p(price) and inverse-transform predictions",
    )
    parser.add_argument(
        "--weight-quantile",
        type=float,
        default=0.9,
        help="Quantile for spike weighting (default: 0.9)",
    )
    parser.add_argument(
        "--weight-multiplier",
        type=float,
        default=3.0,
        help="High-price sample weight (default: 3.0)",
    )
    parser.add_argument(
        "--regime-quantile",
        type=float,
        default=0.9,
        help="Quantile threshold for spike regime (default: 0.9)",
    )
    parser.add_argument(
        "--regime-prob-threshold",
        type=float,
        default=0.3,
        help="Spike probability cutoff for regime switching (default: 0.3)",
    )
    parser.add_argument(
        "--regime-min-samples",
        type=int,
        default=50,
        help="Minimum spike samples required to train spike model (default: 50)",
    )
    parser.add_argument(
        "--regime-spike-weight",
        type=float,
        default=5.0,
        help="Spike weight used when training spike model on all data (default: 5.0)",
    )
    parser.add_argument(
        "--debug-fold",
        type=int,
        default=None,
        help="Print diagnostics for a specific CV fold (1-based)",
    )
    parser.add_argument(
        "--use-spike-model",
        action="store_true",
        help="Enable spike-aware hybrid adjustment",
    )
    parser.add_argument(
        "--spike-threshold",
        type=float,
        default=50.0,
        help="Spike threshold for price > lag96 + threshold",
    )
    parser.add_argument(
        "--spike-uplift",
        type=float,
        default=0.2,
        help="Multiplicative uplift for spike probability",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    weather_cols = None
    if args.weather_cols:
        weather_cols = [col.strip() for col in args.weather_cols.split(",") if col.strip()]

    df, feature_cols = feature_engineering(
        df,
        weather_cols=weather_cols,
    )

    train_df, test_df = time_based_split(df, test_size=args.test_size)
    model_type = select_boosting_backend(args.boosting)

    models, predictions, cv_metrics, y_test, regime_outputs, regime_diag_cv = train_models(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        price_col="price",
        model_type=model_type,
        n_splits=args.cv_splits,
        spike_threshold=args.spike_threshold,
        spike_uplift=args.spike_uplift,
        use_spike_model=args.use_spike_model,
        use_log_target=args.use_log_target,
        weight_quantile=args.weight_quantile,
        weight_multiplier=args.weight_multiplier,
        debug_fold=args.debug_fold,
        regime_quantile=args.regime_quantile,
        regime_strategy=args.regime_strategy,
        regime_rolling_window=args.regime_rolling_window,
        regime_zscore_threshold=args.regime_zscore_threshold,
        regime_vol_multiplier=args.regime_vol_multiplier,
        regime_baseline_lag=args.regime_baseline_lag,
        regime_prob_threshold=args.regime_prob_threshold,
        regime_min_samples=args.regime_min_samples,
        regime_spike_weight=args.regime_spike_weight,
    )

    pred_df = pd.DataFrame({
        "datetime": test_df["datetime"].values,
        "actual": y_test.values,
    })
    for name, preds in predictions.items():
        pred_df[name] = preds
    if regime_outputs is not None:
        pred_df["Regime_spike_prob"] = regime_outputs["spike_prob"]
        pred_df["Regime_use_spike"] = regime_outputs["use_spike"].astype(int)

    summary_rows = []
    for model_name, preds in predictions.items():
        if model_name == "P90":
            continue
        metrics = evaluate(y_test, preds)
        metrics["model"] = model_name
        summary_rows.append(metrics)

    comparison_df = pd.DataFrame(summary_rows)
    comparison_df = comparison_df[["model", "mae", "r2", "smape"]]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    comparison_df.to_csv(output_dir / "comparison_table.csv", index=False)
    cv_metrics.to_csv(output_dir / "metrics_cv.csv", index=False)
    if regime_outputs is not None:
        holdout_diag = pd.DataFrame([regime_outputs["holdout_diag"]])
        holdout_diag.to_csv(output_dir / "regime_diagnostics_holdout.csv", index=False)
    if regime_diag_cv is not None:
        regime_diag_cv.to_csv(output_dir / "regime_diagnostics_cv.csv", index=False)

    plot_results(pred_df, output_dir, plot_points=args.plot_points, boost_label=model_type.upper())

    print("\nComparison Table (Holdout):")
    print(comparison_df.to_string(index=False))
    print("\nCV Metrics (per fold):")
    print(cv_metrics.to_string(index=False))
    if regime_outputs is not None:
        print("\nRegime spike usage:", int(regime_outputs["use_spike"].sum()))
        print("Regime spike fallback:", bool(regime_outputs["spike_fallback"]))
        print("Regime holdout diagnostics:", regime_outputs["holdout_diag"])
    print("\nBoosting params:")
    print(models["boosting_params"])


if __name__ == "__main__":
    main()
