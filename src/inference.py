from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import resolve_market_policy
from src.mcp_pipeline import (
    feature_engineering,
    predict_regime,
    inverse_transform_target,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _resolve_output_path(output: str, deploy_dir: Path) -> Path:
    output_path = Path(output)
    if not output_path.is_absolute():
        return deploy_dir / output_path
    return output_path


def run_inference(input_path: str, output_path: str, deploy_dir: str) -> Path:
    deploy_path = Path(deploy_dir)
    feature_path = deploy_path / "feature_list.json"
    metadata_path = deploy_path / "model_metadata.json"

    feature_payload = _load_json(feature_path)
    metadata = _load_json(metadata_path)

    feature_list = feature_payload["feature_list"]
    feature_config = metadata["feature_config"]
    required_raw_columns = metadata.get("required_raw_columns", [])
    model_files = metadata["model_files"]

    input_df = pd.read_csv(input_path)
    _validate_columns(input_df, required_raw_columns)

    market_policy = resolve_market_policy(feature_config["target_market"])
    engineered_df, _, _, _ = feature_engineering(
        input_df,
        datetime_col=feature_config["datetime_col"],
        target_market=feature_config["target_market"],
        target_col=feature_config["target_col"],
        market_columns=feature_config["market_columns"],
        lead_steps=feature_config["lead_steps"],
        target_lags=feature_config["target_lags"],
        rolling_windows=feature_config["rolling_windows"],
        cross_market_lags=feature_config["cross_market_lags"],
        renewable_cols=feature_config["renewable_cols"],
        buy_col=feature_config["buy_col"],
        sell_col=feature_config["sell_col"],
        solar_col=feature_config["solar_col"],
        weather_cols=feature_config["weather_cols"],
        market_policy=market_policy,
        auction_cutoff_hour=feature_config["auction_cutoff_hour"],
        auction_cutoff_minute=feature_config["auction_cutoff_minute"],
        dropna_target=False,
    )

    missing_features = [col for col in feature_list if col not in engineered_df.columns]
    if missing_features:
        raise ValueError(
            "Missing engineered features required by the model: "
            + ", ".join(missing_features)
        )

    X = engineered_df[feature_list]

    models_dir = deploy_path / "models"
    classifier = joblib.load(models_dir / model_files["regime_classifier"])
    low_model = joblib.load(models_dir / model_files["regime_low"])
    mid_model = joblib.load(models_dir / model_files["regime_mid"])
    high_model = joblib.load(models_dir / model_files["regime_high"])
    # Optional quantile models for deployed outputs
    p10_model = joblib.load(models_dir / model_files.get("p10")) if model_files.get("p10") else None
    p90_model = joblib.load(models_dir / model_files.get("p90")) if model_files.get("p90") else None

    preds, regime_prob, _ = predict_regime(
        X,
        classifier,
        low_model,
        mid_model,
        high_model,
        feature_config["regime_prob_threshold"],
        feature_config["target_transform"],
        feature_config["regime_bidirectional"],
    )

    # Compute optional quantile predictions if models are available
    p10_preds = None
    p90_preds = None
    if p10_model is not None:
        p10_preds = inverse_transform_target(p10_model.predict(X), feature_config["target_transform"])
    if p90_model is not None:
        p90_preds = inverse_transform_target(p90_model.predict(X), feature_config["target_transform"])

    output_payload = {
        "timestamp": pd.to_datetime(engineered_df[feature_config["datetime_col"]]).values,
        "prediction": preds,
        "p10": p10_preds,
        "p90": p90_preds,
        "regime_probability": regime_prob,
    }

    output_df = pd.DataFrame(
        output_payload,
        columns=["timestamp", "prediction", "p10", "p90", "regime_probability"],
    )

    output_file = _resolve_output_path(output_path, deploy_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_file, index=False)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GDAM inference")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Path to output predictions CSV",
    )
    parser.add_argument(
        "--deploy-dir",
        default=str(Path(__file__).resolve().parent),
        help="Path to deploy directory",
    )
    args = parser.parse_args()

    output_path = run_inference(args.input, args.output, args.deploy_dir)
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
