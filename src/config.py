"""Configuration types, market specs, and config parsing for the MCP pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MarketSpec:
    lead_steps: int
    target_lags: list[int]
    rolling_windows: list[int]
    cross_market_lags: list[int]
    spike_baseline_lag: int


@dataclass(frozen=True)
class MarketPolicy:
    allowed_groups: set[str]
    allowed_cross_markets: set[str]


@dataclass
class Config:
    input_csv: Path
    output_dir: Path
    datetime_col: str
    target_market: str
    target_col: str
    market_columns: dict[str, str]
    lead_steps: int
    target_lags: list[int]
    rolling_windows: list[int]
    cross_market_lags: list[int]
    spike_baseline_lag: int
    renewable_cols: list[str]
    buy_col: str
    sell_col: str
    solar_col: str
    weather_cols: list
    test_size: float
    n_splits: int
    gbm_backend: str
    quantile_backend: str
    tune_gbm: bool
    enable_spike_hybrid: bool
    spike_threshold: float
    spike_adjustment: float
    plot_points: int
    target_transform: str
    weight_quantile: float
    weight_multiplier: float
    debug_fold: int | None
    regime_quantile: float
    regime_extreme_quantile: float
    regime_prob_threshold: float
    regime_min_samples: int
    regime_spike_weight: float
    regime_strategy: str
    regime_classes: int
    regime_zscore_threshold: float
    regime_vol_multiplier: float
    regime_baseline_lag: int
    regime_bidirectional: bool
    regime_valley_quantile: float | None
    compare_regime_strategies: bool
    feature_include: list[str]
    save_deployment_artifacts: bool
    rolling_window_days: int
    rolling_step_days: int
    calibration_bins: int
    spike_penalty_multiplier: float
    underprediction_multiplier: float
    auction_cutoff_hour: int
    auction_cutoff_minute: int


MARKET_SPECS = {
    "DAM": MarketSpec(
        lead_steps=96,
        target_lags=[96, 192],
        rolling_windows=[96],
        cross_market_lags=[96, 192],
        spike_baseline_lag=96,
    ),
    "GDAM": MarketSpec(
        lead_steps=96,
        target_lags=[96, 192],
        rolling_windows=[96],
        cross_market_lags=[96, 192],
        spike_baseline_lag=96,
    ),
    "RTM": MarketSpec(
        lead_steps=4,
        target_lags=[4, 8, 12, 96],
        rolling_windows=[16, 96],
        cross_market_lags=[4, 8, 12, 96],
        spike_baseline_lag=4,
    ),
}

MARKET_CUTOFFS = {
    "DAM": (12, 0),
    "GDAM": (12, 0),
}


def parse_int_list(value: str | None) -> list[int]:
    if value is None:
        return []
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in items]


def parse_str_list(value: str | None) -> list[str]:
    if value is None:
        return []
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items


def parse_market_columns(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    mapping: dict[str, str] = {}
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            key, col = chunk.split("=", 1)
        elif ":" in chunk:
            key, col = chunk.split(":", 1)
        else:
            raise ValueError(
                f"Invalid market column mapping '{chunk}'. Use DAM=col,GDAM=col,RTM=col."
            )
        mapping[key.strip().upper()] = col.strip()
    return mapping


def infer_market_column(columns: list[str], market: str) -> str | None:
    candidates = [
        f"{market.lower()}_price",
        f"price_{market.lower()}",
        f"{market.lower()}_mcp",
        f"mcp_{market.lower()}",
    ]
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def infer_renewable_columns(columns: list[str]) -> list[str]:
    renewable_tokens = ("solar", "wind", "hydro", "renewable", "bio")
    unit_tokens = ("mw", "mwh", "gen", "generation", "output")
    inferred = []
    for col in columns:
        lower = col.lower()
        if any(token in lower for token in renewable_tokens) and any(
            token in lower for token in unit_tokens
        ):
            inferred.append(col)
    return inferred


def resolve_market_columns(
    columns: list[str],
    target_market: str,
    target_col: str,
    market_columns: dict[str, str],
) -> dict[str, str]:
    resolved = {key.upper(): value for key, value in market_columns.items()}
    resolved[target_market] = target_col
    for market in MARKET_SPECS.keys():
        if market not in resolved:
            inferred = infer_market_column(columns, market)
            if inferred:
                resolved[market] = inferred
    if target_market not in resolved or resolved[target_market] not in columns:
        raise ValueError(
            f"Target column '{target_col}' not found for market {target_market}."
        )
    return {market: col for market, col in resolved.items() if col in columns}


def resolve_market_policy(target_market: str) -> MarketPolicy:
    policies = {
        "DAM": MarketPolicy(
            allowed_groups={
                "temporal",
                "target_lag",
                "target_roll",
                "market_state",
                "weather",
                "cross_market_lag",
                "cross_market_spread",
                "cross_market_vol",
            },
            allowed_cross_markets={"GDAM", "RTM"},
        ),
        "GDAM": MarketPolicy(
            allowed_groups={
                "temporal",
                "target_lag",
                "target_roll",
                "market_state",
                "renewable",
                "weather",
                "cross_market_lag",
                "cross_market_spread",
                "cross_market_vol",
            },
            allowed_cross_markets={"DAM", "RTM"},
        ),
        "RTM": MarketPolicy(
            allowed_groups={
                "temporal",
                "target_lag",
                "target_roll",
                "weather",
                "cross_market_lag",
                "cross_market_spread",
                "cross_market_vol",
            },
            allowed_cross_markets={"DAM", "GDAM"},
        ),
    }
    policy = policies.get(target_market.upper())
    if policy is None:
        raise ValueError(f"Unknown market policy for target {target_market}.")
    return policy


def _coerce_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items
    return [value]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Multi-market MCP price prediction pipeline"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output-dir", default="outputs", help="Directory for outputs")
    parser.add_argument("--datetime-col", default="datetime")
    parser.add_argument(
        "--target-market",
        choices=sorted(MARKET_SPECS.keys()),
        default="GDAM",
        help="Target market for forecasting (DAM, GDAM, RTM)",
    )
    parser.add_argument(
        "--target-col",
        "--price-col",
        dest="target_col",
        default=None,
        help="Target price column for the selected market",
    )
    parser.add_argument(
        "--market-columns",
        default="",
        help="Comma-separated market mappings, e.g. DAM=dam_price,GDAM=gdam_price,RTM=rtm_price",
    )
    parser.add_argument(
        "--lead-steps",
        type=int,
        default=None,
        help="Minimum lookback steps enforced for leakage-safe features",
    )
    parser.add_argument(
        "--target-lags",
        default=None,
        help="Comma-separated lag steps for target market features",
    )
    parser.add_argument(
        "--rolling-windows",
        default=None,
        help="Comma-separated window sizes for target market features",
    )
    parser.add_argument(
        "--cross-market-lags",
        default=None,
        help="Comma-separated lag steps for cross-market features",
    )
    parser.add_argument(
        "--spike-baseline-lag",
        type=int,
        default=None,
        help="Lag step used as the baseline for spike hybrid labels",
    )
    parser.add_argument(
        "--renewable-cols",
        default="",
        help="Comma-separated renewable generation columns for mix ratios",
    )
    parser.add_argument("--buy-col", default="buy_mw")
    parser.add_argument("--sell-col", default="sell_mw")
    parser.add_argument("--solar-col", default="solar")
    parser.add_argument(
        "--weather-cols",
        default="temp,solar,cloud,wind,humidity,rain",
        help="Comma-separated list of weather columns to include",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--gbm-backend", choices=["lightgbm", "xgboost"], default="lightgbm")
    parser.add_argument("--quantile-backend", choices=["lightgbm", "sklearn"], default="lightgbm")
    parser.add_argument("--no-gbm-tuning", action="store_true")
    parser.add_argument("--enable-spike-hybrid", action="store_true")
    parser.add_argument("--spike-threshold", type=float, default=50.0)
    parser.add_argument("--spike-adjustment", type=float, default=0.3)
    parser.add_argument("--plot-points", type=int, default=500)
    parser.add_argument("--use-log-target", action="store_true")
    parser.add_argument(
        "--target-transform",
        choices=["none", "log1p", "asinh"],
        default="none",
        help="Target transform mode (none, log1p, asinh)",
    )
    parser.add_argument("--weight-quantile", type=float, default=0.9)
    parser.add_argument("--weight-multiplier", type=float, default=3.0)
    parser.add_argument("--debug-fold", type=int, default=None)
    parser.add_argument("--regime-quantile", type=float, default=0.9)
    parser.add_argument("--regime-extreme-quantile", type=float, default=0.98)
    parser.add_argument("--regime-prob-threshold", type=float, default=0.3)
    parser.add_argument("--regime-min-samples", type=int, default=50)
    parser.add_argument("--regime-spike-weight", type=float, default=5.0)
    parser.add_argument(
        "--regime-strategy",
        default="rolling_quantile",
        choices=[
            "global",
            "daily",
            "rolling",
            "rolling_quantile",
            "zscore",
            "vol_adj",
            "hybrid",
        ],
    )
    parser.add_argument(
        "--regime-zscore-threshold",
        type=float,
        default=1.28,
        help="Z-score threshold for spike regime (default: 1.28)",
    )
    parser.add_argument(
        "--regime-vol-multiplier",
        type=float,
        default=2.0,
        help="Volatility multiplier for spike regime (default: 2.0)",
    )
    parser.add_argument(
        "--regime-baseline-lag",
        type=int,
        default=None,
        help="Lag used as baseline for volatility-adjusted thresholds",
    )
    parser.add_argument(
        "--regime-bidirectional",
        action="store_true",
        help="Enable bidirectional regimes (valley, normal, spike)",
    )
    parser.add_argument(
        "--regime-valley-quantile",
        type=float,
        default=None,
        help="Lower quantile threshold for valleys (default: 1 - regime_quantile)",
    )
    parser.add_argument("--regime-classes", type=int, default=3)
    parser.add_argument("--compare-regime-strategies", action="store_true")
    parser.add_argument("--rolling-window-days", type=int, default=30)
    parser.add_argument("--rolling-step-days", type=int, default=7)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--spike-penalty-multiplier", type=float, default=2.0)
    parser.add_argument("--underprediction-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--auction-cutoff-hour",
        type=int,
        default=None,
        help="Auction gate closure hour (24h clock) for DAM/GDAM forecasts",
    )
    parser.add_argument(
        "--auction-cutoff-minute",
        type=int,
        default=None,
        help="Auction gate closure minute for DAM/GDAM forecasts",
    )
    parser.add_argument(
        "--feature-include",
        default=None,
        help="Comma-separated list of engineered features to keep",
    )
    parser.add_argument(
        "--save-deployment-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write deploy/* artifacts after training",
    )

    args = parser.parse_args()
    if args.use_log_target and args.target_transform == "none":
        args.target_transform = "log1p"
    if args.regime_bidirectional:
        if args.regime_valley_quantile is None:
            args.regime_valley_quantile = 1.0 - args.regime_quantile
        args.regime_classes = 3

    target_market = args.target_market.upper()
    market_spec = MARKET_SPECS[target_market]
    cutoff_defaults = MARKET_CUTOFFS.get(target_market, (12, 0))
    weather_cols = [c.strip() for c in args.weather_cols.split(",") if c.strip()]
    target_lags = (
        parse_int_list(args.target_lags) if args.target_lags else market_spec.target_lags
    )
    rolling_windows = (
        parse_int_list(args.rolling_windows)
        if args.rolling_windows
        else market_spec.rolling_windows
    )
    cross_market_lags = (
        parse_int_list(args.cross_market_lags)
        if args.cross_market_lags
        else market_spec.cross_market_lags
    )
    lead_steps = (
        args.lead_steps if args.lead_steps is not None else market_spec.lead_steps
    )
    spike_baseline_lag = (
        args.spike_baseline_lag
        if args.spike_baseline_lag is not None
        else market_spec.spike_baseline_lag
    )
    spike_baseline_lag = max(spike_baseline_lag, lead_steps)
    if spike_baseline_lag not in target_lags:
        target_lags = sorted(set(target_lags + [spike_baseline_lag]))
    market_columns = parse_market_columns(args.market_columns)
    target_col = args.target_col or market_columns.get(target_market) or "price"
    renewable_cols = parse_str_list(args.renewable_cols)
    return Config(
        input_csv=Path(args.input),
        output_dir=Path(args.output_dir),
        datetime_col=args.datetime_col,
        target_market=target_market,
        target_col=target_col,
        market_columns=market_columns,
        lead_steps=lead_steps,
        target_lags=target_lags,
        rolling_windows=rolling_windows,
        cross_market_lags=cross_market_lags,
        spike_baseline_lag=spike_baseline_lag,
        renewable_cols=renewable_cols,
        buy_col=args.buy_col,
        sell_col=args.sell_col,
        solar_col=args.solar_col,
        weather_cols=weather_cols,
        test_size=args.test_size,
        n_splits=args.n_splits,
        gbm_backend=args.gbm_backend,
        quantile_backend=args.quantile_backend,
        tune_gbm=not args.no_gbm_tuning,
        enable_spike_hybrid=args.enable_spike_hybrid,
        spike_threshold=args.spike_threshold,
        spike_adjustment=args.spike_adjustment,
        plot_points=args.plot_points,
        target_transform=args.target_transform,
        weight_quantile=args.weight_quantile,
        weight_multiplier=args.weight_multiplier,
        debug_fold=args.debug_fold,
        regime_quantile=args.regime_quantile,
        regime_extreme_quantile=args.regime_extreme_quantile,
        regime_prob_threshold=args.regime_prob_threshold,
        regime_min_samples=args.regime_min_samples,
        regime_spike_weight=args.regime_spike_weight,
        regime_strategy=args.regime_strategy,
        regime_classes=args.regime_classes,
        regime_zscore_threshold=args.regime_zscore_threshold,
        regime_vol_multiplier=args.regime_vol_multiplier,
        regime_baseline_lag=(
            args.regime_baseline_lag
            if args.regime_baseline_lag is not None
            else spike_baseline_lag
        ),
        regime_bidirectional=args.regime_bidirectional,
        regime_valley_quantile=args.regime_valley_quantile,
        compare_regime_strategies=args.compare_regime_strategies,
        rolling_window_days=args.rolling_window_days,
        rolling_step_days=args.rolling_step_days,
        calibration_bins=args.calibration_bins,
        spike_penalty_multiplier=args.spike_penalty_multiplier,
        underprediction_multiplier=args.underprediction_multiplier,
        feature_include=parse_str_list(args.feature_include),
        save_deployment_artifacts=bool(args.save_deployment_artifacts),
        auction_cutoff_hour=(
            args.auction_cutoff_hour
            if args.auction_cutoff_hour is not None
            else cutoff_defaults[0]
        ),
        auction_cutoff_minute=(
            args.auction_cutoff_minute
            if args.auction_cutoff_minute is not None
            else cutoff_defaults[1]
        ),
    )


def config_from_dict(data: dict) -> Config:
    if not data or "input_csv" not in data:
        raise ValueError("Config must include input_csv.")

    target_market = str(data.get("target_market", "GDAM")).upper()
    market_spec = MARKET_SPECS[target_market]
    cutoff_defaults = MARKET_CUTOFFS.get(target_market, (12, 0))
    market_columns = {
        str(key).upper(): str(val)
        for key, val in (data.get("market_columns") or {}).items()
    }
    target_col = data.get("target_col") or market_columns.get(target_market) or "price"
    weather_cols = _coerce_list(data.get("weather_cols", "temp,solar,cloud,wind,humidity,rain"))
    weather_cols = [str(col) for col in weather_cols]

    target_lags = _coerce_list(data.get("target_lags", market_spec.target_lags))
    target_lags = [int(lag) for lag in target_lags]
    rolling_windows = _coerce_list(data.get("rolling_windows", market_spec.rolling_windows))
    rolling_windows = [int(window) for window in rolling_windows]
    cross_market_lags = _coerce_list(data.get("cross_market_lags", market_spec.cross_market_lags))
    cross_market_lags = [int(lag) for lag in cross_market_lags]

    lead_steps = int(data.get("lead_steps", market_spec.lead_steps))
    spike_baseline_lag = int(data.get("spike_baseline_lag", market_spec.spike_baseline_lag))
    spike_baseline_lag = max(spike_baseline_lag, lead_steps)
    if spike_baseline_lag not in target_lags:
        target_lags = sorted(set(target_lags + [spike_baseline_lag]))

    renewable_cols = _coerce_list(data.get("renewable_cols", []))
    renewable_cols = [str(col) for col in renewable_cols]

    regime_baseline_lag = int(data.get("regime_baseline_lag", spike_baseline_lag))
    regime_bidirectional = bool(data.get("regime_bidirectional", False))
    regime_valley_quantile_raw = data.get("regime_valley_quantile")
    target_transform = str(data.get("target_transform", "none")).lower()
    if target_transform == "none" and bool(data.get("use_log_target", False)):
        target_transform = "log1p"
    if target_transform not in {"none", "log1p", "asinh"}:
        raise ValueError(
            "target_transform must be one of: none, log1p, asinh"
        )

    regime_quantile = float(data.get("regime_quantile", 0.9))
    regime_extreme_quantile = float(data.get("regime_extreme_quantile", 0.98))
    regime_classes = int(data.get("regime_classes", 3))
    if regime_bidirectional:
        if regime_valley_quantile_raw is None:
            regime_valley_quantile = 1.0 - regime_quantile
        else:
            regime_valley_quantile = float(regime_valley_quantile_raw)
        regime_classes = 3
    else:
        regime_valley_quantile = (
            float(regime_valley_quantile_raw)
            if regime_valley_quantile_raw is not None
            else None
        )

    return Config(
        input_csv=Path(str(data["input_csv"])),
        output_dir=Path(str(data.get("output_dir", "outputs"))),
        datetime_col=str(data.get("datetime_col", "datetime")),
        target_market=target_market,
        target_col=str(target_col),
        market_columns=market_columns,
        lead_steps=lead_steps,
        target_lags=target_lags,
        rolling_windows=rolling_windows,
        cross_market_lags=cross_market_lags,
        spike_baseline_lag=spike_baseline_lag,
        renewable_cols=renewable_cols,
        buy_col=str(data.get("buy_col", "buy_mw")),
        sell_col=str(data.get("sell_col", "sell_mw")),
        solar_col=str(data.get("solar_col", "solar")),
        weather_cols=weather_cols,
        test_size=float(data.get("test_size", 0.2)),
        n_splits=int(data.get("n_splits", 5)),
        gbm_backend=str(data.get("gbm_backend", "lightgbm")),
        quantile_backend=str(data.get("quantile_backend", "lightgbm")),
        tune_gbm=bool(data.get("tune_gbm", True)),
        enable_spike_hybrid=bool(data.get("enable_spike_hybrid", False)),
        spike_threshold=float(data.get("spike_threshold", 50.0)),
        spike_adjustment=float(data.get("spike_adjustment", 0.3)),
        plot_points=int(data.get("plot_points", 500)),
        target_transform=target_transform,
        weight_quantile=float(data.get("weight_quantile", 0.9)),
        weight_multiplier=float(data.get("weight_multiplier", 3.0)),
        debug_fold=data.get("debug_fold"),
        regime_quantile=regime_quantile,
        regime_extreme_quantile=regime_extreme_quantile,
        regime_prob_threshold=float(data.get("regime_prob_threshold", 0.3)),
        regime_min_samples=int(data.get("regime_min_samples", 50)),
        regime_spike_weight=float(data.get("regime_spike_weight", 5.0)),
        regime_strategy=str(data.get("regime_strategy", "rolling_quantile")),
        regime_classes=regime_classes,
        regime_zscore_threshold=float(data.get("regime_zscore_threshold", 1.28)),
        regime_vol_multiplier=float(data.get("regime_vol_multiplier", 2.0)),
        regime_baseline_lag=regime_baseline_lag,
        regime_bidirectional=regime_bidirectional,
        regime_valley_quantile=regime_valley_quantile,
        compare_regime_strategies=bool(data.get("compare_regime_strategies", False)),
        feature_include=[str(value) for value in _coerce_list(data.get("feature_include", []))],
        save_deployment_artifacts=bool(data.get("save_deployment_artifacts", True)),
        rolling_window_days=int(data.get("rolling_window_days", 30)),
        rolling_step_days=int(data.get("rolling_step_days", 7)),
        calibration_bins=int(data.get("calibration_bins", 10)),
        spike_penalty_multiplier=float(data.get("spike_penalty_multiplier", 2.0)),
        underprediction_multiplier=float(data.get("underprediction_multiplier", 2.0)),
        auction_cutoff_hour=int(data.get("auction_cutoff_hour", cutoff_defaults[0])),
        auction_cutoff_minute=int(data.get("auction_cutoff_minute", cutoff_defaults[1])),
    )
