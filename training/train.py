"""
Trains the V1 RandomForestRegressor baseline on the leakage-safe historical
panel, evaluates it chronologically (train/validation/test), computes
simple pregame-average baselines on the same splits, and persists a
complete, versioned artifact set.

This is a fixed baseline, not a tuning run: hyperparameters
(training.config.V1_BASELINE_MODEL_PARAMS) are set once, deliberately
matching the currently deployed legacy model's configuration, and are not
adjusted based on anything this script observes. No GridSearchCV/
RandomizedSearchCV/manual tuning happens here.

Does not touch models/points_regression.pkl or any production code path --
see training/model_registry.py's module docstring.

Usage:
    python -m training.train
    python -m training.train --panel training/data/processed/v1_panel.parquet
    python -m training.train --version v1_baseline_manual --overwrite
"""

from __future__ import annotations

import argparse
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

from src.features import v1_schema
from training import config, evaluate, model_registry, splits

logger = logging.getLogger(__name__)

FORBIDDEN_COLUMNS = {
    "closing_line",
    "is_star",
    "sportsbook_line",
    "sportsbook",
    "odds",
    "edge",
    "bookmaker",
}

BASELINE_FEATURES = ("player_avg_pts", "last5_pts", "last10_pts")


def verify_native_nan_support() -> None:
    """
    RandomForestRegressor's ability to accept NaN input directly is a
    real but version-specific sklearn feature (input_tags.allow_nan),
    not something to assume. This is checked at the start of every
    training run, not just in the test suite, so an sklearn
    downgrade/change that removes this support fails loudly here rather
    than producing a confusing error mid-fit or silent wrong behavior.
    """
    probe_model = RandomForestRegressor(n_estimators=3, random_state=0)
    X = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0], [6.0, np.nan]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    try:
        probe_model.fit(X, y)
        probe_model.predict(np.array([[1.5, np.nan]]))
    except Exception as exc:
        raise RuntimeError(
            "RandomForestRegressor in this sklearn version "
            f"({sklearn.__version__}) does not support NaN input natively. "
            "The V1 training pipeline's missing-value policy assumes native "
            "support (see training/config.py and this function's docstring) "
            "and must be revisited (e.g. add median imputation fit on train "
            "only) before training can proceed."
        ) from exc


def validate_feature_contract(panel_df: pd.DataFrame) -> None:
    """
    Asserts, before any training happens:
      - every V1_FEATURE_NAMES column exists
      - closing_line/is_star/other sportsbook-adjacent columns are absent
    Feature ORDER is enforced by construction in select_feature_matrix()
    (which always selects list(V1_FEATURE_NAMES), never panel_df.columns),
    not by a separate check here.
    """
    missing = [f for f in v1_schema.V1_FEATURE_NAMES if f not in panel_df.columns]
    if missing:
        raise ValueError(
            f"Processed panel is missing required V1 feature(s): {missing}. "
            "Refusing to train against an incomplete feature contract."
        )

    present_forbidden = FORBIDDEN_COLUMNS & set(panel_df.columns)
    if present_forbidden:
        raise ValueError(
            f"Processed panel contains forbidden/sportsbook-adjacent column(s): "
            f"{present_forbidden}. The V1 model must never see these."
        )


def select_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """The ONE place training/inference-within-this-step selects features --
    always in the explicit V1_FEATURE_NAMES order, never inferred from
    whatever columns happen to be present."""
    return df[list(v1_schema.V1_FEATURE_NAMES)].copy()


def load_eligible_splits(panel_df: pd.DataFrame):
    """Returns (train_df, validation_df, test_df), each filtered to
    TRAINING_ELIGIBLE == 1 and the matching SPLIT label. Rows outside
    these splits (ineligible, or SPLIT == 'unassigned') are excluded from
    all three -- never trained on, never evaluated on here."""
    eligible = panel_df[panel_df["TRAINING_ELIGIBLE"] == 1]
    train_df = eligible[eligible["SPLIT"] == splits.SPLIT_TRAIN].reset_index(drop=True)
    val_df = eligible[eligible["SPLIT"] == splits.SPLIT_VALIDATION].reset_index(
        drop=True
    )
    test_df = eligible[eligible["SPLIT"] == splits.SPLIT_TEST].reset_index(drop=True)
    return train_df, val_df, test_df


def train_baseline_model(train_df: pd.DataFrame) -> RandomForestRegressor:
    X_train = select_feature_matrix(train_df)
    y_train = train_df[v1_schema.V1_TARGET_COLUMN]
    model = RandomForestRegressor(**config.V1_BASELINE_MODEL_PARAMS)
    model.fit(X_train, y_train)
    return model


def _git_commit_hash() -> str | None:
    """Best-effort provenance capture -- must never block training if git
    isn't available (e.g. a copied working directory with no .git)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=config.REPO_ROOT,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        logger.debug(
            "git rev-parse HEAD exited %d; recording no commit hash", result.returncode
        )
    except Exception as exc:  # noqa: BLE001 - metadata capture must never fail training
        logger.debug(
            "Could not determine git commit hash: %s: %s", type(exc).__name__, exc
        )
    return None


def default_version_name() -> str:
    return "v1_baseline_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _date_range(df: pd.DataFrame) -> dict | None:
    if df.empty:
        return None
    return {"min": str(df["GAME_DATE"].min()), "max": str(df["GAME_DATE"].max())}


def build_metadata(
    *, panel_path, train_df, val_df, test_df, training_start, training_end
) -> dict:
    return {
        "model_type": "RandomForestRegressor",
        "hyperparameters": dict(config.V1_BASELINE_MODEL_PARAMS),
        "target": v1_schema.V1_TARGET_COLUMN,
        "feature_schema": list(v1_schema.V1_FEATURE_NAMES),
        "missing_value_policy": (
            "native RandomForestRegressor NaN support "
            f"(sklearn {sklearn.__version__}, input_tags.allow_nan=True verified "
            "at training start) -- no imputation performed"
        ),
        "panel_path": str(panel_path),
        "split_boundary": {
            "train_seasons": list(splits.TRAIN_SEASONS),
            "validation_test_season": splits.VALIDATION_TEST_SEASON,
            "validation_test_boundary_date": str(splits.VALIDATION_TEST_BOUNDARY_DATE),
        },
        "eligible_row_counts": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "data_date_ranges": {
            "train": _date_range(train_df),
            "validation": _date_range(val_df),
            "test": _date_range(test_df),
        },
        "git_commit": _git_commit_hash(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "training_start_utc": training_start.isoformat(),
        "training_end_utc": training_end.isoformat(),
        "training_duration_seconds": (training_end - training_start).total_seconds(),
    }


def run_training(
    panel_path=None, version=None, overwrite: bool = False, output_dir=None
) -> dict:
    verify_native_nan_support()
    logger.info(
        "Native NaN support verified for RandomForestRegressor (sklearn %s)",
        sklearn.__version__,
    )

    panel_path = Path(panel_path) if panel_path else config.DEFAULT_PANEL_PATH
    logger.info("Loading processed panel from %s", panel_path)
    panel_df = pd.read_parquet(panel_path)
    logger.info(
        "Panel loaded: %d rows, %d columns", len(panel_df), len(panel_df.columns)
    )

    validate_feature_contract(panel_df)
    logger.info(
        "Feature contract validated: all %d V1 features present, no forbidden columns",
        len(v1_schema.V1_FEATURE_NAMES),
    )

    train_df, val_df, test_df = load_eligible_splits(panel_df)
    logger.info(
        "Eligible rows -- train: %d, validation: %d, test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    if train_df.empty:
        raise ValueError(
            "No eligible training rows found in the panel -- cannot train."
        )

    training_start = datetime.now(timezone.utc)
    logger.info(
        "Training RandomForestRegressor with params: %s",
        config.V1_BASELINE_MODEL_PARAMS,
    )
    model = train_baseline_model(train_df)
    training_end = datetime.now(timezone.utc)
    logger.info(
        "Training complete in %.1fs", (training_end - training_start).total_seconds()
    )

    split_frames = [("train", train_df), ("validation", val_df), ("test", test_df)]
    split_metrics = {}
    residual_frames = []

    for split_name, split_df in split_frames:
        if split_df.empty:
            logger.warning(
                "Split '%s' has no eligible rows -- skipping evaluation", split_name
            )
            continue
        X = select_feature_matrix(split_df)
        y_true = split_df[v1_schema.V1_TARGET_COLUMN]
        y_pred = model.predict(X)
        metrics = evaluate.compute_regression_metrics(y_true, y_pred)
        split_metrics[split_name] = metrics
        logger.info(
            "%s metrics: n=%d MAE=%.3f RMSE=%.3f R2=%.3f bias=%.3f",
            split_name,
            metrics["n"],
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
            metrics["bias_mean_pred_minus_actual"],
        )
        if split_name in ("validation", "test"):
            residual_frames.append(
                evaluate.build_residuals_df(split_df, y_true, y_pred)
            )

    residuals_df = (
        pd.concat(residual_frames, ignore_index=True)
        if residual_frames
        else pd.DataFrame()
    )
    segment_diagnostics = (
        evaluate.build_segment_diagnostics(residuals_df)
        if not residuals_df.empty
        else {}
    )

    simple_baselines = {}
    for feature in BASELINE_FEATURES:
        simple_baselines[feature] = {}
        for split_name, split_df in split_frames:
            if split_df.empty:
                continue
            simple_baselines[feature][split_name] = (
                evaluate.compute_simple_baseline_metrics(split_df, feature)
            )
        logger.info("Simple baseline '%s' computed for available splits", feature)

    feature_importances = dict(
        sorted(
            zip(
                v1_schema.V1_FEATURE_NAMES,
                (float(x) for x in model.feature_importances_),
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    metadata = build_metadata(
        panel_path=panel_path,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        training_start=training_start,
        training_end=training_end,
    )

    metrics_payload = {
        "train": split_metrics.get("train"),
        "validation": split_metrics.get("validation"),
        "test": split_metrics.get("test"),
        "simple_baselines": simple_baselines,
        "segment_diagnostics": segment_diagnostics,
    }

    version = version or default_version_name()
    if output_dir is not None:
        config.MODEL_REGISTRY_DIR = Path(output_dir)  # test/manual override only

    artifact_dir = model_registry.save_artifacts(
        version,
        model=model,
        metadata=metadata,
        metrics=metrics_payload,
        feature_importances=feature_importances,
        residuals_df=residuals_df,
        overwrite=overwrite,
    )
    model_registry.set_current_pointer(version)
    logger.info("Artifacts written to %s", artifact_dir)
    logger.info("CURRENT_V1 pointer updated to %s", version)

    return {
        "version": version,
        "artifact_dir": artifact_dir,
        "model": model,
        "metadata": metadata,
        "metrics": metrics_payload,
        "feature_importances": feature_importances,
        "residuals_df": residuals_df,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.train",
        description="Train and evaluate the V1 RandomForestRegressor baseline.",
    )
    parser.add_argument(
        "--panel",
        type=str,
        default=None,
        help=f"Path to the processed V1 panel (default: {config.DEFAULT_PANEL_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Model registry root directory (default: {config.MODEL_REGISTRY_DIR}).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Artifact version name (default: v1_baseline_<UTC timestamp>).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the artifact directory if it already exists.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    result = run_training(
        panel_path=args.panel,
        version=args.version,
        overwrite=args.overwrite,
        output_dir=args.output_dir,
    )

    logger.info("=== V1 baseline training complete: %s ===", result["version"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
