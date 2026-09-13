"""
Versioned artifact persistence for training-side models.

This registry is entirely separate from, and does not touch,
models/points_regression.pkl (the deployed legacy model) or anything the
Streamlit apps load. It exists so training/train.py has a reproducible,
inspectable place to write a model + its metadata/metrics/feature
importances/residuals, and so a "current" pointer can name the latest V1
artifact without that pointer being read by any production code path.

Layout, under training.config.MODEL_REGISTRY_DIR (models/registry/ at the
repo root, alongside -- but never replacing -- points_regression.pkl):

    <version>/
        model.pkl               -- gitignored (large, regenerable)
        metadata.json            -- committed
        metrics.json              -- committed
        feature_importances.json -- committed
        residuals.parquet        -- gitignored (large, regenerable)
    CURRENT_V1              -- plain text file naming the active <version>;
                               MACHINE-LOCAL ONLY, gitignored. It points at
                               a model.pkl that only exists on the machine
                               that trained it, so committing it would let
                               a clean clone see a "current" version whose
                               model artifact was never shipped -- a
                               registry that looks usable but isn't. This
                               will change once a later step introduces
                               real artifact storage; until then, treat
                               CURRENT_V1 as something you regenerate
                               locally via `python -m training.train`,
                               never something you expect to find already
                               set after cloning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from training import config

logger = logging.getLogger(__name__)

CURRENT_POINTER_NAME = "CURRENT_V1"


def version_dir(version: str) -> Path:
    return config.MODEL_REGISTRY_DIR / version


def current_pointer_path() -> Path:
    return config.MODEL_REGISTRY_DIR / CURRENT_POINTER_NAME


def save_artifacts(
    version: str,
    *,
    model,
    metadata: dict,
    metrics: dict,
    feature_importances: dict,
    residuals_df: pd.DataFrame,
    overwrite: bool = False,
) -> Path:
    """
    Persists all five artifacts for one training run under
    models/registry/<version>/. Raises FileExistsError if the directory
    already exists and overwrite=False.
    """
    target_dir = version_dir(version)
    if target_dir.exists() and not overwrite:
        raise FileExistsError(
            f"{target_dir} already exists. Pass overwrite=True / --overwrite to replace it."
        )
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = target_dir / "model.pkl"
    joblib.dump(model, model_path)
    logger.info("Wrote %s", model_path)

    metadata_path = target_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Wrote %s", metadata_path)

    metrics_path = target_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Wrote %s", metrics_path)

    importances_path = target_dir / "feature_importances.json"
    with open(importances_path, "w", encoding="utf-8") as f:
        json.dump(feature_importances, f, indent=2, default=str)
    logger.info("Wrote %s", importances_path)

    residuals_path = target_dir / "residuals.parquet"
    residuals_df.to_parquet(residuals_path, index=False)
    logger.info("Wrote %s (%d rows)", residuals_path, len(residuals_df))

    return target_dir


def set_current_pointer(version: str) -> Path:
    """
    Writes the CURRENT_V1 pointer file. Training-side only: nothing in
    apps/ or src/shared_app.py reads this file, so this has no effect on
    the deployed application. Machine-local and gitignored -- it names a
    version whose model.pkl exists only on this machine, so it is never
    committed (see the module docstring / .gitignore comment). Regenerate
    it locally by rerunning `python -m training.train`.
    """
    pointer_path = current_pointer_path()
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = pointer_path.with_suffix(".tmp")
    tmp_path.write_text(version, encoding="utf-8")
    tmp_path.replace(pointer_path)
    logger.info("CURRENT_V1 now points to %s", version)
    return pointer_path


def get_current_version() -> str | None:
    pointer_path = current_pointer_path()
    if not pointer_path.exists():
        return None
    return pointer_path.read_text(encoding="utf-8").strip()


def load_model(version: str):
    return joblib.load(version_dir(version) / "model.pkl")


def load_metadata(version: str) -> dict:
    with open(version_dir(version) / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def load_metrics(version: str) -> dict:
    with open(version_dir(version) / "metrics.json", encoding="utf-8") as f:
        return json.load(f)


def load_feature_importances(version: str) -> dict:
    with open(version_dir(version) / "feature_importances.json", encoding="utf-8") as f:
        return json.load(f)


def load_residuals(version: str) -> pd.DataFrame:
    return pd.read_parquet(version_dir(version) / "residuals.parquet")
