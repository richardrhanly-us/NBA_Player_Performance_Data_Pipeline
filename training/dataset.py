"""
Builds the V1 processed feature panel from the persisted raw historical
gamelogs, assigns chronological train/validation/test splits, and persists
the result.

Does not train anything -- see the module docstring on
src/features/build_v1_features.py for the feature-construction pipeline
this orchestrates, and training/splits.py for the split logic.

Usage:
    python -m training.dataset build
    python -m training.dataset build --season 2023-24 2024-25
    python -m training.dataset build --output training/data/processed/v1_panel.parquet
    python -m training.dataset build --overwrite
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.features import build_v1_features
from training import config, diagnostics, splits
from training.data import storage

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = config.PROCESSED_DATA_DIR / "v1_panel.parquet"


def load_raw_seasons(seasons=None) -> tuple[pd.DataFrame, dict]:
    """
    Loads and concatenates the persisted raw gamelogs for `seasons`
    (default: every season in training.config.TRAINING_SEASONS), via
    training.data.storage -- never re-fetches from stats.nba.com. Returns
    (combined_df, {season: raw_row_count}).
    """
    seasons = list(seasons) if seasons else list(config.TRAINING_SEASONS)
    raw_row_counts = {}
    frames = []

    for season in seasons:
        logger.info("Loading persisted raw gamelogs for season %s ...", season)
        season_df = storage.load_season_gamelogs(season)
        raw_row_counts[season] = len(season_df)
        logger.info("Season %s: %d raw rows loaded", season, len(season_df))
        if not season_df.empty:
            frames.append(season_df)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    logger.info("Total raw rows across %d season(s): %d", len(seasons), len(combined))
    return combined, raw_row_counts


def build_dataset(seasons=None) -> tuple[pd.DataFrame, dict]:
    """
    Full non-persisting pipeline: load raw seasons -> build the V1 feature
    panel -> assign splits -> compute diagnostics. Returns (panel_df, report).
    """
    raw_df, raw_row_counts = load_raw_seasons(seasons)

    if raw_df.empty:
        logger.warning(
            "No raw data found for the requested season(s); nothing to build."
        )
        empty_panel = pd.DataFrame()
        return empty_panel, diagnostics.build_diagnostics_report(
            empty_panel, raw_row_counts
        )

    logger.info("Building V1 feature panel from %d raw rows ...", len(raw_df))
    panel = build_v1_features.build_v1_feature_panel(raw_df)
    logger.info("Feature panel built: %d rows", len(panel))

    logger.info("Assigning chronological train/validation/test splits ...")
    panel = splits.assign_splits(panel)
    split_counts = panel["SPLIT"].value_counts().to_dict()
    logger.info("Split counts: %s", split_counts)

    report = diagnostics.build_diagnostics_report(panel, raw_row_counts)
    return panel, report


def save_dataset(panel: pd.DataFrame, output_path=None, overwrite: bool = False):
    """
    Persists `panel` to `output_path` (default: DEFAULT_OUTPUT_PATH). A
    relative `output_path` is resolved against the repository root (the
    parent of training/), matching the CLI's --output examples. Returns
    the resolved Path written to.
    """
    output_path = output_path or DEFAULT_OUTPUT_PATH
    if not output_path.is_absolute():
        output_path = config.TRAINING_ROOT.parent / output_path

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass overwrite=True / --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".parquet.tmp")
    panel.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)
    logger.info("Processed panel written to %s (%d rows)", output_path, len(panel))
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.dataset",
        description="Build the V1 processed feature panel from persisted raw gamelogs.",
    )
    subparsers = parser.add_subparsers(dest="command")
    build_parser = subparsers.add_parser(
        "build", help="Build and persist the processed panel."
    )
    build_parser.add_argument(
        "--season",
        nargs="+",
        default=None,
        metavar="SEASON",
        help=f"Seasons to include (default: all of {', '.join(config.TRAINING_SEASONS)}).",
    )
    build_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output Parquet path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    build_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    build_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # "python -m training.dataset" with no subcommand behaves like "build".
    command = args.command or "build"

    logging.basicConfig(
        level=getattr(logging, getattr(args, "log_level", "INFO")),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if command != "build":
        parser.error(f"Unknown command: {command}")
        return 2

    output_path = (
        Path(args.output) if getattr(args, "output", None) else DEFAULT_OUTPUT_PATH
    )

    panel, report = build_dataset(seasons=getattr(args, "season", None))

    if panel.empty:
        logger.warning("Nothing to persist -- panel is empty.")
        return 1

    save_dataset(
        panel, output_path=output_path, overwrite=getattr(args, "overwrite", False)
    )

    logger.info("%s", diagnostics.format_diagnostics_report(report))
    logger.info("=== Dataset build complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
