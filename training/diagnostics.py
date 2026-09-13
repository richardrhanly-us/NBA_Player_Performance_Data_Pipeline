"""
Diagnostic reporting for the V1 processed feature panel.

Pure functions over an already-built panel (as returned by
src.features.build_v1_features.build_v1_feature_panel, after
training.splits.assign_splits has added a SPLIT column) -- no I/O, no
network, nothing here mutates the panel. training/dataset.py calls these
and logs/prints the result; tests call them directly on small fixtures.
"""

from src.features import v1_schema

COLD_START_BUCKETS = (
    ("0", lambda n: n == 0),
    ("1-2", lambda n: (n >= 1) & (n <= 2)),
    ("3-4", lambda n: (n >= 3) & (n <= 4)),
    ("5-9", lambda n: (n >= 5) & (n <= 9)),
    ("10-19", lambda n: (n >= 10) & (n <= 19)),
    ("20+", lambda n: n >= 20),
)


def cold_start_histogram(panel_df):
    """Row counts bucketed by PRIOR_GAMES_THIS_SEASON, per the fixed
    buckets requested for judging the eligibility threshold."""
    prior = panel_df["PRIOR_GAMES_THIS_SEASON"]
    return {
        label: int(predicate(prior).sum()) for label, predicate in COLD_START_BUCKETS
    }


def duplicate_key_count(panel_df):
    return int(panel_df.duplicated(subset=["PLAYER_ID", "GAME_ID"]).sum())


def missing_value_rates(panel_df):
    feature_cols = list(v1_schema.V1_FEATURE_NAMES)
    present = [c for c in feature_cols if c in panel_df.columns]
    if not present or panel_df.empty:
        return {c: None for c in feature_cols}
    rates = panel_df[present].isna().mean().to_dict()
    return {c: round(float(rates.get(c, 1.0)), 4) for c in feature_cols}


def feature_value_summary(panel_df):
    """min/median/max per V1 feature, for spotting suspicious extreme values."""
    feature_cols = [c for c in v1_schema.V1_FEATURE_NAMES if c in panel_df.columns]
    summary = {}
    for col in feature_cols:
        series = panel_df[col].dropna()
        if series.empty:
            summary[col] = {"min": None, "median": None, "max": None}
        else:
            summary[col] = {
                "min": float(series.min()),
                "median": float(series.median()),
                "max": float(series.max()),
            }
    return summary


def build_diagnostics_report(panel_df, raw_row_counts_by_season: dict) -> dict:
    """
    Assembles the full diagnostics report: raw/processed/eligible row
    counts, exclusion counts, players represented, date range, rows per
    season/split, missing-value rates and min/median/max per feature,
    duplicate-key count, target PTS mean/std, and the cold-start
    histogram.
    """
    raw_rows_total = sum(raw_row_counts_by_season.values())
    processed_rows = len(panel_df)
    eligible_rows = int(panel_df["TRAINING_ELIGIBLE"].sum()) if processed_rows else 0
    excluded_rows = processed_rows - eligible_rows

    rows_per_season = (
        panel_df["SEASON"].value_counts().sort_index().to_dict()
        if processed_rows
        else {}
    )
    rows_per_split = (
        panel_df["SPLIT"].value_counts().to_dict()
        if "SPLIT" in panel_df.columns and processed_rows
        else {}
    )

    if processed_rows:
        date_min = str(panel_df["GAME_DATE"].min())
        date_max = str(panel_df["GAME_DATE"].max())
        players_represented = int(panel_df["PLAYER_ID"].nunique())
        target_mean = float(panel_df["PTS"].mean())
        target_std = float(panel_df["PTS"].std())
    else:
        date_min = date_max = None
        players_represented = 0
        target_mean = target_std = None

    return {
        "raw_rows_by_season": dict(raw_row_counts_by_season),
        "raw_rows_total": raw_rows_total,
        "processed_rows": processed_rows,
        "training_eligible_rows": eligible_rows,
        "rows_excluded_insufficient_history": excluded_rows,
        "players_represented": players_represented,
        "date_range": {"min": date_min, "max": date_max},
        "rows_per_season": {str(k): int(v) for k, v in rows_per_season.items()},
        "rows_per_split": {str(k): int(v) for k, v in rows_per_split.items()},
        "missing_value_rate_per_feature": missing_value_rates(panel_df),
        "feature_min_median_max": feature_value_summary(panel_df),
        "duplicate_key_count": duplicate_key_count(panel_df),
        "target_pts_mean": target_mean,
        "target_pts_std": target_std,
        "cold_start_histogram": cold_start_histogram(panel_df)
        if processed_rows
        else {},
    }


def format_diagnostics_report(report: dict) -> str:
    """Human-readable multi-line rendering for logging/console output."""
    lines = []
    lines.append("=== V1 Dataset Diagnostics ===")
    lines.append(f"Raw rows by season: {report['raw_rows_by_season']}")
    lines.append(f"Raw rows total: {report['raw_rows_total']}")
    lines.append(f"Processed rows: {report['processed_rows']}")
    lines.append(f"Training-eligible rows: {report['training_eligible_rows']}")
    lines.append(
        f"Rows excluded (insufficient history): {report['rows_excluded_insufficient_history']}"
    )
    lines.append(f"Players represented: {report['players_represented']}")
    lines.append(
        f"Date range: {report['date_range']['min']} -> {report['date_range']['max']}"
    )
    lines.append(f"Rows per season: {report['rows_per_season']}")
    lines.append(f"Rows per split: {report['rows_per_split']}")
    lines.append(
        f"Duplicate (PLAYER_ID, GAME_ID) count: {report['duplicate_key_count']}"
    )
    lines.append(
        f"Target PTS mean/std: {report['target_pts_mean']} / {report['target_pts_std']}"
    )
    lines.append(
        f"Cold-start histogram (prior games this season): {report['cold_start_histogram']}"
    )
    lines.append("Missing-value rate per feature:")
    for feature, rate in report["missing_value_rate_per_feature"].items():
        lines.append(f"  {feature}: {rate}")
    lines.append("Feature min / median / max:")
    for feature, stats in report["feature_min_median_max"].items():
        lines.append(f"  {feature}: {stats}")
    return "\n".join(lines)
