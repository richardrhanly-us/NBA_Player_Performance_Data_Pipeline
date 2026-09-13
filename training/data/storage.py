"""
Raw-data persistence for the historical gamelog collector.

Layout on disk, under training.config.RAW_DATA_DIR:

    <season>/
        players/
            <player_id>.parquet   -- one player's deduplicated raw rows
        _manifest.json            -- progress checkpoint + run summary

The manifest exists for fast lookups and human-readable reporting, but it is
never the sole source of truth: `is_player_collected()` always verifies the
player's actual Parquet file is present and readable. A crash between
writing a player's file and updating the manifest can therefore never cause
either data loss or a false "already collected" skip.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from training import config

logger = logging.getLogger(__name__)

# Columns persisted for every raw gamelog row. This is deliberately wider
# than the 24 features the legacy model uses -- it preserves every native
# PlayerGameLog column plus a small set of lossless, deterministic
# derivations (own team / opponent / home-away parsed out of MATCHUP) and
# collection provenance (SEASON, PLAYER_NAME). Rolling windows, GmSc,
# usage proxy, and every other engineered feature are training-pipeline
# concerns and are intentionally NOT computed here.
RAW_GAMELOG_COLUMNS = (
    "SEASON",
    "SEASON_ID",
    "PLAYER_ID",
    "PLAYER_NAME",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "TEAM_ABBREVIATION",
    "OPPONENT_ABBREVIATION",
    "IS_HOME",
    "WL",
    "MIN",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
    "PLUS_MINUS",
    "VIDEO_AVAILABLE",
)

# Natural uniqueness key for one player-game row. A rerun (including a
# forced re-fetch of a player already on disk) must never produce two rows
# for the same (player, game).
UNIQUE_KEY = ("PLAYER_ID", "GAME_ID")

_NUMERIC_COLUMNS = (
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
    "PLUS_MINUS",
)


def season_dir(season: str) -> Path:
    return config.RAW_DATA_DIR / season


def players_dir(season: str) -> Path:
    return season_dir(season) / "players"


def player_file_path(season: str, player_id) -> Path:
    return players_dir(season) / f"{int(player_id)}.parquet"


def manifest_path(season: str) -> Path:
    return season_dir(season) / "_manifest.json"


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the (PLAYER_ID, GAME_ID) uniqueness key, keeping the first occurrence."""
    if df.empty:
        return df
    return df.drop_duplicates(subset=list(UNIQUE_KEY), keep="first").reset_index(
        drop=True
    )


def _finalize_gamelog_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shared tail of the raw-gamelog pipeline: enforce dtypes (including
    re-coercing PLAYER_ID/GAME_ID/GAME_DATE/IS_HOME unconditionally, not
    just trusting the caller already got them right), fill any missing
    RAW_GAMELOG_COLUMNS, select/order columns, drop rows missing a key
    identifier, deduplicate, and sort.

    Used by BOTH normalize_raw_gamelog (raw nba_api DataFrame -> our
    schema) and gamelogs_to_dataframe (canonical PlayerGameLog records ->
    our schema), so persisted output is byte-identical regardless of
    which path produced it -- see gamelogs_to_dataframe's docstring.
    """
    df = df.copy()

    df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce").astype("Int64")
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    if "IS_HOME" in df.columns:
        df["IS_HOME"] = pd.to_numeric(df["IS_HOME"], errors="coerce").astype("Int64")

    for col in _NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in RAW_GAMELOG_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[list(RAW_GAMELOG_COLUMNS)]
    df = df.dropna(subset=["PLAYER_ID", "GAME_ID", "GAME_DATE"])
    df = deduplicate(df)
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    return df


def normalize_raw_gamelog(
    df: pd.DataFrame, *, season: str, player_id, player_name: str
) -> pd.DataFrame:
    """
    Attach derived/provenance columns, enforce dtypes, and deduplicate one
    player's raw PlayerGameLog frame. Every native API column is preserved;
    nothing here computes a training feature.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=RAW_GAMELOG_COLUMNS)

    df = df.copy()

    # nba_api's raw column casing (Player_ID, Game_ID) is inconsistent with
    # the rest of its own schema (SEASON_ID, GAME_DATE, ...) -- normalize once.
    df = df.rename(columns={"Player_ID": "PLAYER_ID", "Game_ID": "GAME_ID"})

    df["SEASON"] = season
    df["PLAYER_NAME"] = player_name

    if "MATCHUP" in df.columns:
        matchup = df["MATCHUP"].astype(str)
    else:
        matchup = pd.Series([""] * len(df), index=df.index)
    df["MATCHUP"] = matchup
    df["TEAM_ABBREVIATION"] = matchup.str.split().str[0]
    df["OPPONENT_ABBREVIATION"] = matchup.str.split().str[-1]
    df["IS_HOME"] = matchup.str.contains("vs", case=False, na=False).astype("Int64")

    return _finalize_gamelog_frame(df)


def gamelogs_to_dataframe(records) -> pd.DataFrame:
    """
    Inverse of the canonical-record boundary
    (src.data.basketball.normalization.dataframe_to_player_game_logs):
    reconstructs the exact RAW_GAMELOG_COLUMNS-shaped DataFrame this
    module has always persisted, from a list of
    src.data.basketball.models.PlayerGameLog records.

    Passes through the same _finalize_gamelog_frame tail that
    normalize_raw_gamelog uses, so persisted output is identical
    regardless of whether it came from the raw-nba_api path or the
    provider/canonical-record path -- this is what makes the Step 6
    provider refactor safe for the historical collector: this module
    (storage.py) is the one place that owns RAW_GAMELOG_COLUMNS and both
    directions of conversion to/from it.
    """
    records = list(records)
    if not records:
        return pd.DataFrame(columns=RAW_GAMELOG_COLUMNS)

    rows = [
        {
            "SEASON": r.season,
            "SEASON_ID": r.season_id,
            "PLAYER_ID": r.player_id,
            "PLAYER_NAME": r.player_name,
            "GAME_ID": r.game_id,
            "GAME_DATE": r.game_date,
            "MATCHUP": r.matchup,
            "TEAM_ABBREVIATION": r.team_abbreviation,
            "OPPONENT_ABBREVIATION": r.opponent_abbreviation,
            "IS_HOME": r.is_home,
            "WL": r.wl,
            "MIN": r.minutes,
            "FGM": r.fgm,
            "FGA": r.fga,
            "FG_PCT": r.fg_pct,
            "FG3M": r.fg3m,
            "FG3A": r.fg3a,
            "FG3_PCT": r.fg3_pct,
            "FTM": r.ftm,
            "FTA": r.fta,
            "FT_PCT": r.ft_pct,
            "OREB": r.oreb,
            "DREB": r.dreb,
            "REB": r.reb,
            "AST": r.ast,
            "STL": r.stl,
            "BLK": r.blk,
            "TOV": r.tov,
            "PF": r.pf,
            "PTS": r.points,
            "PLUS_MINUS": r.plus_minus,
            "VIDEO_AVAILABLE": r.video_available,
        }
        for r in records
    ]
    df = pd.DataFrame(rows)
    return _finalize_gamelog_frame(df)


def save_player_gamelog(df: pd.DataFrame, *, season: str, player_id) -> Path:
    path = player_file_path(season, player_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write-then-rename so a crash mid-write can never leave a truncated
    # Parquet file behind masquerading as a completed player.
    tmp_path = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)
    return path


def load_player_gamelog(season: str, player_id) -> pd.DataFrame:
    path = player_file_path(season, player_id)
    if not path.exists():
        return pd.DataFrame(columns=RAW_GAMELOG_COLUMNS)
    return pd.read_parquet(path)


def is_player_collected(season: str, player_id) -> bool:
    """
    Ground-truth resumability check. A player counts as already collected
    only if a readable Parquet file exists for them -- this is checked
    independently of (and takes precedence over) whatever the manifest
    claims, so a stale or missing manifest can never cause a skipped
    re-fetch or silent data loss.
    """
    path = player_file_path(season, player_id)
    if not path.exists():
        return False
    try:
        pd.read_parquet(path)
        return True
    except Exception:
        logger.warning(
            "Existing raw file for player_id=%s season=%s is unreadable; will refetch",
            player_id,
            season,
        )
        return False


def load_season_gamelogs(season: str) -> pd.DataFrame:
    """Concatenate every collected player's file for a season into one frame."""
    directory = players_dir(season)
    if not directory.exists():
        return pd.DataFrame(columns=RAW_GAMELOG_COLUMNS)

    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        return pd.DataFrame(columns=RAW_GAMELOG_COLUMNS)

    frames = [pd.read_parquet(p) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    return deduplicate(combined)


def _empty_manifest(season: str) -> dict:
    return {
        "season": season,
        "completed_player_ids": [],
        "player_row_counts": {},
        "failed_players": {},
        "players_succeeded": 0,
        "players_failed": 0,
        "rows_collected": 0,
        "start_time": None,
        "last_resumed_at": None,
        "end_time": None,
    }


def load_manifest(season: str) -> dict:
    path = manifest_path(season)
    if not path.exists():
        return _empty_manifest(season)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning(
            "Manifest for season %s is unreadable; starting a fresh one", season
        )
        return _empty_manifest(season)


def save_manifest(season: str, manifest: dict) -> None:
    path = manifest_path(season)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write-then-rename for the same crash-safety reason as player files:
    # a kill -9 mid-write must never leave a half-written, unparseable
    # manifest that then blocks every future resume attempt.
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    tmp_path.replace(path)
