"""
The provider-payload <-> canonical-record normalization boundary.

Both functions here are pure, provider-agnostic, and independent of
column ordering or DataFrame index quirks: they look up every field BY
NAME (never by position), and never mutate their input.

They intentionally do NOT know anything about nba_api's raw response
shape (Player_ID/Game_ID casing, MATCHUP-string splitting, etc.) --
that provider-specific parsing is a concrete provider's job (see
providers/nba_api_provider.py, which calls training.data.storage's
existing, already-tested parsing before handing a DataFrame to
dataframe_to_player_game_logs() below). What lives here only needs an
input already shaped like training.data.storage.RAW_GAMELOG_COLUMNS /
the roster columns nba_client.fetch_season_roster returns -- i.e. our
own already-normalized column names, whichever provider produced them.
"""

from __future__ import annotations

import math

import pandas as pd

from src.data.basketball.models import Player, PlayerGameLog


def _safe_int(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NA:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _none_if_na(value):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def dataframe_to_players(df: pd.DataFrame) -> list[Player]:
    """
    Maps an already-normalized roster DataFrame (PLAYER_ID/PLAYER_NAME
    columns present, any order, any extra columns ignored) into our
    canonical Player records, preserving row order.
    """
    if df is None or df.empty:
        return []
    players = []
    for row in df.to_dict(orient="records"):
        player_id = _safe_int(row.get("PLAYER_ID"))
        if player_id is None:
            continue
        player_name = _none_if_na(row.get("PLAYER_NAME"))
        players.append(Player(player_id=player_id, player_name=str(player_name or "")))
    return players


def dataframe_to_player_game_logs(df: pd.DataFrame) -> list[PlayerGameLog]:
    """
    Maps an already-normalized gamelog DataFrame (one already shaped
    like training.data.storage.RAW_GAMELOG_COLUMNS -- i.e. one whose
    provider-specific raw parsing has already happened) into our
    canonical PlayerGameLog records, preserving row order.
    """
    if df is None or df.empty:
        return []
    records = []
    for row in df.to_dict(orient="records"):
        records.append(
            PlayerGameLog(
                season=row.get("SEASON"),
                season_id=_none_if_na(row.get("SEASON_ID")),
                player_id=_safe_int(row.get("PLAYER_ID")),
                player_name=str(_none_if_na(row.get("PLAYER_NAME")) or ""),
                game_id=_none_if_na(row.get("GAME_ID")),
                game_date=_none_if_na(row.get("GAME_DATE")),
                matchup=str(_none_if_na(row.get("MATCHUP")) or ""),
                team_abbreviation=_none_if_na(row.get("TEAM_ABBREVIATION")),
                opponent_abbreviation=_none_if_na(row.get("OPPONENT_ABBREVIATION")),
                is_home=_safe_int(row.get("IS_HOME")),
                wl=_none_if_na(row.get("WL")),
                minutes=_none_if_na(row.get("MIN")),
                fgm=_none_if_na(row.get("FGM")),
                fga=_none_if_na(row.get("FGA")),
                fg_pct=_none_if_na(row.get("FG_PCT")),
                fg3m=_none_if_na(row.get("FG3M")),
                fg3a=_none_if_na(row.get("FG3A")),
                fg3_pct=_none_if_na(row.get("FG3_PCT")),
                ftm=_none_if_na(row.get("FTM")),
                fta=_none_if_na(row.get("FTA")),
                ft_pct=_none_if_na(row.get("FT_PCT")),
                oreb=_none_if_na(row.get("OREB")),
                dreb=_none_if_na(row.get("DREB")),
                reb=_none_if_na(row.get("REB")),
                ast=_none_if_na(row.get("AST")),
                stl=_none_if_na(row.get("STL")),
                blk=_none_if_na(row.get("BLK")),
                tov=_none_if_na(row.get("TOV")),
                pf=_none_if_na(row.get("PF")),
                points=_none_if_na(row.get("PTS")),
                plus_minus=_none_if_na(row.get("PLUS_MINUS")),
                video_available=_none_if_na(row.get("VIDEO_AVAILABLE")),
            )
        )
    return records


def player_game_logs_to_dataframe(records) -> pd.DataFrame:
    """
    The inverse of dataframe_to_player_game_logs(): builds a DataFrame
    with the same uppercase, training.data.storage.RAW_GAMELOG_COLUMNS-
    style column names, from a list of canonical PlayerGameLog records.

    Deliberately simple -- a straightforward, deterministic column-by-
    column reconstruction, preserving row order and every field's value
    exactly as given. It does NOT deduplicate, drop rows, or sort (that
    is training.data.storage's own persistence-schema concern --
    training.data.storage.gamelogs_to_dataframe wraps this function and
    then applies its own dedup/dropna/sort finalize step on top). Kept
    here, with no dependency on training/, so src/shared_app.py's live
    path can reuse it directly without needing anything from the
    training package.
    """
    records = list(records)
    columns = (
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
    if not records:
        return pd.DataFrame(columns=columns)

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
    return pd.DataFrame(rows, columns=columns)
