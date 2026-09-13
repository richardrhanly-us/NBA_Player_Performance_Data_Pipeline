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
