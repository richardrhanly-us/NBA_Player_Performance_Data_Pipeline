from __future__ import annotations

import json
import os
import time
import unicodedata
from collections.abc import Callable
from typing import Any

import joblib
import pandas as pd
import requests
import streamlit as st

# Step 7: this module's basketball-data access (player search, player
# metadata, recent gamelogs, today's schedule, live box score) goes
# through the same provider boundary Step 6 introduced for historical
# collection -- see src/data/basketball/__init__.py. No direct nba_api
# import remains in this file; NBA-specific endpoint/response handling
# lives entirely behind NBAApiProvider.
from src.data.basketball.errors import ProviderError
from src.data.basketball.normalization import player_game_logs_to_dataframe
from src.data.basketball.provider import BasketballDataProvider, get_basketball_provider

# Canonical feature-building implementation. shared_app re-exports this name
# so existing callers (`from src.shared_app import build_player_feature_row`)
# keep working unchanged; the actual logic lives in src/features/ so it can
# be shared with a future training pipeline without duplication.
from src.features.build_features import build_player_feature_row
from src.results_pipeline import (
    get_final_points_from_gamelog as results_pipeline_get_final_points_from_gamelog,
)
from src.results_pipeline import (
    update_all_pending_sheet_results as results_pipeline_update_all_pending_sheet_results,
)

# get_results_sheet / get_strong_plays_sheet are used directly below.
# SHEET_KEY, get_gsheet_client, and get_historical_lines_sheet are not used
# in this module -- they are imported solely so that
# `from src.shared_app import ...` / `shared_app.<name>` keeps working for
# existing callers (scripts/top_plays_rebuild.py, apps/adminapp.py,
# scripts/pregame_pipeline.py). __all__ below marks that re-export as
# deliberate for static analysis instead of silently deleting it.
from src.sheets_utils import (
    SHEET_KEY,
    get_gsheet_client,
    get_historical_lines_sheet,
    get_results_sheet,
    get_strong_plays_sheet,
)

# Names imported above but not referenced in this module's own code, kept
# solely as compatibility re-exports for existing external callers. Listing
# them here is what tells static analysis (Ruff F401, Pylance) that this is
# deliberate rather than dead code -- do not remove a name from this list
# without first confirming nothing still imports it from shared_app.
__all__ = [
    "SHEET_KEY",
    "get_gsheet_client",
    "get_historical_lines_sheet",
]

CURRENT_SEASON = "2025-26"
APP_VERSION = "v1.2"
BOOKMAKER_KEY = "draftkings"
EDGE_THRESHOLD = 3.0
IS_STREAMLIT = "STREAMLIT_SERVER_RUNNING" in os.environ

def _noop_cache_data(**kwargs):
    def wrapper(func):
        return func
    return wrapper


def _noop_cache_resource(func):
    return func


# st.cache_data / st.cache_resource are Streamlit's CacheDataAPI /
# CacheResourceAPI objects, which pyright can't unify with a `def`
# fallback of the same name under one inferred type (that pattern trips
# reportRedeclaration). Both are genuinely just callables used as
# decorators/decorator factories, so binding each name once via a ternary,
# with an explicit Callable[..., Any] annotation, is the accurate (not
# merely permissive) type -- and avoids the two-declaration conflict.
cache_data: Callable[..., Any] = st.cache_data if IS_STREAMLIT else _noop_cache_data
cache_resource: Callable[..., Any] = st.cache_resource if IS_STREAMLIT else _noop_cache_resource

def get_final_points_from_gamelog(player_name, game_date):
    return results_pipeline_get_final_points_from_gamelog(
        player_name=player_name,
        game_date=game_date,
        load_active_players=load_active_players,
        normalize_name=normalize_name,
        get_player_gamelog_df=get_player_gamelog_df,
        CURRENT_SEASON=CURRENT_SEASON,
        safe_float=safe_float,
    )


def update_all_pending_sheet_results(debug=False):
    return results_pipeline_update_all_pending_sheet_results(
        load_active_players=load_active_players,
        normalize_name=normalize_name,
        get_player_gamelog_df=get_player_gamelog_df,
        CURRENT_SEASON=CURRENT_SEASON,
        safe_float=safe_float,
        debug=debug,
    )

def normalize_name(name: str) -> str:
    if not name:
        return ""

    name = str(name)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = name.lower()

    replacements = {
        ".": "",
        ",": "",
        "’": "'",
        "'": "",
        "-": " ",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    parts = name.split()
    suffixes = {"jr", "sr", "ii", "iii", "iv", "v"}

    parts = [p for p in parts if p not in suffixes]
    name = " ".join(parts)

    return " ".join(name.split()).strip()


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None




def parse_game_clock_to_minutes(clock_value):
    if clock_value is None:
        return None

    text = str(clock_value).strip()
    if not text:
        return None

    try:
        if text.startswith("PT"):
            text = text.replace("PT", "")
            mins = 0.0
            secs = 0.0

            if "M" in text:
                m_part = text.split("M")[0]
                mins = float(m_part) if m_part else 0.0
                text = text.split("M")[1]

            if "S" in text:
                s_part = text.replace("S", "")
                secs = float(s_part) if s_part else 0.0

            return mins + (secs / 60.0)

        if ":" in text:
            parts = text.split(":")
            if len(parts) == 2:
                mins = float(parts[0])
                secs = float(parts[1])
                return mins + (secs / 60.0)

        return float(text)

    except Exception:
        return None


def compute_game_minutes_remaining(period, game_clock_minutes):
    if period is None or game_clock_minutes is None:
        return None

    try:
        period = int(period)
        game_clock_minutes = float(game_clock_minutes)
    except Exception:
        return None

    if period <= 4:
        remaining_prior_periods = max(4 - period, 0) * 12.0
        return remaining_prior_periods + game_clock_minutes

    overtime_periods_left = 0.0
    return (5.0 * overtime_periods_left) + game_clock_minutes


def parse_minutes_to_float(minutes_value):
    """
    Moved here from apps/publicapp.py (Step 8) so
    get_live_adjusted_projection below -- and, via it, the new
    src/services/prediction_service.py -- can reuse it without importing
    a Streamlit app module. Behavior is unchanged: publicapp.py now
    imports this name from here instead of defining it locally.
    """
    if minutes_value is None:
        return None

    text = str(minutes_value).strip()
    if not text:
        return None

    try:
        if ":" in text:
            parts = text.split(":")
            if len(parts) == 2:
                mins = float(parts[0])
                secs = float(parts[1])
                return mins + (secs / 60.0)

        text = text.replace("PT", "")

        mins = 0.0
        secs = 0.0

        if "M" in text:
            m_part = text.split("M")[0]
            mins = float(m_part) if m_part else 0.0
            text = text.split("M")[1]

        if "S" in text:
            s_part = text.replace("S", "")
            secs = float(s_part) if s_part else 0.0

        return mins + (secs / 60.0)
    except Exception:
        return None


def get_live_adjusted_projection(predicted_points, live_stats):
    """
    Moved here from apps/publicapp.py (Step 8), unchanged, so the new
    single-player prediction service (src/services/prediction_service.py)
    can produce the exact same live-adjusted number the existing UI does,
    instead of duplicating this math. publicapp.py now imports this name
    from here instead of defining it locally.
    """
    if not live_stats:
        return predicted_points

    current_points = live_stats.get("points")
    minutes_played = parse_minutes_to_float(live_stats.get("minutes"))
    game_minutes_remaining = live_stats.get("game_minutes_remaining")

    try:
        current_points = float(current_points)
    except Exception:
        return predicted_points

    try:
        game_minutes_remaining = float(game_minutes_remaining)
    except Exception:
        game_minutes_remaining = None

    if game_minutes_remaining is None:
        return predicted_points

    if game_minutes_remaining <= 0:
        return current_points

    if minutes_played is None or minutes_played <= 0:
        return max(predicted_points, current_points)

    pregame_points_per_min = predicted_points / 48.0
    live_points_per_min = current_points / minutes_played

    live_weight = min(max(minutes_played / 24.0, 0.25), 0.75)
    pregame_weight = 1.0 - live_weight

    blended_points_per_min = (
        (pregame_points_per_min * pregame_weight) +
        (live_points_per_min * live_weight)
    )

    adjusted_projection = current_points + (blended_points_per_min * game_minutes_remaining)
    adjusted_projection = max(adjusted_projection, current_points)

    if game_minutes_remaining <= 0.25:
        return current_points
    if game_minutes_remaining <= 1.0:
        return min(adjusted_projection, current_points + 0.75)
    if game_minutes_remaining <= 2.0:
        return min(adjusted_projection, current_points + 1.5)
    if game_minutes_remaining <= 4.0:
        return min(adjusted_projection, current_points + 3.0)

    return adjusted_projection


def format_sportsbook_name(book_name):
    text = str(book_name or "").strip()
    lower = text.lower()

    if lower == "draftkings":
        return "DraftKings"
    if lower == "fanduel":
        return "FanDuel"
    if lower == "betmgm":
        return "BetMGM"
    if lower == "espnbet":
        return "ESPNBet"
    if lower == "betrivers":
        return "BetRivers"
    if lower == "hardrockbet":
        return "HardRockBet"

    return text.title() if text else ""


def format_event_game_date(commence_time):
    try:
        return pd.to_datetime(commence_time, utc=True).tz_convert("US/Central").strftime("%B %d, %Y")
    except Exception:
        return pd.Timestamp.now(tz="US/Central").strftime("%B %d, %Y")



@cache_data(ttl=120)
def get_sheet_records_df():
    sheet = get_results_sheet()
    values = sheet.get_all_values()

    if not values or len(values) < 2:
        return pd.DataFrame()

    headers = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=headers)


@cache_data(ttl=120)
def get_strong_plays_df():
    try:
        sheet = get_strong_plays_sheet()
        values = sheet.get_all_values()

        if not values or len(values) < 2:
            return pd.DataFrame()

        headers = values[0]
        rows = values[1:]
        return pd.DataFrame(rows, columns=headers)

    except Exception as e:
        print(f"[ERROR] get_strong_plays_df failed: {e}")
        return pd.DataFrame()


@cache_data(ttl=120)
def get_strong_plays_summary():
    df = get_strong_plays_df()
    if df.empty or "bet_status" not in df.columns:
        return None, 0

    df = df.copy()
    df["bet_status"] = df["bet_status"].astype(str).str.strip().str.upper()
    graded_df = df[df["bet_status"].isin(["WIN", "LOSS"])].copy()

    total_games = len(graded_df)
    if total_games == 0:
        return None, 0

    wins = len(graded_df[graded_df["bet_status"] == "WIN"])
    win_rate = (wins / total_games) * 100

    return win_rate, total_games



@cache_data(ttl=120)
def get_strong_plays_health():
    df = get_strong_plays_df()
    if df.empty:
        return None

    df = df.copy()

    for col in ["PLAYER_NAME", "sportsbook_line", "bet_status", "result_logged_at"]:
        if col not in df.columns:
            df[col] = ""

    df["PLAYER_NAME"] = df["PLAYER_NAME"].astype(str).str.strip()
    df["sportsbook_line"] = pd.to_numeric(df["sportsbook_line"], errors="coerce")
    df["bet_status"] = df["bet_status"].astype(str).str.strip().str.upper()

    # only count real play rows
    df = df[
        (df["PLAYER_NAME"] != "") &
        (df["sportsbook_line"].notna())
    ].copy()

    total = len(df)
    graded = len(df[df["bet_status"].isin(["WIN", "LOSS"])])
    pending = len(df[df["bet_status"] == "PENDING"])

    last_update = None
    if "result_logged_at" in df.columns:
        try:
            last_update = pd.to_datetime(df["result_logged_at"], errors="coerce").max()
        except Exception:
            last_update = None

    return {
        "total": total,
        "graded": graded,
        "pending": pending,
        "last_update": last_update
    }



@cache_data(ttl=900)
def load_model():
    model_path = "models/points_regression.pkl"
    print("MODEL PATH:", model_path, flush=True)
    print("EXISTS:", os.path.exists(model_path), flush=True)
    if os.path.exists(model_path):
        print("SIZE:", os.path.getsize(model_path), flush=True)
        with open(model_path, "rb") as f:
            print("FIRST 40 BYTES:", f.read(40), flush=True)
    return joblib.load(model_path)


@cache_data(ttl=3600)
def load_model_stats():
    with open("models/points_model_stats.json", "r") as f:
        return json.load(f)

@cache_data(ttl=3600)
def load_active_players(_provider: BasketballDataProvider | None = None):
    provider = _provider or get_basketball_provider()
    active_players = provider.get_active_players()

    actual_name_to_id = {}
    normalized_to_actual = {}

    for p in active_players:
        actual_name = str(p.player_name).strip()
        player_id = p.player_id

        actual_name_to_id[actual_name] = player_id

        normalized = normalize_name(actual_name)
        normalized_to_actual[normalized] = actual_name

        parts = normalized.split()
        if len(parts) >= 2:
            first = parts[0]
            last = parts[-1]
            normalized_to_actual[f"{first} {last}"] = actual_name

    return actual_name_to_id, normalized_to_actual


@cache_data(ttl=3600)
def get_player_details(player_id, _provider: BasketballDataProvider | None = None):
    provider = _provider or get_basketball_provider()
    try:
        return provider.get_player_details(player_id)
    except ProviderError:
        return None

def resolve_player_name(raw_name, normalized_to_actual):
    normalized = normalize_name(raw_name)

    exact = normalized_to_actual.get(normalized)
    if exact:
        return exact

    raw_parts = normalized.split()
    if len(raw_parts) >= 2:
        first = raw_parts[0]
        last = raw_parts[-1]

        for norm_name, actual_name in normalized_to_actual.items():
            norm_parts = norm_name.split()
            if len(norm_parts) >= 2:
                if norm_parts[0] == first and norm_parts[-1] == last:
                    return actual_name

    for norm_name, actual_name in normalized_to_actual.items():
        if normalized == norm_name:
            return actual_name

    return None

@cache_data(ttl=3600, show_spinner=False)
def get_player_gamelog_df(player_id, season, _provider: BasketballDataProvider | None = None):
    # Pre-Step-7 this function retried the raw nba_api call itself (2
    # attempts, a flat 2s sleep between them). That retry/backoff
    # responsibility now lives inside the provider (NBAApiProvider ->
    # training.data.nba_client.call_with_retries: 5 attempts, exponential
    # backoff) -- a deliberate, disclosed change in *how long a total
    # failure takes to surface*, not in what data/features/predictions
    # result from a successful fetch. See the Step 7 report's cache/
    # latency review.
    provider = _provider or get_basketball_provider()
    try:
        records = provider.get_player_game_logs(player_id, season)
        return player_game_logs_to_dataframe(records)
    except ProviderError as e:
        print(
            f"[PIPELINE] Gamelog fetch failed for player_id={player_id}: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        return pd.DataFrame()


@cache_data(ttl=180)
def get_scoreboard_for_date(game_date=None, _provider: BasketballDataProvider | None = None):
    """Returns list[ScheduledGame] for `game_date` (nba_api's own
    "%m/%d/%Y" string; None uses the provider's own default -- see
    NBAApiProvider.get_todays_scoreboard for exactly what that default
    is and why it's preserved unchanged from before Step 7)."""
    provider = _provider or get_basketball_provider()
    try:
        return provider.get_todays_scoreboard(game_date)
    except ProviderError:
        return []


def get_live_player_stats(player_name, provider: BasketballDataProvider | None = None):
    provider = provider or get_basketball_provider()

    actual_name_to_id, normalized_to_actual = load_active_players(_provider=provider)
    actual_name = normalized_to_actual.get(normalize_name(player_name), player_name)
    player_id = actual_name_to_id.get(actual_name)
    if not player_id:
        return None

    player_details = get_player_details(player_id, _provider=provider)
    if player_details is None or player_details.team_id is None:
        return None
    team_id = player_details.team_id

    try:
        eastern_now = pd.Timestamp.now(tz="US/Eastern")
        game_date = eastern_now.strftime("%m/%d/%Y")
        games = get_scoreboard_for_date(game_date, _provider=provider)
    except Exception:
        return None

    team_game = next(
        (g for g in games if g.home_team_id == team_id or g.away_team_id == team_id),
        None,
    )
    if team_game is None:
        return None

    game_id = team_game.game_id
    game_status_text = str(team_game.game_status_text or "Live").strip()

    try:
        box_score = provider.get_live_box_score(game_id)
        if box_score is None:
            return None

        matched = None

        for line in box_score.players:
            if str(line.player_id or "") == str(player_id):
                matched = line
                break

        if matched is None:
            for line in box_score.players:
                full_name = f"{line.first_name} {line.last_name}".strip()
                if full_name.lower() == actual_name.lower():
                    matched = line
                    break

        if matched is None:
            return None

        points = matched.points
        minutes = matched.minutes

        period = box_score.period
        game_clock = box_score.game_clock

        clock_minutes = parse_game_clock_to_minutes(game_clock)
        game_minutes_remaining = compute_game_minutes_remaining(period, clock_minutes)

        return {
            "points": points if str(points).strip() != "" else 0,
            "minutes": str(minutes) if minutes is not None else "0",
            "game_status": game_status_text,
            "period": period,
            "game_clock": game_clock,
            "game_minutes_remaining": game_minutes_remaining
        }

    except Exception:
        return None


def fetch_upcoming_nba_events(api_key):
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    resp = requests.get(url, params={"apiKey": api_key}, timeout=20)
    resp.raise_for_status()
    return resp.json()


def fetch_player_points_market(api_key, event_id, bookmaker_key):
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
    resp = requests.get(
        url,
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "player_points",
            "bookmakers": bookmaker_key,
            "oddsFormat": "american"
        },
        timeout=20
    )
    resp.raise_for_status()
    return resp.json()


def fetch_all_today_player_props(api_key, bookmaker_key):
    events = fetch_upcoming_nba_events(api_key)
    rows = []

    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue

        try:
            event_odds = fetch_player_points_market(api_key, event_id, bookmaker_key)
        except Exception:
            continue

        time.sleep(0.3)

        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        for bookmaker in event_odds.get("bookmakers", []):
            book_title = bookmaker.get("title", "Unknown")
            book_key = bookmaker.get("key", bookmaker_key)

            for market in bookmaker.get("markets", []):
                if market.get("key") != "player_points":
                    continue

                market_last_update = market.get("last_update", "")
                grouped = {}

                for outcome in market.get("outcomes", []):
                    player_desc = outcome.get("description", "")
                    point = outcome.get("point")
                    side = outcome.get("name")
                    price = outcome.get("price")

                    if not player_desc or point is None or side not in ("Over", "Under"):
                        continue

                    key = (normalize_name(player_desc), float(point))
                    if key not in grouped:
                        grouped[key] = {
                            "player_name_raw": player_desc,
                            "line": float(point),
                            "over_price": None,
                            "under_price": None
                        }

                    if side == "Over":
                        grouped[key]["over_price"] = price
                    elif side == "Under":
                        grouped[key]["under_price"] = price

                for _, item in grouped.items():
                    if item["over_price"] is None or item["under_price"] is None:
                        continue

                    rows.append({
                        "player_name_raw": item["player_name_raw"],
                        "line": item["line"],
                        "bookmaker": book_title,
                        "bookmaker_key": str(book_key).lower(),
                        "last_update": market_last_update,
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": event.get("commence_time", ""),
                        "over_price": item["over_price"],
                        "under_price": item["under_price"]
                    })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(
        subset=["player_name_raw", "line", "bookmaker_key", "commence_time"]
    ).reset_index(drop=True)
    return df


@cache_data(ttl=300)
def get_today_games(api_key):
    try:
        return fetch_upcoming_nba_events(api_key)
    except Exception:
        return []


@cache_data(ttl=300, show_spinner=False)
def get_available_sportsbooks():
    return [
        "draftkings",
        "fanduel",
        "betmgm",
        "caesars",
        "espnbet",
        "betrivers",
        "hardrockbet",
    ]


def get_odds_api_key():
    """
    Extracted from get_player_points_lines (Step 8) so the new daily
    prediction board (src/services/prediction_service.py) can resolve
    the same odds API key the same way, without duplicating this
    lookup. Behavior unchanged: Streamlit secrets first, then the
    ODDS_API_KEY environment variable, else None.
    """
    api_key = None
    try:
        if "ODDS_API_KEY" in st.secrets:
            api_key = st.secrets["ODDS_API_KEY"]
    except Exception:
        pass

    if not api_key:
        api_key = os.environ.get("ODDS_API_KEY")

    return api_key


@cache_data(ttl=300, show_spinner=False)
def get_player_points_lines(player_name, bookmaker_key):
    api_key = get_odds_api_key()
    if not api_key:
        return None

    props_df = fetch_all_today_player_props(api_key, bookmaker_key)
    if props_df.empty:
        return None

    props_df = props_df.copy()

    if "player_name_raw" in props_df.columns:
        name_col = "player_name_raw"
    elif "player_name" in props_df.columns:
        name_col = "player_name"
    elif "description" in props_df.columns:
        name_col = "description"
    else:
        return None

    props_df[name_col] = props_df[name_col].astype(str)
    props_df["normalized_name"] = props_df[name_col].apply(normalize_name)

    normalized_target = normalize_name(player_name)

    player_df = props_df[props_df["normalized_name"] == normalized_target].copy()

    if player_df.empty:
        player_df = props_df[
            props_df["normalized_name"].str.contains(normalized_target, na=False)
        ].copy()

    if player_df.empty:
        player_df = props_df[
            props_df["normalized_name"].apply(
                lambda x: normalized_target in x if isinstance(x, str) else False
            )
        ].copy()

    if player_df.empty:
        return None

    if "line" not in player_df.columns:
        return None

    player_df["line"] = pd.to_numeric(player_df["line"], errors="coerce")
    player_df = player_df[player_df["line"].notna()].copy()

    if player_df.empty:
        return None

    row = player_df.iloc[0]

    return {
        "player_name": row.get(name_col),
        "points_line": float(row.get("line")),
        "sportsbook": row.get("bookmaker", bookmaker_key),
        "last_update": row.get("last_update", ""),
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "commence_time": row.get("commence_time", ""),
        "over_price": row.get("over_price"),
        "under_price": row.get("under_price"),
    }


@cache_data(ttl=300, show_spinner=False)
def get_top_plays_today_df(api_key, debug=False):
    print("[PIPELINE] START get_top_plays_today_df", flush=True)

    model = load_model()
    actual_name_to_id, normalized_to_actual = load_active_players()
    model_feature_names = list(getattr(model, "feature_names_in_", []))

    print("[PIPELINE] Fetching props dataframe...", flush=True)
    props_df = fetch_all_today_player_props(api_key, BOOKMAKER_KEY)
    print(f"[PIPELINE] Props dataframe rows: {len(props_df)}", flush=True)

    if props_df.empty:
        print("[PIPELINE] No props returned", flush=True)
        return pd.DataFrame()

    
    props_df["normalized_name"] = props_df["player_name_raw"].apply(normalize_name)
    props_df = props_df.drop_duplicates(subset=["normalized_name"]).copy()

    print(f"[PIPELINE] Unique player rows after dedupe: {len(props_df)}", flush=True)

    rows = []
    gamelog_cache = {}
    total_rows = len(props_df)

    skipped_unresolved_name = 0
    skipped_missing_player_id = 0
    skipped_empty_gamelog = 0
    skipped_empty_features = 0
    skipped_missing_line = 0
    skipped_below_edge = 0
    
    
    status_box = None
    progress_bar = None

    if debug:
        status_box = st.empty()
        progress_bar = st.progress(0)

    print("[PIPELINE] Beginning scoring loop...", flush=True)

    for i, (_, row) in enumerate(props_df.iterrows(), start=1):
        raw_name = row["player_name_raw"]

        if debug and status_box is not None and progress_bar is not None:
            status_box.markdown(
                f"""
                <div class="status-box">
                    <div><span class="muted">Top Plays Status:</span> Scoring player {i} of {total_rows}</div>
                    <div><span class="muted">Current Player:</span> {raw_name}</div>
                    <div><span class="muted">Step:</span> Loading gamelog and running model</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            progress_bar.progress(i / total_rows)
        
        actual_name = resolve_player_name(raw_name, normalized_to_actual)
        if not actual_name:
            skipped_unresolved_name += 1
            continue

        player_id = actual_name_to_id.get(actual_name)
        if not player_id:
            skipped_missing_player_id += 1
            continue

        if player_id in gamelog_cache:
            df = gamelog_cache[player_id]
        else:
            df = get_player_gamelog_df(player_id, CURRENT_SEASON)
            if not df.empty:
                gamelog_cache[player_id] = df

        if df.empty:
            skipped_empty_gamelog += 1
            continue

        X = build_player_feature_row(df, actual_name, row["line"])
        if X is None or X.empty:
            skipped_empty_features += 1
            continue

        if model_feature_names:
            X = X.reindex(columns=model_feature_names, fill_value=0)

        predicted_points = float(model.predict(X)[0])

        line = safe_float(row["line"])
        if line is None:
            skipped_missing_line += 1
            continue

        edge = predicted_points - line
        if abs(edge) < EDGE_THRESHOLD:
            skipped_below_edge += 1
            print(
                f"[PIPELINE] Skip: edge below threshold for {actual_name} "
                f"(edge={round(edge, 2)}, threshold={EDGE_THRESHOLD})",
                flush=True
            )
            continue

        sportsbook_name = row.get("bookmaker", "")
        sportsbook_key = row.get("bookmaker_key", "").lower()
        
        rows.append({
            "PLAYER_NAME": actual_name,

            
            "GAME_DATE": format_event_game_date(row.get("commence_time", "")),
            "last_update": row.get("last_update", ""),

            "sportsbook": sportsbook_key,
            "sportsbook_name": sportsbook_name,
            "sportsbook_line": line,

            "predicted_points": round(predicted_points, 2),
            "edge": round(edge, 2),
            "model_pick": "OVER" if edge > 0 else "UNDER",

            "home_team": row.get("home_team", ""),
            "away_team": row.get("away_team", ""),
            "commence_time": row.get("commence_time", "")
        })

        time.sleep(0.5)
    
    if debug and status_box is not None:
        status_box.markdown(
            """
            <div class="status-box">
                <div><span class="muted">Top Plays Status:</span> Ranking strongest edges</div>
                <div><span class="muted">Step:</span> Finalizing board</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if debug and status_box is not None and progress_bar is not None:
        progress_bar.progress(1.0)
        time.sleep(0.3)
        status_box.empty()
        progress_bar.empty()

    
    print(
        "[PIPELINE] Skip summary | "
        f"unresolved_name={skipped_unresolved_name} | "
        f"missing_player_id={skipped_missing_player_id} | "
        f"empty_gamelog={skipped_empty_gamelog} | "
        f"empty_features={skipped_empty_features} | "
        f"missing_line={skipped_missing_line} | "
        f"below_edge={skipped_below_edge}",
        flush=True
    )
    
    print(f"[PIPELINE] Rows that passed edge threshold: {len(rows)}", flush=True)
    
    if not rows:
        return pd.DataFrame()

    top_df = pd.DataFrame(rows)
    top_df = top_df.sort_values("edge", ascending=False, key=lambda s: s.abs()).reset_index(drop=True)

    print(f"[PIPELINE] Returning top_df with {len(top_df)} rows", flush=True)
    return top_df
