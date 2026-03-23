import os
import json
import time
import requests
import joblib
import pandas as pd
import unicodedata
import streamlit as st

from datetime import datetime
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, commonplayerinfo, scoreboardv2

from src.sheets_utils import (
    SHEET_KEY,
    RESULTS_SHEET_NAME,
    STRONG_PLAYS_SHEET_NAME,
    HISTORICAL_LINES_SHEET_NAME,
    clear_app_caches,
    get_gsheet_client,
    get_worksheet,
    get_historical_lines_sheet,
    get_strong_plays_sheet,
    get_results_sheet,
    get_worksheet_with_df,
    column_letter_from_index,
    build_header_index_map,
)

from src.results_pipeline import (
    normalize_sheet_date,
    get_final_points_from_gamelog as results_pipeline_get_final_points_from_gamelog,
    is_blank_cell,
    is_pending_result_row,
    update_sheet_with_final_result,
    populate_closing_lines_and_clv,
    update_all_pending_sheet_results as results_pipeline_update_all_pending_sheet_results,
)

CURRENT_SEASON = "2025-26"
APP_VERSION = "v1.1"
BOOKMAKER_KEY = "draftkings"
EDGE_THRESHOLD = 3.0

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
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    return (
        name.lower()
        .replace(".", "")
        .replace("’", "'")
        .replace("-", " ")
        .replace(" jr", "")
        .replace(" sr", "")
        .replace(" iii", "")
        .replace(" ii", "")
        .strip()
    )


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



@st.cache_data(ttl=120)
def get_sheet_records_df():
    sheet = get_results_sheet()
    values = sheet.get_all_values()

    if not values or len(values) < 2:
        return pd.DataFrame()

    headers = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=headers)


@st.cache_data(ttl=120)
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


@st.cache_data(ttl=120)
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


@st.cache_data(ttl=120)
def get_strong_plays_health():
    df = get_strong_plays_df()
    if df.empty or "bet_status" not in df.columns:
        return None

    df = df.copy()
    df["bet_status"] = df["bet_status"].astype(str).str.strip().str.upper()

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


@st.cache_resource
def load_model():
    return joblib.load("models/points_regression.pkl")


@st.cache_data(ttl=3600)
def load_model_stats():
    with open("models/points_model_stats.json", "r") as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_active_players():
    active_players = players.get_active_players()
    actual_name_to_id = {p["full_name"]: p["id"] for p in active_players}
    normalized_to_actual = {}
    for actual_name in actual_name_to_id.keys():
        normalized_to_actual[normalize_name(actual_name)] = actual_name
    return actual_name_to_id, normalized_to_actual


@st.cache_data(ttl=3600)
def get_player_info_df(player_id):
    try:
        return commonplayerinfo.CommonPlayerInfo(
            player_id=player_id,
            timeout=12
        ).get_data_frames()[0]
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900)
def get_player_gamelog_df(player_id, season):
    for attempt in range(2):
        try:
            return playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                timeout=12
            ).get_data_frames()[0]
        except Exception:
            if attempt == 1:
                return pd.DataFrame()
            time.sleep(2)
    return pd.DataFrame()


def build_player_feature_row(df, player_name):
    df = df.copy()
    df["PLAYER_NAME"] = player_name
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    numeric_cols = [
        "PTS", "FGM", "FGA", "FTA", "FTM", "OREB", "DREB",
        "STL", "AST", "BLK", "PF", "TOV", "MIN"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "FG3A" in df.columns:
        df["FG3A"] = pd.to_numeric(df["FG3A"], errors="coerce")

    df["gmsc"] = (
        df["PTS"]
        + 0.4 * df["FGM"]
        - 0.7 * df["FGA"]
        - 0.4 * (df["FTA"] - df["FTM"])
        + 0.7 * df["OREB"]
        + 0.3 * df["DREB"]
        + df["STL"]
        + 0.7 * df["AST"]
        + 0.7 * df["BLK"]
        - 0.4 * df["PF"]
        - df["TOV"]
    )

    df["player_avg_pts"] = df.groupby("PLAYER_NAME")["PTS"].transform(lambda x: x.shift(1).expanding().mean())
    df["player_avg_pts_sq"] = df["player_avg_pts"] ** 2
    df["last3_pts"] = df.groupby("PLAYER_NAME")["PTS"].transform(lambda x: x.shift(1).rolling(3).mean())
    df["last5_pts"] = df.groupby("PLAYER_NAME")["PTS"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["last10_pts"] = df.groupby("PLAYER_NAME")["PTS"].transform(lambda x: x.shift(1).rolling(10).mean())
    df["last20_pts"] = df.groupby("PLAYER_NAME")["PTS"].transform(lambda x: x.shift(1).rolling(20).mean())
    df["last5_fga"] = df.groupby("PLAYER_NAME")["FGA"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["last5_fta"] = df.groupby("PLAYER_NAME")["FTA"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["last5_minutes"] = df.groupby("PLAYER_NAME")["MIN"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["last5_gmsc"] = df.groupby("PLAYER_NAME")["gmsc"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["home_game"] = df["MATCHUP"].astype(str).str.contains("vs").astype(int)
    df["days_rest"] = df.groupby("PLAYER_NAME")["GAME_DATE"].diff().dt.days.fillna(3)
    df["is_back_to_back"] = (df["days_rest"] == 1).astype(int)
    df["usage_proxy"] = df["FGA"] + 0.44 * df["FTA"] + df["TOV"]
    df["last5_usage_proxy"] = df.groupby("PLAYER_NAME")["usage_proxy"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["season_minutes_avg"] = df.groupby("PLAYER_NAME")["MIN"].transform(lambda x: x.shift(1).expanding().mean())
    df["minutes_volatility"] = df.groupby("PLAYER_NAME")["MIN"].transform(lambda x: x.shift(1).rolling(5).std())
    df["points_volatility"] = df.groupby("PLAYER_NAME")["PTS"].transform(lambda x: x.shift(1).rolling(5).std())

    if "FG3A" in df.columns:
        df["last5_3pa"] = df.groupby("PLAYER_NAME")["FG3A"].transform(lambda x: x.shift(1).rolling(5).mean())

    required_features = [
        "player_avg_pts",
        "player_avg_pts_sq",
        "season_minutes_avg",
        "home_game",
        "days_rest",
        "is_back_to_back",
        "last3_pts",
        "last5_pts",
        "last10_pts",
        "last20_pts",
        "last5_fga",
        "last5_fta",
        "last5_minutes",
        "last5_gmsc",
        "last5_usage_proxy",
        "minutes_volatility",
        "points_volatility"
    ]

    if "last5_3pa" in df.columns:
        required_features.append("last5_3pa")

    df_features = df.dropna(subset=required_features).reset_index(drop=True)
    if df_features.empty:
        return None

    latest = df_features.iloc[-1]

    feature_data = {
        "player_avg_pts": latest["player_avg_pts"],
        "player_avg_pts_sq": latest["player_avg_pts_sq"],
        "season_minutes_avg": latest["season_minutes_avg"],
        "home_game": latest["home_game"],
        "days_rest": latest["days_rest"],
        "is_back_to_back": latest["is_back_to_back"],
        "last3_pts": latest["last3_pts"],
        "last5_pts": latest["last5_pts"],
        "last10_pts": latest["last10_pts"],
        "last20_pts": latest["last20_pts"],
        "last5_fga": latest["last5_fga"],
        "last5_fta": latest["last5_fta"],
        "last5_minutes": latest["last5_minutes"],
        "last5_gmsc": latest["last5_gmsc"],
        "last5_usage_proxy": latest["last5_usage_proxy"],
        "minutes_volatility": latest["minutes_volatility"],
        "points_volatility": latest["points_volatility"]
    }

    if "last5_3pa" in df_features.columns and pd.notna(latest.get("last5_3pa", None)):
        feature_data["last5_3pa"] = latest["last5_3pa"]

    return pd.DataFrame([feature_data])


@st.cache_data(ttl=180)
def get_scoreboard_for_date(game_date=None):
    try:
        if game_date is None:
            game_date = datetime.now().strftime("%m/%d/%Y")
        return scoreboardv2.ScoreboardV2(
            game_date=game_date,
            day_offset=0,
            league_id="00",
            timeout=12
        ).get_data_frames()
    except Exception:
        return []


def get_live_player_stats(player_name):
    try:
        from nba_api.live.nba.endpoints import boxscore as live_boxscore
    except Exception:
        return None

    actual_name_to_id, normalized_to_actual = load_active_players()
    actual_name = normalized_to_actual.get(normalize_name(player_name), player_name)
    player_id = actual_name_to_id.get(actual_name)
    if not player_id:
        return None

    info_df = get_player_info_df(player_id)
    if info_df is None or info_df.empty:
        return None

    try:
        team_id = int(info_df.iloc[0]["TEAM_ID"])
    except Exception:
        return None

    try:
        eastern_now = pd.Timestamp.now(tz="US/Eastern")
        game_date = eastern_now.strftime("%m/%d/%Y")
        board_frames = get_scoreboard_for_date(game_date)
    except Exception:
        return None

    if not board_frames or len(board_frames) < 2:
        return None

    try:
        game_header = board_frames[0]
    except Exception:
        return None

    if game_header is None or game_header.empty:
        return None

    team_game = game_header[
        (game_header["HOME_TEAM_ID"] == team_id) |
        (game_header["VISITOR_TEAM_ID"] == team_id)
    ]

    if team_game.empty:
        return None

    game = team_game.iloc[0]
    game_id = str(game["GAME_ID"])
    game_status_text = str(game.get("GAME_STATUS_TEXT", "Live")).strip()

    try:
        live = live_boxscore.BoxScore(game_id=game_id)
        data = live.get_dict()

        players_live = []
        players_live.extend(data.get("game", {}).get("homeTeam", {}).get("players", []))
        players_live.extend(data.get("game", {}).get("awayTeam", {}).get("players", []))

        matched = None

        for p in players_live:
            if str(p.get("personId", "")) == str(player_id):
                matched = p
                break

        if matched is None:
            for p in players_live:
                full_name = f"{p.get('firstName', '').strip()} {p.get('familyName', '').strip()}".strip()
                if full_name.lower() == actual_name.lower():
                    matched = p
                    break

        if matched is None:
            return None

        stats = matched.get("statistics", {})
        points = stats.get("points", 0)
        minutes = stats.get("minutes", "0")

        game_data = data.get("game", {})
        period = game_data.get("period")
        game_clock = game_data.get("gameClock")

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


@st.cache_data(ttl=300)
def get_today_games(api_key):
    try:
        return fetch_upcoming_nba_events(api_key)
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
def get_player_points_lines(player_name, bookmaker_key):
    api_key = None

    try:
        if "ODDS_API_KEY" in st.secrets:
            api_key = st.secrets["ODDS_API_KEY"]
    except Exception:
        pass

    if not api_key:
        api_key = os.environ.get("ODDS_API_KEY")

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


@st.cache_data(ttl=300, show_spinner=False)
def get_top_plays_today_df(api_key, debug=False):
    print("[PIPELINE] START get_top_plays_today_df", flush=True)
    model = load_model()
    actual_name_to_id, normalized_to_actual = load_active_players()
    model_feature_names = list(getattr(model, "feature_names_in_", []))

    props_df = fetch_all_today_player_props(api_key, BOOKMAKER_KEY)
    if props_df.empty:
        return pd.DataFrame()

    print("[PIPELINE] Fetching odds...", flush=True)
    props_df = props_df.head(10).copy()
    props_df["normalized_name"] = props_df["player_name_raw"].apply(normalize_name)
    props_df = props_df.drop_duplicates(subset=["normalized_name"]).copy()

    rows = []
    gamelog_cache = {}
    total_rows = len(props_df)

    status_box = None
    progress_bar = None
    print(f"[PIPELINE] Built dataframe with {len(df)} rows", flush=True)
    
    if debug:
        status_box = st.empty()
        progress_bar = st.progress(0)

    print("[PIPELINE] Filtering for top plays...", flush=True)
    for i, (_, row) in enumerate(props_df.iterrows(), start=1):
        raw_name = row["player_name_raw"]

        if debug:
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

        normalized = normalize_name(raw_name)
        actual_name = normalized_to_actual.get(normalized)
        if not actual_name:
            continue

        player_id = actual_name_to_id.get(actual_name)
        if not player_id:
            continue

        if player_id in gamelog_cache:
            df = gamelog_cache[player_id]
        else:
            df = get_player_gamelog_df(player_id, CURRENT_SEASON)
            if not df.empty:
                gamelog_cache[player_id] = df

        if df.empty:
            continue

        X = build_player_feature_row(df, actual_name)
        if X is None or X.empty:
            continue

        if model_feature_names:
            X = X.reindex(columns=model_feature_names, fill_value=0)

        predicted_points = float(model.predict(X)[0])
        line = safe_float(row["line"])
        if line is None:
            continue

        edge = predicted_points - line
        if abs(edge) < EDGE_THRESHOLD:
            continue

        rows.append({
            "PLAYER_NAME": actual_name,
            "sportsbook": row["bookmaker"],
            "sportsbook_line": line,
            "predicted_points": round(predicted_points, 2),
            "edge": round(edge, 2),
            "model_pick": "OVER" if edge > 0 else "UNDER",
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "commence_time": row["commence_time"]
        })

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

    if debug and progress_bar is not None:
        progress_bar.progress(1.0)
        time.sleep(0.3)
        status_box.empty()
        progress_bar.empty()

    if not rows:
        return pd.DataFrame()

    top_df = pd.DataFrame(rows)
    top_df = top_df.sort_values("edge", ascending=False, key=lambda s: s.abs()).reset_index(drop=True)
    print(f"[PIPELINE] Returning {len(top_df)} top plays", flush=True)
    return top_df



    df = get_player_gamelog_df(player_id, CURRENT_SEASON)
    if df.empty or "GAME_DATE" not in df.columns:
        return None

    df = df.copy()
    df["GAME_DATE_NORM"] = pd.to_datetime(df["GAME_DATE"], errors="coerce").dt.strftime("%B %d, %Y")
    target_date = normalize_sheet_date(game_date)

    match_df = df[df["GAME_DATE_NORM"] == target_date]
    if match_df.empty:
        return None

    final_points = safe_float(match_df.iloc[0].get("PTS"))
    return final_points


