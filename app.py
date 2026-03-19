import os
import json
import time
import requests
import joblib
import pandas as pd
import streamlit as st
import pytz
import unicodedata

from scipy.stats import norm
from datetime import datetime

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, commonplayerinfo, scoreboardv2

APP_VERSION = "v1.25 - Updated model - debug"


st.set_page_config(
    page_title="NBA Points Prop Predictor",
    page_icon="🏀",
    layout="centered"
)


st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #081120 0%, #0f172a 100%);
        color: #f8fafc;
    }

    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 3rem;
        max-width: 980px;
    }

    hr, div[data-testid="stDivider"] {
        display: none !important;
    }

    .hero {
        background:
            radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 34%),
            radial-gradient(circle at top right, rgba(168,85,247,0.14), transparent 30%),
            linear-gradient(135deg, #111827 0%, #1e293b 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 26px 24px 18px 24px;
        margin-bottom: 14px;
        box-shadow: 0 14px 34px rgba(0,0,0,0.30);
        position: relative;
        overflow: hidden;
    }

    .hero::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.03), transparent);
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 900;
        margin-bottom: 6px;
        color: #ffffff;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        margin-bottom: 14px;
        position: relative;
        z-index: 1;
    }

    .hero-pills {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }

    .hero-pill {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        color: #dbeafe;
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .section-card {
        background: rgba(15, 23, 42, 0.96);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 16px;
        margin-top: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 12px;
        color: #f8fafc;
    }

    .summary-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
    }

    .summary-item {
        background: #0f172a;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 10px 12px;
    }

    .summary-label {
        color: #94a3b8;
        font-size: 0.72rem;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .summary-value {
        color: #f8fafc;
        font-size: 0.98rem;
        font-weight: 700;
        line-height: 1.2;
    }

    .model-card {
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        margin-top: 4px;
        margin-bottom: 16px;
    }

    .model-title {
        font-size: 1.15rem;
        font-weight: 900;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 14px;
        color: white;
        text-shadow: 0 0 6px rgba(255,255,255,0.25);
    }

    .model-subtitle {
        font-size: 0.75rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        opacity: 0.7;
        margin-bottom: 10px;
        color: #e2e8f0;
    }

    .model-main {
        display: grid;
        grid-template-columns: 1.2fr 0.9fr 0.9fr 1fr;
        gap: 14px;
    }

    .model-stat {
        border-radius: 14px;
        padding: 14px 16px;
    }

    .model-stat-label {
        font-size: 0.75rem;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .model-stat-value {
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 900;
    }

    .pick-banner {
        margin-top: 16px;
        border-radius: 14px;
        padding: 14px 16px;
        font-size: 1.05rem;
        font-weight: 900;
        text-align: center;
        letter-spacing: 0.05em;
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.84rem;
        margin-top: 10px;
    }

    .recent-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-bottom: 12px;
    }

    .recent-box {
        background: #0f172a;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 10px 12px;
    }

    .recent-label {
        color: #94a3b8;
        font-size: 0.72rem;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .recent-value {
        color: #f8fafc;
        font-size: 0.98rem;
        font-weight: 700;
    }

    .stSelectbox label, .stNumberInput label, .stCheckbox label {
        color: #e5e7eb !important;
        font-weight: 600;
    }

    div[data-baseweb="select"] > div {
        background-color: #111827 !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 14px !important;
        color: white !important;
    }

    .stNumberInput input {
        background-color: #111827 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
    }

    .stNumberInput input:disabled {
        background-color: #1f2937 !important;
        color: #e5e7eb !important;
        -webkit-text-fill-color: #e5e7eb !important;
        opacity: 1 !important;
    }

    .stNumberInput input[disabled] {
        opacity: 1 !important;
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    input, textarea {
        color: #ffffff !important;
        caret-color: #ffffff !important;
    }

    input::placeholder {
        color: #9ca3af !important;
    }

    div[data-baseweb="select"] input {
        color: white !important;
    }

    div[data-baseweb="select"] span {
        color: white !important;
    }

    ul[role="listbox"] {
        background-color: #111827 !important;
        color: white !important;
    }

    @media (max-width: 900px) {
        .summary-strip,
        .recent-grid,
        .model-main {
            grid-template-columns: 1fr 1fr;
        }
    }

    @media (max-width: 640px) {
        .summary-strip,
        .recent-grid,
        .model-main {
            grid-template-columns: 1fr;
        }

        .hero-title {
            font-size: 1.7rem;
        }
    }
</style>
""", unsafe_allow_html=True)


CURRENT_SEASON = "2025-26"

BOOKMAKER_MAP = {
    "DraftKings": "draftkings",
    "FanDuel": "fanduel",
    "BetMGM": "betmgm",
    "Caesars": "caesars",
    "ESPN BET": "espnbet",
    "Bovada": "bovada"
}

NBA_TEAMS = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards"
}

TEAM_THEMES = {
    "ATL": {"primary": "#E03A3E", "secondary": "#C1D32F"},
    "BOS": {"primary": "#007A33", "secondary": "#BA9653"},
    "BKN": {"primary": "#000000", "secondary": "#FFFFFF"},
    "CHA": {"primary": "#1D1160", "secondary": "#00788C"},
    "CHI": {"primary": "#CE1141", "secondary": "#000000"},
    "CLE": {"primary": "#860038", "secondary": "#FDBB30"},
    "DAL": {"primary": "#00538C", "secondary": "#B8C4CA"},
    "DEN": {"primary": "#0E2240", "secondary": "#FEC524"},
    "DET": {"primary": "#C8102E", "secondary": "#1D42BA"},
    "GSW": {"primary": "#1D428A", "secondary": "#FFC72C"},
    "HOU": {"primary": "#CE1141", "secondary": "#C4CED4"},
    "IND": {"primary": "#002D62", "secondary": "#FDBB30"},
    "LAC": {"primary": "#C8102E", "secondary": "#1D428A"},
    "LAL": {"primary": "#552583", "secondary": "#FDB927"},
    "MEM": {"primary": "#5D76A9", "secondary": "#12173F"},
    "MIA": {"primary": "#98002E", "secondary": "#F9A01B"},
    "MIL": {"primary": "#00471B", "secondary": "#EEE1C6"},
    "MIN": {"primary": "#0C2340", "secondary": "#236192"},
    "NOP": {"primary": "#0C2340", "secondary": "#C8102E"},
    "NYK": {"primary": "#006BB6", "secondary": "#F58426"},
    "OKC": {"primary": "#007AC1", "secondary": "#EF3B24"},
    "ORL": {"primary": "#0077C0", "secondary": "#C4CED4"},
    "PHI": {"primary": "#006BB6", "secondary": "#ED174C"},
    "PHX": {"primary": "#1D1160", "secondary": "#E56020"},
    "POR": {"primary": "#E03A3E", "secondary": "#000000"},
    "SAC": {"primary": "#5A2D81", "secondary": "#63727A"},
    "SAS": {"primary": "#000000", "secondary": "#C4CED4"},
    "TOR": {"primary": "#CE1141", "secondary": "#000000"},
    "UTA": {"primary": "#002B5C", "secondary": "#F9A01B"},
    "WAS": {"primary": "#002B5C", "secondary": "#E31837"}
}


def get_model_mtime():
    return os.path.getmtime("models/points_regression.pkl")

@st.cache_resource
def load_model(_mtime):
    return joblib.load("models/points_regression.pkl")


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


@st.cache_data
def load_active_players():
    active_players = players.get_active_players()

    actual_name_to_id = {p["full_name"]: p["id"] for p in active_players}

    search_name_to_actual = {}
    for actual_name in actual_name_to_id.keys():
        search_name = normalize_name(actual_name).title()
        search_name_to_actual[search_name] = actual_name

    search_names = sorted(search_name_to_actual.keys())

    return active_players, actual_name_to_id, search_name_to_actual, search_names


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(56,189,248,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def get_team_theme(team_abbr: str):
    return TEAM_THEMES.get(team_abbr, {"primary": "#38bdf8", "secondary": "#60a5fa"})


def get_pick_label(prob_over, prob_under):
    if abs(prob_over - prob_under) <= 0.02:
        return "No Edge", "neutral"
    if prob_over > prob_under:
        return "Lean Over", "over"
    return "Lean Under", "under"


def american_odds_text(price):
    if price is None:
        return "N/A"
    try:
        price = int(price)
        return f"+{price}" if price > 0 else str(price)
    except Exception:
        return str(price)


def run_with_retry(func, retries=3, delay=1.2):
    last_error = None
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(delay)
    raise last_error


def get_player_info_df(player_id):
    return run_with_retry(
        lambda: commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
    )


def get_player_gamelog_df(player_id, season):
    return run_with_retry(
        lambda: playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    )


def get_scoreboard_for_date(target_date_str):
    return run_with_retry(
        lambda: scoreboardv2.ScoreboardV2(game_date=target_date_str)
    )


def get_team_game_info(team_id, team_abbr, target_date_str):
    board = get_scoreboard_for_date(target_date_str)
    game_header = board.game_header.get_data_frame()
    line_score = board.line_score.get_data_frame()

    team_game = game_header[
        (game_header["HOME_TEAM_ID"] == team_id) |
        (game_header["VISITOR_TEAM_ID"] == team_id)
    ]

    if team_game.empty:
        return None

    game = team_game.iloc[0]
    game_id = game["GAME_ID"]

    game_lines = line_score[
        line_score["GAME_ID"] == game_id
    ][["TEAM_ID", "TEAM_ABBREVIATION"]]

    if int(game["HOME_TEAM_ID"]) == team_id:
        opponent_id = int(game["VISITOR_TEAM_ID"])
        opponent_row = game_lines[game_lines["TEAM_ID"] == opponent_id]
        matchup_text = f"{team_abbr} vs {opponent_row.iloc[0]['TEAM_ABBREVIATION']}"
    else:
        opponent_id = int(game["HOME_TEAM_ID"])
        opponent_row = game_lines[game_lines["TEAM_ID"] == opponent_id]
        matchup_text = f"{team_abbr} @ {opponent_row.iloc[0]['TEAM_ABBREVIATION']}"

    game_date = pd.to_datetime(game["GAME_DATE_EST"]).strftime("%B %d, %Y")
    game_time = game["GAME_STATUS_TEXT"]

    return {
        "matchup": matchup_text,
        "date": game_date,
        "time": game_time
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_upcoming_nba_events(api_key):
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    resp = requests.get(url, params={"apiKey": api_key}, timeout=20)
    resp.raise_for_status()
    return resp.json()


def find_matching_event_id(events, matchup_text):
    parts = matchup_text.replace("vs", "@").split("@")
    teams = [p.strip() for p in parts if p.strip()]

    if len(teams) != 2:
        return None

    team1 = NBA_TEAMS.get(teams[0])
    team2 = NBA_TEAMS.get(teams[1])

    if not team1 or not team2:
        return None

    team1 = team1.lower()
    team2 = team2.lower()

    for event in events:
        home = event.get("home_team", "").lower()
        away = event.get("away_team", "").lower()

        if (team1 in home and team2 in away) or (team2 in home and team1 in away):
            return event.get("id")

    return None


@st.cache_data(ttl=300, show_spinner=False)
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


def extract_player_prop(event_odds_json, selected_player):
    target_name = normalize_name(selected_player)

    for bookmaker in event_odds_json.get("bookmakers", []):
        book_title = bookmaker.get("title", "Unknown")

        for market in bookmaker.get("markets", []):
            if market.get("key") != "player_points":
                continue

            book_last_update = market.get("last_update", "")
            over_outcomes = []
            under_outcomes = []

            for outcome in market.get("outcomes", []):
                outcome_name = normalize_name(outcome.get("description", ""))
                if outcome_name != target_name:
                    continue

                if outcome.get("name") == "Over":
                    over_outcomes.append(outcome)
                elif outcome.get("name") == "Under":
                    under_outcomes.append(outcome)

            for over in over_outcomes:
                over_point = over.get("point")
                for under in under_outcomes:
                    if under.get("point") == over_point:
                        return {
                            "line": float(over_point),
                            "over_price": over.get("price"),
                            "under_price": under.get("price"),
                            "bookmaker": book_title,
                            "last_update": book_last_update
                        }

    return None

model = load_model(get_model_mtime())
model_stats = load_model_stats(get_model_stats_mtime())
points_std = model_stats["std_dev"]
_, player_name_map, search_name_to_actual, player_search_names = load_active_players()

st.write("Model modified:", time.ctime(get_model_mtime()))

st.markdown(f"""
<div class="hero">
    <div class="hero-title">NBA Points Prop Predictor</div>
    <p class="hero-subtitle">Search a player, pull the latest line, and compare it to our model.</p>
    <div class="hero-pills">
        <div class="hero-pill">Live sportsbook lookup</div>
        <div class="hero-pill">Compare Model vs line edge</div>
        <div class="hero-pill">{APP_VERSION}</div>
    </div>
</div>
""", unsafe_allow_html=True)


selected_search_name = st.selectbox(
    "Search Player",
    options=player_search_names,
    index=None,
    placeholder="Start typing a player name..."
)

selected_player = search_name_to_actual[selected_search_name] if selected_search_name else None

selected_book = st.selectbox(
    "Sportsbook",
    options=list(BOOKMAKER_MAP.keys()),
    index=0
)

odds_api_key = os.getenv("ODDS_API_KEY")

if selected_player:
    try:
        player_id = player_name_map[selected_player]

        player_info = get_player_info_df(player_id)
        team_id = int(player_info.loc[0, "TEAM_ID"])
        team_abbr = player_info.loc[0, "TEAM_ABBREVIATION"]
        team_theme = get_team_theme(team_abbr)

        primary = team_theme["primary"]
        secondary = team_theme["secondary"]

        model_bg = (
            f"linear-gradient(135deg, "
            f"{hex_to_rgba(primary, 0.35)} 0%, "
            f"{hex_to_rgba(secondary, 0.25)} 50%, "
            f"rgba(15, 23, 42, 0.95) 100%)"
        )

        model_border = primary
        model_glow = hex_to_rgba(primary, 0.28)
        model_stat_bg = "rgba(255, 255, 255, 0.06)"
        model_stat_border = hex_to_rgba(secondary, 0.32)
        model_label_color = "#cbd5e1"

        eastern = pytz.timezone("US/Eastern")
        now_et = datetime.now(eastern)

        if now_et.hour < 4:
            now_et = now_et - pd.Timedelta(days=1)

        today_str = now_et.strftime("%m/%d/%Y")
        today_game_info = get_team_game_info(team_id, team_abbr, today_str)

        if today_game_info:
            game_status = "Game today"
            matchup = today_game_info["matchup"]
            game_date = today_game_info["date"]
            game_time = today_game_info["time"]
        else:
            next_game_info = None
            for i in range(1, 8):
                future_date = now_et + pd.Timedelta(days=i)
                future_date_str = future_date.strftime("%m/%d/%Y")
                next_game_info = get_team_game_info(team_id, team_abbr, future_date_str)
                if next_game_info:
                    break

            if next_game_info:
                game_status = "No game today"
                matchup = next_game_info["matchup"]
                game_date = next_game_info["date"]
                game_time = next_game_info["time"]
            else:
                game_status = "No game found"
                matchup = "N/A"
                game_date = "N/A"
                game_time = "N/A"

        sportsbook_line = None
        over_price = None
        under_price = None
        book_name = selected_book
        book_updated = None
        line_source = "Manual"

        game_available_in_feed = False

        if odds_api_key and matchup != "N/A":
            try:
                events = fetch_upcoming_nba_events(odds_api_key)
                event_id = find_matching_event_id(events, matchup)

                game_available_in_feed = event_id is not None

                if event_id:
                    event_odds = fetch_player_points_market(
                        odds_api_key,
                        event_id,
                        BOOKMAKER_MAP[selected_book]
                    )

                    prop = extract_player_prop(event_odds, selected_player)

                    if prop:
                        sportsbook_line = prop["line"]
                        over_price = prop["over_price"]
                        under_price = prop["under_price"]
                        book_name = prop["bookmaker"]
                        book_updated = prop["last_update"]
                        line_source = "Sportsbook API"

            except Exception:
                sportsbook_line = None

        default_line = sportsbook_line if sportsbook_line is not None else 20.5

        col_line, col_toggle = st.columns([4, 1])

        with col_toggle:
            manual_override = st.checkbox("Manual line", value=False)

        with col_line:
            line = st.number_input(
                "Points line",
                min_value=0.0,
                value=float(default_line),
                step=0.5,
                disabled=(sportsbook_line is not None and not manual_override)
            )

        if sportsbook_line is not None and not manual_override:
            line = sportsbook_line
        elif sportsbook_line is None:
            line_source = "Manual fallback"

        df = get_player_gamelog_df(player_id, CURRENT_SEASON)

        if df.empty:
            st.warning("No game log found for this player yet.")
            st.stop()

        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values("GAME_DATE").reset_index(drop=True)

        numeric_cols = [
            "PTS", "FGM", "FGA", "FTA", "FTM", "OREB", "DREB",
            "STL", "AST", "BLK", "PF", "TOV", "MIN"
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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
        
        df["PLAYER_NAME"] = selected_player
        
        df["player_avg_pts"] = df.groupby("PLAYER_NAME")["PTS"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df["player_avg_pts_sq"] = df["player_avg_pts"] ** 2
        
        df["last3_pts"] = df.groupby("PLAYER_NAME")["PTS"].transform(
            lambda x: x.shift(1).rolling(3).mean()
        )
        df["last5_pts"] = df.groupby("PLAYER_NAME")["PTS"].transform(
            lambda x: x.shift(1).rolling(5).mean()
        )
        df["last10_pts"] = df.groupby("PLAYER_NAME")["PTS"].transform(
            lambda x: x.shift(1).rolling(10).mean()
        )
        df["last20_pts"] = df.groupby("PLAYER_NAME")["PTS"].transform(
            lambda x: x.shift(1).rolling(20).mean()
        )
        
        df["last5_fga"] = df.groupby("PLAYER_NAME")["FGA"].transform(
            lambda x: x.shift(1).rolling(5).mean()
        )
        df["last5_fta"] = df.groupby("PLAYER_NAME")["FTA"].transform(
            lambda x: x.shift(1).rolling(5).mean()
        )
        df["last5_minutes"] = df.groupby("PLAYER_NAME")["MIN"].transform(
            lambda x: x.shift(1).rolling(5).mean()
        )
        df["last5_gmsc"] = df.groupby("PLAYER_NAME")["gmsc"].transform(
            lambda x: x.shift(1).rolling(5).mean()
        )
        
        df["home_game"] = df["MATCHUP"].str.contains("vs").astype(int)
        
        df["days_rest"] = df.groupby("PLAYER_NAME")["GAME_DATE"].diff().dt.days
        df["days_rest"] = df["days_rest"].fillna(3)
        df["is_back_to_back"] = (df["days_rest"] == 1).astype(int)
        
        df["usage_proxy"] = df["FGA"] + 0.44 * df["FTA"] + df["TOV"]
        df["last5_usage_proxy"] = df.groupby("PLAYER_NAME")["usage_proxy"].transform(
            lambda x: x.shift(1).rolling(5).mean()
        )
        
        df["season_minutes_avg"] = df.groupby("PLAYER_NAME")["MIN"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        
        df["minutes_volatility"] = df.groupby("PLAYER_NAME")["MIN"].transform(
            lambda x: x.shift(1).rolling(5).std()
        )
        df["points_volatility"] = df.groupby("PLAYER_NAME")["PTS"].transform(
            lambda x: x.shift(1).rolling(5).std()
        )
        
        if "FG3A" in df.columns:
            df["FG3A"] = pd.to_numeric(df["FG3A"], errors="coerce")
            df["last5_3pa"] = df.groupby("PLAYER_NAME")["FG3A"].transform(
                lambda x: x.shift(1).rolling(5).mean()
            )

        
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
            st.warning("Not enough recent games to build features yet.")
            st.stop()

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

        X = pd.DataFrame([feature_data])

        if hasattr(model, "feature_names_in_"):
            X = X.reindex(columns=model.feature_names_in_, fill_value=0)

        predicted_points = float(model.predict(X)[0])
        edge = predicted_points - line

        prob_over = 1 - norm.cdf(line, loc=predicted_points, scale=points_std)
        prob_under = 1 - prob_over
        pick_text, pick_kind = get_pick_label(prob_over, prob_under)

        if pick_kind == "over":
            pick_bg = "rgba(34,197,94,0.25)"
            pick_border = "#22c55e"
            pick_text_color = "#22c55e"
        elif pick_kind == "under":
            pick_bg = "rgba(239,68,68,0.25)"
            pick_border = "#ef4444"
            pick_text_color = "#ef4444"
        else:
            pick_bg = "rgba(148,163,184,0.12)"
            pick_border = "#94a3b8"
            pick_text_color = "#e5e7eb"

        interpretation_text = (
            "The model sees no meaningful edge either way, with the probability split essentially even."
            if abs(prob_over - prob_under) <= 0.02
            else f"The model projects a {prob_over:.0%} chance of the over hitting compared to {prob_under:.0%} for the under."
        )

        model_html = "\n".join([
            f'<div class="model-card" style="background: {model_bg}; border: 3px solid {model_border}; box-shadow: 0 0 0 1px {hex_to_rgba(secondary, 0.16)}, 0 0 28px {model_glow}, 0 0 50px {hex_to_rgba(primary, 0.18)};">',
            f'<div class="model-title" style="color: #ffffff;">{selected_player}</div>',
            '<div class="model-subtitle">Model Output</div>',
            '<div class="model-main">',
            f'<div class="model-stat" style="background: {model_stat_bg}; border: 1px solid {model_stat_border};"><div class="model-stat-label" style="color: {model_label_color};">Predicted Points</div><div class="model-stat-value">{predicted_points:.2f}</div></div>',
            f'<div class="model-stat" style="background: {model_stat_bg}; border: 1px solid {model_stat_border};"><div class="model-stat-label" style="color: {model_label_color};">Sportsbook Line</div><div class="model-stat-value">{line:.1f}</div></div>',
            f'<div class="model-stat" style="background: {model_stat_bg}; border: 1px solid {model_stat_border};"><div class="model-stat-label" style="color: {model_label_color};">Model Edge</div><div class="model-stat-value">{edge:+.2f}</div></div>',
            f'<div class="model-stat" style="background: {model_stat_bg}; border: 1px solid {model_stat_border};"><div class="model-stat-label" style="color: {model_label_color};">Probability Split</div><div class="model-stat-value">O {prob_over:.1%} / U {prob_under:.1%}</div></div>',
            '</div>',
            f'<div class="prob-interpretation" style="margin-top: 8px; margin-bottom: 10px; padding: 8px 2px 0 2px; font-size: 0.98rem; color: #cbd5e1; opacity: 0.95;">{interpretation_text}</div>',
            f'<div class="pick-banner" style="background: {pick_bg}; color: {pick_text_color}; border: 2px solid {pick_border};">{pick_text}</div>',
            '<div class="small-note">Trained regression model output compared against the current sportsbook line.</div>',
            '</div>'
        ])
        st.markdown(model_html, unsafe_allow_html=True)

        st.markdown(f"""
<div class="section-card">
    <div class="section-title">Game Info</div>
    <div class="summary-strip">
        <div class="summary-item">
            <div class="summary-label">Status</div>
            <div class="summary-value">{game_status}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Matchup</div>
            <div class="summary-value">{matchup}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Date</div>
            <div class="summary-value">{game_date}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Time</div>
            <div class="summary-value">{game_time}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        update_text = book_updated if book_updated else "N/A"

        sportsbook_message = ""
        if not odds_api_key:
            sportsbook_message = "ODDS_API_KEY not found. Using manual line only."
        elif matchup != "N/A" and not game_available_in_feed:
            sportsbook_message = "This game is not yet available in the sportsbook events feed. Using manual fallback."
        elif sportsbook_line is None:
            sportsbook_message = "Game found, but no player points line was posted for this player/book yet. Using manual fallback."

        st.markdown(f"""
<div class="section-card">
    <div class="section-title">Sportsbook Line</div>
    <div class="summary-strip">
        <div class="summary-item">
            <div class="summary-label">Book</div>
            <div class="summary-value">{book_name}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Source</div>
            <div class="summary-value">{line_source}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Line</div>
            <div class="summary-value">{line:.1f}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Prices</div>
            <div class="summary-value">O {american_odds_text(over_price)} / U {american_odds_text(under_price)}</div>
        </div>
    </div>
    <div class="small-note">Last update: {update_text}</div>
</div>
""", unsafe_allow_html=True)

        if sportsbook_message:
            st.info(sportsbook_message)

        recent_games = df.sort_values("GAME_DATE", ascending=False).head(5).copy()
        recent_games["GAME_DATE"] = recent_games["GAME_DATE"].dt.strftime("%Y-%m-%d")

        st.markdown(f"""
<div class="section-card">
    <div class="section-title">Scoring Snapshot</div>
    <div class="recent-grid">
        <div class="recent-box">
            <div class="recent-label">Season Avg PPG</div>
            <div class="recent-value">{latest["player_avg_pts"]:.2f}</div>
        </div>
        <div class="recent-box">
            <div class="recent-label">Last 5 Avg PTS</div>
            <div class="recent-value">{latest["last5_pts"]:.2f}</div>
        </div>
        <div class="recent-box">
            <div class="recent-label">Last 5 Avg MIN</div>
            <div class="recent-value">{latest["last5_minutes"]:.2f}</div>
        </div>
        <div class="recent-box">
            <div class="recent-label">Last 5 Avg GmSc</div>
            <div class="recent-value">{latest["last5_gmsc"]:.2f}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        st.dataframe(
            recent_games[["GAME_DATE", "MATCHUP", "PTS", "MIN", "FGA", "FTA"]],
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.error(f"Something went wrong: {e}")

else:
    st.markdown("""
<div class="section-card">
    <div class="section-title">Get Started</div>
    <div class="small-note">
        Select a player to load game info, sportsbook line, and prediction.
    </div>
</div>
""", unsafe_allow_html=True)
