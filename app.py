import os
import re
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


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="NBA Points Prop Predictor",
    page_icon="🏀",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #081120 0%, #0f172a 100%);
        color: #f8fafc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }

    h1, h2, h3 {
        color: #f8fafc;
        letter-spacing: 0.2px;
    }

    .hero {
        background: linear-gradient(135deg, #111827 0%, #1e293b 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 26px 24px 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.28);
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 6px;
        color: #ffffff;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .section-card {
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 20px;
        margin-top: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.22);
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 14px;
        color: #f8fafc;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 8px;
    }

    .stat-box {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 14px;
    }

    .stat-label {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-bottom: 6px;
    }

    .stat-value {
        color: #f8fafc;
        font-size: 1.12rem;
        font-weight: 700;
    }

    .pick-banner {
        margin-top: 18px;
        border-radius: 16px;
        padding: 16px 18px;
        font-size: 1.05rem;
        font-weight: 700;
        text-align: center;
    }

    .pick-over {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.25);
    }

    .pick-under {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.22);
    }

    .pick-none {
        background: rgba(59, 130, 246, 0.14);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.22);
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.86rem;
        margin-top: 10px;
    }

    .selected-player {
        margin-top: 12px;
        padding: 10px 14px;
        border-radius: 12px;
        background: rgba(59, 130, 246, 0.14);
        border: 1px solid rgba(96, 165, 250, 0.22);
        color: #dbeafe;
        font-weight: 600;
    }

    .suggestions-line {
        margin-top: 8px;
        color: #94a3b8;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .stSelectbox label, .stNumberInput label, .stCheckbox label, .stTextInput label {
        color: #e5e7eb !important;
        font-weight: 600;
    }

    .stTextInput > div > div > input {
        background-color: #111827 !important;
        color: white !important;
        border-radius: 14px !important;
    }

    .stNumberInput > div > div > input {
        background-color: #111827 !important;
        color: white !important;
        border-radius: 14px !important;
    }

    .stNumberInput button {
        background-color: #1f2937 !important;
        color: white !important;
        border: none !important;
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Constants
# -----------------------------
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


# -----------------------------
# Session state
# -----------------------------
if "selected_player" not in st.session_state:
    st.session_state.selected_player = None

if "excluded_players" not in st.session_state:
    st.session_state.excluded_players = set()


# -----------------------------
# Helpers / cache
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/points_regression.pkl")


@st.cache_data
def load_model_stats():
    with open("models/points_model_stats.json", "r") as f:
        return json.load(f)


@st.cache_data
def load_active_players():
    active_players = players.get_active_players()
    name_map = {p["full_name"]: p["id"] for p in active_players}
    return active_players, name_map, sorted(name_map.keys())


def normalize_name(name: str) -> str:
    if not name:
        return ""

    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(char for char in name if not unicodedata.combining(char))
    name = name.lower()
    name = name.replace(".", "")
    name = name.replace("’", "'")
    name = name.replace("-", " ")
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def rank_player_match(player_name: str, query: str):
    norm_name = normalize_name(player_name)
    tokens = norm_name.split()

    if norm_name == query:
        return (0, 0, len(norm_name))

    if norm_name.startswith(query):
        return (1, 0, len(norm_name))

    for i, token in enumerate(tokens):
        if token.startswith(query):
            return (2, i, len(norm_name))

    if query in norm_name:
        return (3, norm_name.find(query), len(norm_name))

    return (99, 999, 999)


def get_player_suggestions(search_text, player_names, max_results=6):
    if not search_text or len(search_text.strip()) < 2:
        return []

    query = normalize_name(search_text)
    matches = []

    for name in player_names:
        if name in st.session_state.excluded_players:
            continue

        score = rank_player_match(name, query)
        if score[0] < 99:
            matches.append((score, name))

    matches.sort(key=lambda x: x[0])
    return [name for _, name in matches[:max_results]]


def get_pick_label(prob_over, prob_under):
    if prob_over >= 0.60:
        return "Lean Over", "pick-over"
    if prob_under >= 0.60:
        return "Lean Under", "pick-under"
    return "No Edge", "pick-none"


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

    full_1 = NBA_TEAMS.get(teams[0])
    full_2 = NBA_TEAMS.get(teams[1])

    for event in events:
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        if {home_team, away_team} == {full_1, full_2}:
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
                if normalize_name(outcome.get("description", "")) != target_name:
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


# -----------------------------
# Load resources
# -----------------------------
model = load_model()
model_stats = load_model_stats()
points_std = model_stats["std_dev"]
_, player_name_map, player_names = load_active_players()


# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">NBA Points Prop Predictor</div>
    <p class="hero-subtitle">Search a player, auto-load the sportsbook line, and compare it to the model.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Search UI
# -----------------------------
st.caption("Search for a player by name")

player_search = st.text_input(
    "Player",
    value="",
    placeholder="Start typing a player name...",
    key="player_search"
)

suggestions = get_player_suggestions(player_search, player_names)
selected_player = None

if len(player_search.strip()) >= 2:
    if suggestions:
        selected_player = suggestions[0]
        st.session_state.selected_player = selected_player

        if len(suggestions) > 1:
            st.markdown(
                '<div class="suggestions-line">Suggestions: ' + " • ".join(suggestions[1:]) + '</div>',
                unsafe_allow_html=True
            )
    else:
        st.session_state.selected_player = None
        st.markdown(
            '<div class="small-note">No players matched that search.</div>',
            unsafe_allow_html=True
        )
else:
    st.session_state.selected_player = None

selected_player = st.session_state.selected_player

if selected_player:
    st.markdown(
        f'<div class="selected-player">Selected player: {selected_player}</div>',
        unsafe_allow_html=True
    )

selected_book = st.selectbox(
    "Sportsbook",
    options=list(BOOKMAKER_MAP.keys()),
    index=0
)

manual_override = st.checkbox("Manually override sportsbook line", value=False)
odds_api_key = os.getenv("ODDS_API_KEY")


# -----------------------------
# Main app
# -----------------------------
if selected_player:
    try:
        player_id = player_name_map[selected_player]

        player_info = get_player_info_df(player_id)
        team_id = int(player_info.loc[0, "TEAM_ID"])
        team_abbr = player_info.loc[0, "TEAM_ABBREVIATION"]

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

        if odds_api_key and matchup != "N/A":
            try:
                events = fetch_upcoming_nba_events(odds_api_key)
                event_id = find_matching_event_id(events, matchup)

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
            st.session_state.excluded_players.add(selected_player)
            st.session_state.selected_player = None
            st.warning("That player does not have a current-season game log yet, so they were removed from suggestions.")
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

        df["player_avg_pts"] = df["PTS"].shift(1).expanding().mean()
        df["last5_pts"] = df["PTS"].shift(1).rolling(5).mean()
        df["last5_fga"] = df["FGA"].shift(1).rolling(5).mean()
        df["last5_fta"] = df["FTA"].shift(1).rolling(5).mean()
        df["last5_minutes"] = df["MIN"].shift(1).rolling(5).mean()
        df["last5_gmsc"] = df["gmsc"].shift(1).rolling(5).mean()

        df_features = df.dropna().reset_index(drop=True)

        if df_features.empty:
            st.warning("Not enough recent games to build features yet.")
            st.stop()

        latest = df_features.iloc[-1]

        X = pd.DataFrame([{
            "player_avg_pts": latest["player_avg_pts"],
            "last5_pts": latest["last5_pts"],
            "last5_fga": latest["last5_fga"],
            "last5_fta": latest["last5_fta"],
            "last5_minutes": latest["last5_minutes"],
            "last5_gmsc": latest["last5_gmsc"]
        }])

        predicted_points = float(model.predict(X)[0])
        edge = predicted_points - line

        prob_over = 1 - norm.cdf(line, loc=predicted_points, scale=points_std)
        prob_under = 1 - prob_over
        pick_text, pick_class = get_pick_label(prob_over, prob_under)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Game Info</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-label">Status</div>
                <div class="stat-value">{game_status}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Matchup</div>
                <div class="stat-value">{matchup}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Date</div>
                <div class="stat-value">{game_date}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Time</div>
                <div class="stat-value">{game_time}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Sportsbook Line</div>', unsafe_allow_html=True)
        update_text = book_updated if book_updated else "N/A"
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-label">Book</div>
                <div class="stat-value">{book_name}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Line Source</div>
                <div class="stat-value">{line_source}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Points Line</div>
                <div class="stat-value">{line:.1f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Prices</div>
                <div class="stat-value">O {american_odds_text(over_price)} / U {american_odds_text(under_price)}</div>
            </div>
        </div>
        <div class="small-note">Last update: {update_text}</div>
        """, unsafe_allow_html=True)

        if not odds_api_key:
            st.info("ODDS_API_KEY not found. Using manual line only.")
        elif sportsbook_line is None:
            st.info("No player points line found for this player/book yet. Using manual fallback.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-label">Predicted Points</div>
                <div class="stat-value">{predicted_points:.2f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Line</div>
                <div class="stat-value">{line:.1f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Edge</div>
                <div class="stat-value">{edge:+.2f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Confidence Split</div>
                <div class="stat-value">O {prob_over:.1%} / U {prob_under:.1%}</div>
            </div>
        </div>

        <div class="pick-banner {pick_class}">
            {pick_text}
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="small-note">This is a model lean, not guaranteed betting advice.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        recent_games = df.sort_values("GAME_DATE", ascending=False).head(5).copy()
        recent_games["GAME_DATE"] = recent_games["GAME_DATE"].dt.strftime("%Y-%m-%d")

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recent Form</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-label">Avg Points</div>
                <div class="stat-value">{latest["player_avg_pts"]:.2f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Last 5 Points</div>
                <div class="stat-value">{latest["last5_pts"]:.2f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Last 5 Minutes</div>
                <div class="stat-value">{latest["last5_minutes"]:.2f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Last 5 GmSc</div>
                <div class="stat-value">{latest["last5_gmsc"]:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            recent_games[["GAME_DATE", "MATCHUP", "PTS", "MIN", "FGA", "FTA"]],
            use_container_width=True,
            hide_index=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Something went wrong: {e}")

else:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Get Started</div>
        <div class="small-note">
            Start typing at least 2 letters to search players.
        </div>
    </div>
    """, unsafe_allow_html=True)
