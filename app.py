import streamlit as st
import pandas as pd
import joblib
import json
from scipy.stats import norm
from datetime import datetime
import pytz

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

    .stSelectbox label, .stNumberInput label {
        color: #e5e7eb !important;
        font-weight: 600;
    }

    div[data-baseweb="select"] > div {
        background-color: #111827 !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 14px !important;
        color: white !important;
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
# Helpers
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/points_regression.pkl")


@st.cache_data
def load_model_stats():
    with open("models/points_model_stats.json", "r") as f:
        return json.load(f)


@st.cache_data
def load_players():
    all_players = players.get_players()
    name_map = {p["full_name"]: p["id"] for p in all_players}
    return all_players, name_map, sorted(name_map.keys())


def get_pick_label(prob_over, prob_under):
    if prob_over >= 0.60:
        return "Lean Over", "pick-over"
    if prob_under >= 0.60:
        return "Lean Under", "pick-under"
    return "No Edge", "pick-none"


def get_team_game_info(team_id, team_abbr, target_date_str):
    board = scoreboardv2.ScoreboardV2(game_date=target_date_str)
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


# -----------------------------
# Load resources
# -----------------------------
model = load_model()
model_stats = load_model_stats()
points_std = model_stats["std_dev"]

_, player_name_map, player_names = load_players()


# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">NBA Points Prop Predictor</div>
    <p class="hero-subtitle">Search a player, set the line, and get a quick model-based lean.</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Inputs
# -----------------------------
st.caption("Search for a player by name")

selected_player = st.selectbox(
    "Player",
    options=player_names,
    index=None,
    placeholder="Start typing a player name..."
)

line = st.number_input(
    "Enter points line",
    min_value=0.0,
    value=20.5,
    step=0.5
)


# -----------------------------
# Main app
# -----------------------------
if selected_player:
    try:
        player_id = player_name_map[selected_player]

        # -----------------------------
        # Current player/team info
        # -----------------------------
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
        team_id = int(player_info.loc[0, "TEAM_ID"])
        team_abbr = player_info.loc[0, "TEAM_ABBREVIATION"]

        eastern = pytz.timezone("US/Eastern")
        now_et = datetime.now(eastern)

        if now_et.hour < 4:
            now_et = now_et - pd.Timedelta(days=1)

        today_str = now_et.strftime("%m/%d/%Y")
        today_game_info = get_team_game_info(team_id, team_abbr, today_str)

        game_status = ""
        matchup = ""
        game_date = ""
        game_time = ""

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

        # -----------------------------
        # Game log / model features
        # -----------------------------
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season="2025-26"
        )

        df = gamelog.get_data_frames()[0]

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

        # Game Score
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

        # Exact feature names from your trained model
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

        # -----------------------------
        # Game info card
        # -----------------------------
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

        # -----------------------------
        # Prediction card
        # -----------------------------
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

        st.markdown(
            '<div class="small-note">This is a model lean, not guaranteed betting advice.</div>',
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

        # -----------------------------
        # Recent form card
        # -----------------------------
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
            Select a player above to load game info and generate a prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)
