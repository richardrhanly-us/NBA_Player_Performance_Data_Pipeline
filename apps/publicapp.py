import sys
import os
import pandas as pd
import streamlit as st
import gspread

from scipy.stats import norm
from google.oauth2.service_account import Credentials

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shared_app import (
    APP_VERSION,
    CURRENT_SEASON,
    normalize_name,
    load_model,
    load_model_stats,
    load_active_players,
    get_strong_plays_summary,
    get_strong_plays_health,
    get_player_gamelog_df,
    get_player_info_df,
    build_player_feature_row,
    get_live_player_stats,
    get_available_sportsbooks,
    get_player_points_lines,
)

try:
    from src.shared_app import get_team_game_info
except ImportError:
    get_team_game_info = None


SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SHEET_KEY = "1uhjV_Si-qcILfNJbKZrD52y4JnT_GvqQ0hzN7POekQM"

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
    "WAS": {"primary": "#002B5C", "secondary": "#E31837"},
}


st.set_page_config(
    page_title="NBA Edge Lab",
    page_icon="🏀",
    layout="centered",
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

    .metric-box {
        background: rgba(15,23,42,0.78);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 12px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 800;
    }

    .mini-card {
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 10px;
    }

    .mini-title {
        color: #cbd5e1;
        font-size: 0.84rem;
        margin-bottom: 6px;
    }

    .mini-value {
        color: #f8fafc;
        font-size: 1.2rem;
        font-weight: 800;
    }

    .muted {
        color: #94a3b8;
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
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
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
        width: 100%;
        display: block;
        box-sizing: border-box;
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.84rem;
        margin-top: 10px;
    }

    .top-play-card {
        background: rgba(15,23,42,0.88);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18);
    }

    .top-play-title {
        font-size: 1.02rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 6px;
    }

    .top-play-sub {
        color: #cbd5e1;
        font-size: 0.92rem;
        margin-bottom: 4px;
    }

    .top-play-meta {
        color: #94a3b8;
        font-size: 0.88rem;
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

    .stNumberInput input {
        background-color: #111827 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #1e293b 0%, #111827 100%) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 14px !important;
        padding: 0.65rem 1.25rem !important;
        font-weight: 700 !important;
        font-size: 0.98rem !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.18) !important;
    }

    @media (max-width: 640px) {
        .hero-title {
            font-size: 1.7rem;
        }
    }
</style>
""", unsafe_allow_html=True)


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


def get_pick_label(edge):
    abs_edge = abs(edge)
    if abs_edge < 1.5:
        return "No Bet", "neutral"
    if abs_edge < 3.0:
        return ("Lean Over", "over") if edge > 0 else ("Lean Under", "under")
    return ("Strong Over", "over") if edge > 0 else ("Strong Under", "under")


def safe_live_display(value, fallback="N/A"):
    if value is None:
        return fallback
    if isinstance(value, str) and not value.strip():
        return fallback
    return str(value)


def format_minutes(minutes_str):
    if not minutes_str:
        return "0:00"

    try:
        if ":" in str(minutes_str):
            return str(minutes_str)

        m = 0
        s = 0
        text = str(minutes_str).replace("PT", "")

        if "M" in text:
            m_part = text.split("M")[0]
            m = int(float(m_part)) if m_part else 0
            text = text.split("M")[1]

        if "S" in text:
            s_part = text.replace("S", "")
            s = int(float(s_part)) if s_part else 0

        return f"{m}:{s:02d}"
    except Exception:
        return str(minutes_str)


@st.cache_resource
def get_gsheet_client():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES
    )
    return gspread.authorize(creds)


@st.cache_data(ttl=300)
def get_top_plays_live_df():
    client = get_gsheet_client()
    sheet = client.open_by_key(SHEET_KEY).worksheet("Top Plays Live")
    values = sheet.get_all_values()

    if not values or len(values) < 2:
        return pd.DataFrame()

    headers = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=headers)

    numeric_cols = ["sportsbook_line", "predicted_points", "edge"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_player_lookup():
    actual_name_to_id, _ = load_active_players()
    player_names = sorted(actual_name_to_id.keys())
    return actual_name_to_id, player_names


@st.cache_data(ttl=120)
def build_prediction(player_name, sportsbook_line):
    model = load_model()
    model_stats = load_model_stats()
    actual_name_to_id, normalized_to_actual = load_active_players()

    normalized = normalize_name(player_name)
    actual_name = normalized_to_actual.get(normalized, player_name)
    player_id = actual_name_to_id.get(actual_name)

    if not player_id:
        return {"error": "Player ID not found."}

    gamelog_df = get_player_gamelog_df(player_id, CURRENT_SEASON)
    if gamelog_df is None or gamelog_df.empty:
        return {"error": "Player gamelog unavailable."}

    X = build_player_feature_row(gamelog_df, actual_name)
    if X is None or X.empty:
        return {"error": "Not enough games to build features."}

    model_feature_names = list(getattr(model, "feature_names_in_", []))
    if model_feature_names:
        X = X.reindex(columns=model_feature_names, fill_value=0)

    predicted_points = float(model.predict(X)[0])

    points_std = None
    if isinstance(model_stats, dict):
        points_std = model_stats.get("std_dev")

    over_prob = None
    under_prob = None
    if sportsbook_line is not None and points_std:
        try:
            over_prob = 1 - norm.cdf(sportsbook_line, loc=predicted_points, scale=points_std)
            under_prob = norm.cdf(sportsbook_line, loc=predicted_points, scale=points_std)
        except Exception:
            over_prob = None
            under_prob = None

    season_avg = None
    last5_avg = None
    games_used = len(gamelog_df)

    try:
        gamelog_df = gamelog_df.copy()
        gamelog_df["PTS"] = pd.to_numeric(gamelog_df["PTS"], errors="coerce")
        season_avg = float(gamelog_df["PTS"].mean())
        last5_avg = float(gamelog_df["PTS"].tail(5).mean())
    except Exception:
        pass

    live_stats = None
    try:
        live_stats = get_live_player_stats(actual_name)
    except Exception:
        live_stats = None

    team_info = None
    if get_team_game_info is not None:
        try:
            team_info = get_team_game_info(actual_name)
        except Exception:
            team_info = None

    player_info_df = None
    try:
        player_info_df = get_player_info_df(player_id)
    except Exception:
        player_info_df = None

    team_name = None
    team_abbr = None
    if player_info_df is not None and not player_info_df.empty:
        try:
            if "TEAM_NAME" in player_info_df.columns:
                team_name = player_info_df.iloc[0]["TEAM_NAME"]
            if "TEAM_ABBREVIATION" in player_info_df.columns:
                team_abbr = player_info_df.iloc[0]["TEAM_ABBREVIATION"]
        except Exception:
            pass

    return {
        "actual_name": actual_name,
        "predicted_points": predicted_points,
        "sportsbook_line": sportsbook_line,
        "edge": predicted_points - sportsbook_line if sportsbook_line is not None else None,
        "over_prob": over_prob,
        "under_prob": under_prob,
        "season_avg": season_avg,
        "last5_avg": last5_avg,
        "games_used": games_used,
        "live_stats": live_stats,
        "team_info": team_info,
        "team_name": team_name,
        "team_abbr": team_abbr,
    }


st.markdown(f"""
<div class="hero">
    <div class="hero-title">NBA Edge Lab</div>
    <p class="hero-subtitle">Model-driven player insights and top play signals.</p>
    <div class="hero-pills">
        <div class="hero-pill">Projected points</div>
        <div class="hero-pill">Top edges</div>
        <div class="hero-pill">Version: {APP_VERSION}</div>
    </div>
</div>
""", unsafe_allow_html=True)

top_games_win_rate, top_games_total = get_strong_plays_summary()
health = get_strong_plays_health()

st.markdown('<div class="section-card"><div class="section-title">Top Plays Today</div>', unsafe_allow_html=True)

if top_games_win_rate is not None:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">Win Rate for Top Games</div>
            <div class="metric-value">
                {top_games_win_rate:.1f}% <span style="color: #94a3b8; font-size: 0.9rem; font-weight: 600;">({top_games_total} graded games)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div class="metric-box">
            <div class="metric-label">Win Rate for Top Games</div>
            <div class="metric-value">
                N/A <span style="color: #94a3b8; font-size: 0.9rem; font-weight: 600;">(no graded games yet)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

if health:
    last_update_str = (
        health["last_update"].strftime("%b %d, %I:%M %p")
        if health["last_update"] is not None
        else "N/A"
    )
    st.caption(
        f"Health Check: Last update {last_update_str} | "
        f"Graded {health.get('graded', 0)} | Pending {health.get('pending', 0)}"
    )

try:
    top_plays_df = get_top_plays_live_df()

    if top_plays_df.empty:
        st.info("No top plays available right now.")
    else:
        st.markdown("### ⭐ Top 3 Plays")

        top3 = top_plays_df.head(3)
        for _, row in top3.iterrows():
            edge_val = pd.to_numeric(row.get("edge"), errors="coerce")
            pred_val = pd.to_numeric(row.get("predicted_points"), errors="coerce")
            line_val = pd.to_numeric(row.get("sportsbook_line"), errors="coerce")
            player_name = row.get("PLAYER_NAME", "Player")
            pick = row.get("model_pick", "")
            matchup = f"{row.get('away_team', '')} @ {row.get('home_team', '')}"

            line_text = f"{line_val:.1f}" if pd.notna(line_val) else "N/A"
            pred_text = f"{pred_val:.2f}" if pd.notna(pred_val) else "N/A"
            edge_text = f"{edge_val:+.2f}" if pd.notna(edge_val) else "N/A"

            st.markdown(
                f"""
                <div class="top-play-card">
                    <div class="top-play-title">{player_name} — {pick} {line_text}</div>
                    <div class="top-play-sub">{matchup}</div>
                    <div class="top-play-meta">Projection: {pred_text} | Edge: {edge_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        display_cols = [
            col for col in [
                "PLAYER_NAME",
                "away_team",
                "home_team",
                "sportsbook_line",
                "predicted_points",
                "edge",
                "model_pick",
                "sportsbook",
            ] if col in top_plays_df.columns
        ]

        display_df = top_plays_df[display_cols].copy()
        display_df = display_df.rename(columns={
            "PLAYER_NAME": "Player",
            "away_team": "Away",
            "home_team": "Home",
            "sportsbook_line": "Line",
            "predicted_points": "Projection",
            "edge": "Edge",
            "model_pick": "Best Bet",
            "sportsbook": "Book",
        })

        st.dataframe(
            display_df.head(10),
            use_container_width=True,
            hide_index=True
        )
        st.caption("Top plays are prebuilt from the latest updater run for faster loading.")
except Exception as e:
    st.info(f"Top plays are temporarily unavailable: {e}")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card"><div class="section-title">Player Projection</div>', unsafe_allow_html=True)

_, player_names = get_player_lookup()

selected_player = st.selectbox(
    "Search for a player",
    options=player_names,
    index=None,
    placeholder="Start typing a player name..."
)

sportsbooks = get_available_sportsbooks()
selected_book = st.selectbox(
    "Sportsbook",
    options=sportsbooks,
    index=0 if sportsbooks else None,
    placeholder="Choose a sportsbook..."
)

live_line = None
player_lines = None

if selected_player and selected_book:
    try:
        player_lines = get_player_points_lines(selected_player, selected_book)
        if player_lines:
            live_line = player_lines.get("points_line")
            st.caption(f"DEBUG: player_lines = {player_lines}")
    except Exception as e:
        st.warning(f"Could not load sportsbook line: {e}")

manual_default = float(live_line) if live_line is not None else 25.5

sportsbook_line = st.number_input(
    "Sportsbook points line",
    min_value=0.0,
    max_value=80.0,
    value=manual_default,
    step=0.5,
    key=f"sportsbook_line_{selected_player}_{selected_book}"
)

if live_line is not None:
    st.caption(f"Loaded {selected_book} line: {float(live_line):.1f}")
else:
    st.caption(f"No live sportsbook line found for {selected_player} at {selected_book}. Using manual input.")
    
if selected_player:
    with st.spinner("Building projection..."):
        result = build_prediction(selected_player, float(sportsbook_line))

    if result.get("error"):
        st.error(result["error"])
    else:
        team_theme = get_team_theme(result.get("team_abbr") or "")
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

        predicted_points = result["predicted_points"]
        season_avg = result.get("season_avg")
        last5_avg = result.get("last5_avg")
        games_used = result.get("games_used")
        edge = result.get("edge")
        over_prob = result.get("over_prob")
        under_prob = result.get("under_prob")

        if edge is not None:
            pick_text, pick_kind = get_pick_label(edge)
        else:
            pick_text, pick_kind = "No Posted Line", "neutral"

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

        probability_text = "N/A"
        if over_prob is not None and under_prob is not None:
            probability_text = f"O {over_prob * 100:.1f}% / U {under_prob * 100:.1f}%"

        if edge is None:
            interpretation_text = ""
        elif abs(edge) < 1.5:
            interpretation_text = (
                f"The model projects {predicted_points:.2f} points against a line of {sportsbook_line:.1f}, "
                f"which is too close to call confidently."
            )
        else:
            over_text = f"{over_prob:.0%}" if over_prob is not None else "N/A"
            under_text = f"{under_prob:.0%}" if under_prob is not None else "N/A"
            interpretation_text = (
                f"The model projects a {over_text} chance of the over hitting compared to "
                f"{under_text} for the under."
            )

        edge_text = f"{edge:+.2f}" if edge is not None else "N/A"

        model_card_html = f"""
<div class="model-card" style="background:{model_bg};border:2px solid {hex_to_rgba(model_border,0.95)};box-shadow:0 0 0 1px rgba(255,255,255,0.04),0 0 22px {model_glow};">
<div class="model-title">{result["actual_name"]}</div>
<div class="model-subtitle">Model Output</div>

<div class="model-main">
<div class="model-stat" style="background:{model_stat_bg};border:1px solid {model_stat_border};">
<div class="model-stat-label" style="color:{model_label_color};">Predicted Points</div>
<div class="model-stat-value">{predicted_points:.2f}</div>
</div>

<div class="model-stat" style="background:{model_stat_bg};border:1px solid {model_stat_border};">
<div class="model-stat-label" style="color:{model_label_color};">Sportsbook Line</div>
<div class="model-stat-value">{sportsbook_line:.1f}</div>
</div>

<div class="model-stat" style="background:{model_stat_bg};border:1px solid {model_stat_border};">
<div class="model-stat-label" style="color:{model_label_color};">Model Edge</div>
<div class="model-stat-value">{edge_text}</div>
</div>

<div class="model-stat" style="background:{model_stat_bg};border:1px solid {model_stat_border};">
<div class="model-stat-label" style="color:{model_label_color};">Probability Split</div>
<div class="model-stat-value">{probability_text}</div>
</div>
</div>

<div class="small-note">{interpretation_text}</div>

<div class="pick-banner" style="background:{pick_bg};border:2px solid {pick_border};color:{pick_text_color};">
{pick_text}
</div>

<div class="small-note">Trained regression model output compared against the current sportsbook line.</div>
</div>
"""
        st.markdown(model_card_html, unsafe_allow_html=True)

        subcol1, subcol2, subcol3 = st.columns(3)

        with subcol1:
            season_text = "N/A" if season_avg is None else f"{season_avg:.2f}"
            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="mini-title">Season Avg</div>
                    <div class="mini-value">{season_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with subcol2:
            last5_text = "N/A" if last5_avg is None else f"{last5_avg:.2f}"
            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="mini-title">Last 5 Avg</div>
                    <div class="mini-value">{last5_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with subcol3:
            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="mini-title">Games Used</div>
                    <div class="mini-value">{games_used}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if over_prob is not None and under_prob is not None:
            prob_col1, prob_col2 = st.columns(2)

            with prob_col1:
                st.markdown(
                    f"""
                    <div class="mini-card">
                        <div class="mini-title">Over Probability</div>
                        <div class="mini-value">{over_prob * 100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with prob_col2:
                st.markdown(
                    f"""
                    <div class="mini-card">
                        <div class="mini-title">Under Probability</div>
                        <div class="mini-value">{under_prob * 100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        live_stats = result.get("live_stats")
        if live_stats:
            st.markdown("#### Live Game Status")
            live_col1, live_col2, live_col3 = st.columns(3)

            with live_col1:
                st.markdown(
                    f"""
                    <div class="mini-card">
                        <div class="mini-title">Current Points</div>
                        <div class="mini-value">{safe_live_display(live_stats.get('points', 'N/A'))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with live_col2:
                st.markdown(
                    f"""
                    <div class="mini-card">
                        <div class="mini-title">Minutes</div>
                        <div class="mini-value">{format_minutes(live_stats.get('minutes'))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with live_col3:
                st.markdown(
                    f"""
                    <div class="mini-card">
                        <div class="mini-title">Game Status</div>
                        <div class="mini-value">{safe_live_display(live_stats.get('game_status', 'Recent Game'))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        team_info = result.get("team_info")
        if team_info:
            st.caption(f"Team Context: {team_info}")

st.markdown("</div>", unsafe_allow_html=True)
