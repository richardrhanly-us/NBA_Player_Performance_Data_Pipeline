import os
import sys
import textwrap
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh



from src.services.performance_service import compute_performance_summary
from src.services.prediction_repository import (
    compute_board_idempotency_key,
    get_settled_predictions,
)
from src.services.prediction_result import PredictionStatus
from src.services.prediction_service import build_daily_prediction_board, predict_player
from src.shared_app import (
    BOOKMAKER_KEY,
    CURRENT_SEASON,
    EDGE_THRESHOLD,
    get_available_sportsbooks,
    get_live_adjusted_projection,
    get_live_player_stats,
    get_odds_api_key,
    get_player_details,
    get_player_gamelog_df,
    get_player_points_lines,
    get_scoreboard_for_date,
    get_strong_plays_health,
    get_strong_plays_summary,
    load_active_players,
    load_model,
    load_model_stats,
    normalize_name,
)

import src.shared_app as _shared_app

get_team_game_info = getattr(_shared_app, "get_team_game_info", None)


SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SHEET_KEY = "1uhjV_Si-qcILfNJbKZrD52y4JnT_GvqQ0hzN7POekQM"
USAGE_LOG_SHEET_NAME = "Usage Log"

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
    page_title="Playbook Analytics - NBA",
    page_icon="🏀",
    layout="centered",
)

st_autorefresh(interval=300000, key="live_refresh")

if "selected_player_from_top_play" not in st.session_state:
    st.session_state.selected_player_from_top_play = None

if "selected_book_from_top_play" not in st.session_state:
    st.session_state.selected_book_from_top_play = "draftkings"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "page_view_logged" not in st.session_state:
    st.session_state.page_view_logged = False

if "top_play_click_logged" not in st.session_state:
    st.session_state.top_play_click_logged = False

if "last_logged_search_key" not in st.session_state:
    st.session_state.last_logged_search_key = None

# True = offseason/historical presentation.
# False = full original in-season sportsbook/live behavior.
OFFSEASON_MODE = True

st.markdown(
    """
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

    .top-play-link {
        text-decoration: none !important;
        color: inherit !important;
        display: block;
    }

    .top-play-link:hover {
        text-decoration: none !important;
        color: inherit !important;
    }

    .top-play-card {
        border-radius: 16px;
        padding: 16px 18px;
        margin-bottom: 12px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.22);
        cursor: pointer;
        transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease;
        overflow: hidden;
    }

    .top-play-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.24);
        border-color: rgba(255,255,255,0.16);
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

    .top-play-card-inner {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .top-play-headshot {
        width: 68px;
        height: 68px;
        border-radius: 16px;
        object-fit: cover;
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 6px 16px rgba(0,0,0,0.28);
        flex-shrink: 0;
        background: rgba(255,255,255,0.04);
    }

    .top-play-content {
        flex: 1;
        min-width: 0;
    }

    .top-play-team {
        color: #cbd5e1;
        font-size: 0.84rem;
        margin-bottom: 4px;
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

    div.stButton > button[kind="secondary"] {
        width: 100%;
    }

    div[data-testid="stStatusWidget"] {
        display: none !important;
    }

    div[data-testid="stSpinner"] {
        display: none !important;
    }

    .stSpinner {
        display: none !important;
    }

    .line-loading {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.35rem;
        margin-bottom: 0.35rem;
    }

    @media (max-width: 640px) {
        .hero-title {
            font-size: 1.7rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


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

@st.cache_data(ttl=3600, show_spinner=False)
def get_top_play_visuals(player_name):
    try:
        actual_name_to_id, normalized_to_actual = load_active_players()
        normalized = normalize_name(player_name)
        actual_name = normalized_to_actual.get(normalized, player_name)
        player_id = actual_name_to_id.get(actual_name)

        if not player_id:
            return {
                "actual_name": actual_name,
                "headshot_url": "",
                "team_name": "",
                "team_abbr": "",
                "primary": "#38bdf8",
                "secondary": "#60a5fa",
            }

        player_details = get_player_details(player_id)

        team_name = ""
        team_abbr = ""

        if player_details is not None:
            team_name = str(player_details.team_name or "")
            team_abbr = str(player_details.team_abbreviation or "")

        theme = get_team_theme(team_abbr)

        return {
            "actual_name": actual_name,
            "headshot_url": get_player_headshot_url(player_id) or "",
            "team_name": team_name,
            "team_abbr": team_abbr,
            "primary": theme["primary"],
            "secondary": theme["secondary"],
        }

    except Exception:
        return {
            "actual_name": str(player_name),
            "headshot_url": "",
            "team_name": "",
            "team_abbr": "",
            "primary": "#38bdf8",
            "secondary": "#60a5fa",
        }

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


# parse_minutes_to_float / get_live_adjusted_projection moved to
# src/shared_app.py in Step 8 (pure functions, no Streamlit dependency)
# so the new single-player prediction service can reuse them without
# importing this Streamlit app module. Imported below instead of
# defined locally -- behavior unchanged.


def format_game_clock(clock_value):
    if not clock_value:
        return "0:00"

    text = str(clock_value).strip()

    try:
        if text.startswith("PT"):
            text = text.replace("PT", "")

            mins = 0
            secs = 0

            if "M" in text:
                m_part = text.split("M")[0]
                mins = int(float(m_part)) if m_part else 0
                text = text.split("M")[1]

            if "S" in text:
                s_part = text.replace("S", "")
                secs = int(float(s_part)) if s_part else 0

            return f"{mins}:{secs:02d}"

        if ":" in text:
            parts = text.split(":")
            if len(parts) == 2:
                mins = int(float(parts[0]))
                secs = int(float(parts[1]))
                return f"{mins}:{secs:02d}"

        return text
    except Exception:
        return text

def format_commence_time(commence_time):
    if not commence_time:
        return "Time TBD"

    try:
        dt = pd.to_datetime(commence_time, utc=True).tz_convert("America/Chicago")
        return dt.strftime("%b %d, %I:%M %p CT")
    except Exception:
        return str(commence_time)

def format_game_status_short(status, live_stats=None):
    if live_stats:
        period = live_stats.get("period")
        if period is not None:
            try:
                period = int(period)
                if period == 1:
                    return "Q1"
                if period == 2:
                    return "Q2"
                if period == 3:
                    return "Q3"
                if period == 4:
                    return "Q4"
                if period >= 5:
                    return f"OT{period - 4}" if period > 5 else "OT"
            except Exception:
                pass

    if not status:
        return "Live"

    text = str(status).lower()

    if "1st" in text or "q1" in text:
        return "Q1"
    if "2nd" in text or "q2" in text:
        return "Q2"
    if "3rd" in text or "q3" in text:
        return "Q3"
    if "4th" in text or "q4" in text:
        return "Q4"
    if "half" in text or "halftime" in text:
        return "HALF"
    if "final" in text:
        return "FINAL"

    return str(status)


def get_player_headshot_url(player_id):
    if not player_id:
        return None
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"


def render_model_card_offseason(
    result,
    model_bg,
    model_border,
    model_glow,
    model_stat_bg,
    model_stat_border,
    model_label_color,
    projection_label,
    predicted_points,
):
    headshot_url = result.get("headshot_url") or ""
    team_name = result.get("team_name") or ""
    position = result.get("position") or ""
    season_avg = result.get("season_avg")
    last5_avg = result.get("last5_avg")
    games_used = result.get("games_used")

    team_position_line = ""
    if team_name and position:
        team_position_line = f"{team_name} • {position}"
    elif team_name:
        team_position_line = team_name
    elif position:
        team_position_line = position

    season_avg_text = "N/A" if season_avg is None else f"{season_avg:.2f}"
    last5_avg_text = "N/A" if last5_avg is None else f"{last5_avg:.2f}"
    games_used_text = "N/A" if games_used is None else str(games_used)

    model_card_html = textwrap.dedent(f"""\
    <div class="model-card"
    style="
    background:{model_bg};
    border:2px solid {hex_to_rgba(model_border,0.95)};
    box-shadow:0 0 1px rgba(255,255,255,0.04),0 0 22px {model_glow};
    ">

    <div style="display:flex; align-items:center; gap:18px; margin-bottom:14px;">
    <img
    src="{headshot_url}"
    style="
    width:84px;
    height:84px;
    border-radius:18px;
    object-fit:cover;
    border:1px solid rgba(255,255,255,0.12);
    box-shadow:0 6px 18px rgba(0,0,0,0.35);
    flex-shrink:0;
    "
    onerror="this.style.display='none';"
    >

    <div>
    <div class="model-title" style="margin-bottom:2px;">
    {result["actual_name"]}
    </div>

    <div style="font-size:0.82rem; color:#cbd5e1; margin-bottom:6px;">
    {team_position_line}
    </div>

    <div class="model-subtitle" style="margin-bottom:0;">
    Player Performance Forecast • {CURRENT_SEASON}
    </div>
    </div>
    </div>

    <div class="model-main">

    <div class="model-stat"
    style="background:{model_stat_bg};border:1px solid {model_stat_border};">
    <div class="model-stat-label" style="color:{model_label_color};">
    {projection_label}
    </div>
    <div class="model-stat-value">
    {predicted_points:.2f}
    </div>
    </div>

    <div class="model-stat"
    style="background:{model_stat_bg};border:1px solid {model_stat_border};">
    <div class="model-stat-label" style="color:{model_label_color};">
    {CURRENT_SEASON} Average
    </div>
    <div class="model-stat-value">
    {season_avg_text}
    </div>
    </div>

    <div class="model-stat"
    style="background:{model_stat_bg};border:1px solid {model_stat_border};">
    <div class="model-stat-label" style="color:{model_label_color};">
    Last 5 Average
    </div>
    <div class="model-stat-value">
    {last5_avg_text}
    </div>
    </div>

    <div class="model-stat"
    style="background:{model_stat_bg};border:1px solid {model_stat_border};">
    <div class="model-stat-label" style="color:{model_label_color};">
    {CURRENT_SEASON} Games Analyzed
    </div>
    <div class="model-stat-value">
    {games_used_text}
    </div>
    </div>

    </div>

    <div class="small-note">
    Forecast generated from the trained regression model using historical
    player performance and rolling statistical features.
    </div>

    </div>
    """)

    st.markdown(model_card_html, unsafe_allow_html=True)


def render_model_card_live(
    result,
    model_bg,
    model_border,
    model_glow,
    model_stat_bg,
    model_stat_border,
    model_label_color,
    projection_label,
    predicted_points,
    sportsbook_line,
    edge_text,
    probability_text,
    interpretation_text,
    pick_bg,
    pick_border,
    pick_text_color,
    pick_text,
):
    headshot_url = result.get("headshot_url") or ""
    team_name = result.get("team_name") or ""
    position = result.get("position") or ""

    team_position_line = ""
    if team_name and position:
        team_position_line = f"{team_name} • {position}"
    elif team_name:
        team_position_line = team_name
    elif position:
        team_position_line = position

    model_card_html = f"""<div class="model-card"
style="background:{model_bg};
border:2px solid {hex_to_rgba(model_border,0.95)};
box-shadow:0 0 1px rgba(255,255,255,0.04),0 0 22px {model_glow};">

<div style="display:flex; align-items:center; gap:18px; margin-bottom:14px;">
<img
src="{headshot_url}"
style="
width:84px;
height:84px;
border-radius:18px;
object-fit:cover;
border:1px solid rgba(255,255,255,0.12);
box-shadow:0 6px 18px rgba(0,0,0,0.35);
flex-shrink:0;
"
onerror="this.style.display='none';"
>
<div>
<div class="model-title" style="margin-bottom:2px;">{result["actual_name"]}</div>
<div style="font-size:0.82rem; color:#cbd5e1; margin-bottom:6px;">{team_position_line}</div>
<div class="model-subtitle" style="margin-bottom:0;">Model Output</div>
</div>
</div>

<div class="model-main">

<div class="model-stat" style="background:{model_stat_bg};border:1px solid {model_stat_border};">
<div class="model-stat-label" style="color:{model_label_color};">{projection_label}</div>
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

<div class="pick-banner"
style="background:{pick_bg};border:2px solid {pick_border};color:{pick_text_color};">
{pick_text}
</div>

<div class="small-note">
Trained regression model output compared against the current sportsbook line.
</div>

</div>"""

    st.markdown(model_card_html, unsafe_allow_html=True)


@st.cache_resource
def get_gsheet_client():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES,
    )
    return gspread.authorize(creds)


def get_or_create_usage_worksheet(sheet_name, rows=5000, cols=10):
    client = get_gsheet_client()
    workbook = client.open_by_key(SHEET_KEY)

    try:
        return workbook.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        return workbook.add_worksheet(title=sheet_name, rows=rows, cols=cols)


def ensure_usage_log_sheet():
    ws = get_or_create_usage_worksheet(USAGE_LOG_SHEET_NAME)

    existing_values = ws.get_all_values()
    if not existing_values:
        ws.update(
            [[
                "timestamp",
                "event_type",
                "session_id",
                "player_name",
                "sportsbook",
                "details",
            ]],
            range_name="A1:F1",
        )

    return ws


def write_usage_log(event_type, session_id, player_name="", sportsbook="", details=""):
    try:
        ws = ensure_usage_log_sheet()
        timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S")

        ws.append_row(
            [
                timestamp,
                str(event_type),
                str(session_id),
                "" if player_name is None else str(player_name),
                "" if sportsbook is None else str(sportsbook),
                str(details),
            ],
            value_input_option=cast(Any, "USER_ENTERED"),
        )
    except Exception:
        pass


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


@st.cache_data(ttl=300, show_spinner=False)
def get_daily_prediction_board_cached(bookmaker_key):
    api_key = get_odds_api_key()
    if not api_key:
        return None
    return build_daily_prediction_board(api_key=api_key, bookmaker_key=bookmaker_key)


@st.cache_data(ttl=300, show_spinner=False)
def get_board_persistence_status_cached(idempotency_key):
    """
    Step 9: read-only check of whether today's board content has already
    been written to prediction history. Persistence itself is NOT done
    from this public, anonymous-facing UI -- it happens via the
    separate, scheduled scripts/persist_prediction_board.py job (see its
    docstring). This just reports status so the UI can be transparent
    about it without letting persistence mechanics dominate the page.

    Returns None if no history database is configured (DATABASE_URL
    unset -- a normal, expected state in dev/offseason), True if a run
    with this exact content already exists, False otherwise.
    """
    if not os.environ.get("DATABASE_URL"):
        return None
    from src.services.db_connection import get_prediction_db_connection
    from src.services.prediction_repository import get_run_by_idempotency_key

    conn = get_prediction_db_connection()
    try:
        return get_run_by_idempotency_key(conn, idempotency_key) is not None
    finally:
        conn.close()


@st.cache_data(ttl=300, show_spinner=False)
def get_prediction_history_cached(start_date, end_date, direction, qualified_only):
    """
    Step 9: read-only prediction-history summary + settled predictions
    for the public "Prediction History" section below. Returns None if
    no history database is configured (DATABASE_URL unset) rather than
    raising -- the section degrades gracefully, same as the odds-API-key
    gap above. ttl=300 matches get_daily_prediction_board_cached --
    settlement runs on its own schedule, not on every page view.
    """
    if not os.environ.get("DATABASE_URL"):
        return None
    from src.services.db_connection import get_prediction_db_connection

    conn = get_prediction_db_connection()
    try:
        summary = compute_performance_summary(
            conn,
            start_date=start_date,
            end_date=end_date,
            direction=direction,
            qualified_only=qualified_only,
        )
        rows = get_settled_predictions(
            conn,
            start_date=start_date,
            end_date=end_date,
            direction=direction,
            qualified_only=qualified_only,
        )
    finally:
        conn.close()
    return summary, rows


@st.cache_data(ttl=300, show_spinner=False)
def get_board_freshness_cached():
    """
    Step 10: read-only freshness classification for the small status
    line near the Edge Board (FRESH / WAITING_FOR_PROPS / NO_GAMES /
    STALE) -- see src/services/orchestration.py::compute_board_freshness.
    Returns None if no history database is configured, so this degrades
    the same way every other DB-backed section in this file does.
    Never raises -- a lookup failure is treated as "unknown" rather than
    surfaced to public users.
    """
    if not os.environ.get("DATABASE_URL"):
        return None
    try:
        from src.services.automation_config import STALE_AFTER_HOURS
        from src.services.db_connection import get_prediction_db_connection
        from src.services.orchestration import compute_board_freshness
        from src.services.prediction_repository import get_latest_run

        try:
            games_today = len(get_scoreboard_for_date()) > 0
        except Exception:
            games_today = False

        conn = get_prediction_db_connection()
        try:
            latest_run = get_latest_run(conn)
        finally:
            conn.close()

        status, message = compute_board_freshness(
            latest_run=latest_run, games_today=games_today, stale_after_hours=STALE_AFTER_HOURS
        )
        return status.value, message
    except Exception:
        return None


@st.cache_data(ttl=60)
def build_prediction(player_name: str, sportsbook_line: float | None) -> dict[str, Any]:
    # Step 8: the "resolve player -> recent gamelog -> legacy feature row
    # -> model.predict" core is now delegated to
    # src.services.prediction_service.predict_player (the same function
    # the new daily board uses) instead of being duplicated here.
    # apply_live_adjustment=False on purpose: this call only supplies the
    # PRE-live-adjustment number (base_predicted_points); live adjustment
    # is still applied below exactly as before (one get_live_player_stats
    # call, one get_live_adjusted_projection call -- unchanged), so this
    # refactor makes zero difference to the number of external calls this
    # function makes.
    model = load_model()
    model_stats = load_model_stats()

    base_result = predict_player(
        player_name,
        sportsbook_line=sportsbook_line,
        model=model,
        apply_live_adjustment=False,
    )

    if base_result.status == PredictionStatus.UNMATCHED:
        return {"error": "Player ID not found."}
    if base_result.status == PredictionStatus.MISSING_HISTORY:
        return {"error": base_result.reason}
    if base_result.status in (PredictionStatus.ERROR, PredictionStatus.MODEL_UNAVAILABLE, PredictionStatus.PROVIDER_UNAVAILABLE):
        return {"error": base_result.reason or "Prediction unavailable."}

    actual_name = base_result.player_name
    player_id = base_result.player_id
    base_predicted_points = base_result.model_projection
    predicted_points = base_predicted_points

    points_std = None
    if isinstance(model_stats, dict):
        points_std = model_stats.get("std_dev")

    season_avg = None
    last5_avg = None
    games_used = 0

    gamelog_df = get_player_gamelog_df(player_id, CURRENT_SEASON)
    if gamelog_df is not None and not gamelog_df.empty:
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

    live_adjusted_projection = get_live_adjusted_projection(base_predicted_points, live_stats)

    if live_stats:
        predicted_points = live_adjusted_projection

    over_prob = None
    under_prob = None
    if sportsbook_line is not None and points_std:
        try:
            over_prob = 1 - norm.cdf(sportsbook_line, loc=predicted_points, scale=points_std)
            under_prob = norm.cdf(sportsbook_line, loc=predicted_points, scale=points_std)
        except Exception:
            over_prob = None
            under_prob = None

    team_info = None
    if get_team_game_info is not None:
        try:
            team_info = get_team_game_info(actual_name)
        except Exception:
            team_info = None

    player_details = None
    try:
        player_details = get_player_details(player_id)
    except Exception:
        player_details = None

    team_name = None
    team_abbr = None
    position = None
    if player_details is not None:
        team_name = player_details.team_name
        team_abbr = player_details.team_abbreviation
        position = player_details.position

    return {
        "actual_name": actual_name,
        "headshot_url": get_player_headshot_url(player_id),
        "player_id": player_id,
        "predicted_points": predicted_points,
        "base_predicted_points": base_predicted_points,
        "live_adjusted_projection": live_adjusted_projection,
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
        "position": position,
    }


if OFFSEASON_MODE:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Playbook Analytics — NBA</div>
            <p class="hero-subtitle">Player Performance Forecasting</p>
            <p class="hero-subtitle">
                Machine-learning analysis of NBA player scoring performance
                using historical game data and rolling statistical features.
            </p>
            <div class="hero-pills">
                <div class="hero-pill">Player Forecasts</div>
                <div class="hero-pill">Model Evaluation</div>
                <div class="hero-pill">Historical Analysis</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Offseason Demo Mode</div>
            <div class="muted">
                The NBA is currently in the offseason, so live game and sportsbook
                data may be unavailable. Historical player data remains available
                to demonstrate the prediction pipeline and model behavior.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Playbook Analytics - NBA</div>
            <p class="hero-subtitle">NBA Player Prop Model</p>
            <p class="hero-subtitle">Model driven insights into player point total projections</p>
            <div class="hero-pills">
                <div class="hero-pill">Projected points</div>
                <div class="hero-pill">Top edges</div>
                <div class="hero-pill">Live Stats</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not st.session_state.page_view_logged:
    write_usage_log(
        event_type="page_view",
        session_id=st.session_state.session_id,
        details="publicapp_loaded"
    )
    st.session_state.page_view_logged = True

SHOW_LIVE_FEATURES = not OFFSEASON_MODE

if SHOW_LIVE_FEATURES:
    top_games_win_rate, top_games_total = get_strong_plays_summary()
    health = get_strong_plays_health()

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
            unsafe_allow_html=True,
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
            unsafe_allow_html=True,
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
        top_plays_df = get_top_plays_live_df().copy()

        if "game_status" in top_plays_df.columns:
            top_plays_df = top_plays_df[
                ~top_plays_df["game_status"].astype(str).str.upper().str.contains("FINAL", na=False)
            ].copy()

        if "sportsbook_line" in top_plays_df.columns:
            top_plays_df["sportsbook_line"] = pd.to_numeric(
                top_plays_df["sportsbook_line"],
                errors="coerce",
            )

        top_plays_df = top_plays_df[top_plays_df["sportsbook_line"].notna()].copy()
        top_plays_df = top_plays_df[top_plays_df["sportsbook_line"] > 0].copy()
        top_plays_df = top_plays_df[top_plays_df["sportsbook_line"] != 25.5].copy()

        if top_plays_df.empty:
            st.info("No top plays available right now.")
        else:
            st.markdown("###  Top 3 Plays")
            st.markdown("##### Highest confidence plays of the day")

            top3 = top_plays_df.head(3)
            for _, row in top3.iterrows():
                edge_val = pd.to_numeric(row.get("edge"), errors="coerce")
                pred_val = pd.to_numeric(row.get("predicted_points"), errors="coerce")
                line_val = pd.to_numeric(row.get("sportsbook_line"), errors="coerce")
                player_name = row.get("PLAYER_NAME", "Player")
                pick = row.get("model_pick", "")
                matchup = f"{row.get('away_team', '')} @ {row.get('home_team', '')}"
                book_name = str(row.get("sportsbook", "draftkings")).lower()

                line_text = f"{line_val:.1f}" if pd.notna(line_val) else "N/A"
                pred_text = f"{pred_val:.2f}" if pd.notna(pred_val) else "N/A"
                edge_text = f"{edge_val:+.2f}" if pd.notna(edge_val) else "N/A"
                game_time_text = format_commence_time(row.get("commence_time", ""))

                href = f"?player={quote_plus(player_name)}&book={quote_plus(book_name)}"

                visuals = get_top_play_visuals(player_name)
                headshot_url = visuals.get("headshot_url", "")
                team_name = visuals.get("team_name", "")
                primary = visuals.get("primary", "#38bdf8")
                secondary = visuals.get("secondary", "#60a5fa")

                card_bg = (
                    f"linear-gradient(135deg, "
                    f"{hex_to_rgba(primary, 0.22)} 0%, "
                    f"{hex_to_rgba(secondary, 0.16)} 42%, "
                    f"rgba(15,23,42,0.95) 100%)"
                )

                card_border = hex_to_rgba(primary, 0.95)

                team_line = team_name if team_name else "NBA"

                st.markdown(
                    f"""
                    <a class="top-play-link" href="{href}">
                        <div class="top-play-card"
                            style="background:{card_bg}; border:1.5px solid {card_border};">
                            <div class="top-play-card-inner">
                                <img
                                    src="{headshot_url}"
                                    class="top-play-headshot"
                                    onerror="this.style.display='none';"
                                >
                                <div class="top-play-content">
                                    <div class="top-play-title">{player_name} — {pick} {line_text}</div>
                                    <div class="top-play-team">{team_line}</div>
                                    <div class="top-play-sub">{matchup}</div>
                                    <div class="top-play-meta">{game_time_text}</div>
                                    <div class="top-play-meta">Projection: {pred_text} | Edge: {edge_text}</div>
                                </div>
                            </div>
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="section-card"><div class="section-title">Top Plays Today</div>',
                unsafe_allow_html=True,
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
                ]
                if col in top_plays_df.columns
            ]

            display_df = top_plays_df[display_cols].copy()
            display_df = display_df.rename(
                columns={
                    "PLAYER_NAME": "Player",
                    "away_team": "Away",
                    "home_team": "Home",
                    "sportsbook_line": "Line",
                    "predicted_points": "Projection",
                    "edge": "Edge",
                    "model_pick": "Best Bet",
                    "sportsbook": "Book",
                }
            )

            def row_color(row):
                edge = row.get("Edge")

                if pd.isna(edge):
                    return [""] * len(row)

                strength = abs(edge)

                if strength >= 6:
                    color = "rgba(34,197,94,0.8)"
                elif strength >= 3:
                    color = "rgba(34,197,94,0.4)"
                else:
                    color = "rgba(148,163,184,0.15)"

                return [f"background-color: {color};"] * len(row)

            styled_df = (
                display_df.head(10)
                .style
                .apply(row_color, axis=1)
                .format(
                    {
                        "Line": "{:.1f}",
                        "Projection": "{:.2f}",
                        "Edge": "{:+.2f}",
                    }
                )
            )

            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Top plays are prebuilt from the latest updater run for faster loading.")
    except Exception as e:
        st.info(f"Top plays are temporarily unavailable: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Step 8: Today's Edge Board -- a freshly-computed (not prebuilt-from-sheet)
# board across today's full points-prop market, via
# src/services/prediction_service.py::build_daily_prediction_board.
# Pregame projections only (see that module's docstring for why); every
# discovered prop is shown, qualified edges are flagged (not filtered), and
# unavailable/unmatched players are reported rather than silently dropped.
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="section-card"><div class="section-title">Today\'s Edge Board</div>',
    unsafe_allow_html=True,
)
try:
    freshness = get_board_freshness_cached()  # Step 10: (status_str, message) or None

    if not get_odds_api_key():
        st.info("Edge board unavailable: no odds data source is configured right now.")
    else:
        board = get_daily_prediction_board_cached(BOOKMAKER_KEY)

        if board is None or not board.predictions:
            if freshness is not None and freshness[0] == "NO_GAMES":
                st.info("No NBA games are scheduled today.")
            elif freshness is not None and freshness[0] == "WAITING_FOR_PROPS":
                st.info("Games are scheduled today, but no player-points props have been posted yet.")
            else:
                st.info("No player-points props are available right now.")
        else:
            st.caption(
                f"{board.props_discovered} props discovered · "
                f"{board.predictions_generated} predictions generated · "
                f"{board.unmatched_count} unmatched · "
                f"{board.unavailable_count} unavailable · "
                f"pregame projections only · model {board.model_version} · "
                f"{board.bookmaker} · generated {board.generated_at_utc}"
            )

            try:
                persisted = get_board_persistence_status_cached(
                    compute_board_idempotency_key(board)
                )
            except Exception:
                persisted = None
            if persisted is not None:
                st.caption(
                    "History: "
                    + (
                        "this board is saved to prediction history."
                        if persisted
                        else "this board has not yet been recorded to prediction "
                        "history (recorded by the scheduled history job, not this page)."
                    )
                )
            if freshness is not None and freshness[0] == "STALE":
                st.caption(f"Note: {freshness[1]}")

            ok_predictions = [r for r in board.predictions if r.status == PredictionStatus.OK]
            other_predictions = [r for r in board.predictions if r.status != PredictionStatus.OK]

            if not ok_predictions:
                st.info("No predictions could be generated for today's props.")
            else:
                board_rows = [
                    {
                        "Player": r.player_name,
                        "Matchup": r.matchup or "—",
                        "Model Projection": r.model_projection,
                        "Line": r.sportsbook_line,
                        "Edge": r.edge,
                        "Lean": r.direction.value if r.direction is not None else "—",
                        "Sportsbook": r.bookmaker or "—",
                        "Qualified": (
                            "Yes"
                            if r.edge is not None and abs(r.edge) >= EDGE_THRESHOLD
                            else ""
                        ),
                    }
                    for r in ok_predictions
                ]
                board_df = pd.DataFrame(board_rows)

                def _edge_row_color(row):
                    if row.get("Qualified") == "Yes":
                        return ["background-color: rgba(34,197,94,0.35);"] * len(row)
                    return [""] * len(row)

                styled_board_df = (
                    board_df.style.apply(_edge_row_color, axis=1)
                    .format(
                        {
                            "Model Projection": "{:.2f}",
                            "Line": "{:.1f}",
                            "Edge": "{:+.2f}",
                        }
                    )
                )
                st.dataframe(styled_board_df, use_container_width=True, hide_index=True)
                st.caption(
                    f"Highlighted rows meet the current qualified-edge threshold "
                    f"(±{EDGE_THRESHOLD:.1f}). This is an analytical projection, "
                    "not a guarantee."
                )

            if other_predictions:
                with st.expander(f"{len(other_predictions)} player(s) unavailable today"):
                    for r in other_predictions:
                        st.caption(f"{r.player_name}: {r.reason or r.status.value}")
except Exception as e:
    st.info(f"Edge board is temporarily unavailable: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Step 9: Prediction History -- a record of past predictions and how they
# actually settled, built from prediction_runs/prediction_snapshots/
# prediction_outcomes (see src/services/{prediction_repository,
# performance_service}.py). Every value shown here is a snapshot captured
# at prediction time; sportsbook lines can move after a prediction is
# made, and predictions are not guarantees of outcome. Pending
# (unsettled) predictions are excluded entirely, and pushes are excluded
# from the win-rate denominator.
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="section-card"><div class="section-title">Prediction History</div>',
    unsafe_allow_html=True,
)
try:
    if not os.environ.get("DATABASE_URL"):
        st.info("Prediction history is unavailable: no history database is configured right now.")
    else:
        hist_col1, hist_col2, hist_col3, hist_col4 = st.columns(4)
        with hist_col1:
            date_range_choice = st.selectbox(
                "Date range",
                ["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
                index=1,
                key="history_date_range",
            )
        with hist_col2:
            direction_choice = st.selectbox(
                "Lean", ["All", "OVER", "UNDER"], key="history_direction"
            )
        with hist_col3:
            qualified_choice = st.selectbox(
                "Edge",
                ["All predictions", f"Qualified only (±{EDGE_THRESHOLD:.1f}+)"],
                key="history_qualified",
            )
        with hist_col4:
            result_choice = st.selectbox(
                "Result", ["All", "WIN", "LOSS", "PUSH"], key="history_result"
            )

        _range_days = {
            "Last 7 days": 7,
            "Last 30 days": 30,
            "Last 90 days": 90,
            "All time": None,
        }[date_range_choice]
        history_start_date = (
            (datetime.now(timezone.utc) - timedelta(days=_range_days)).isoformat()
            if _range_days is not None
            else None
        )
        history_direction = None if direction_choice == "All" else direction_choice
        history_qualified_only = qualified_choice.startswith("Qualified")

        history = get_prediction_history_cached(
            history_start_date, None, history_direction, history_qualified_only
        )
        if history is None:
            st.info("Prediction history is unavailable: no history database is configured right now.")
        else:
            summary, settled_rows = history
            if result_choice != "All":
                settled_rows = [
                    r for r in settled_rows if r.get("outcome_result_status") == result_choice
                ]

            if summary.graded == 0:
                st.info("No settled predictions yet for this filter.")
            else:
                record = f"{summary.wins}-{summary.losses}-{summary.pushes}"
                win_rate_pct = (
                    f"{summary.win_rate * 100:.1f}%" if summary.win_rate is not None else "—"
                )
                qualified_record = (
                    f"{summary.qualified_wins}-{summary.qualified_losses}-"
                    f"{summary.qualified_pushes}"
                )
                qualified_win_rate_pct = (
                    f"{summary.qualified_win_rate * 100:.1f}%"
                    if summary.qualified_win_rate is not None
                    else "—"
                )

                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                metric_col1.metric("Predictions Graded", summary.graded)
                metric_col2.metric("Record (W-L-P)", record)
                metric_col3.metric("Win Rate", win_rate_pct)
                metric_col4.metric(
                    "Qualified Win Rate",
                    qualified_win_rate_pct,
                    help=f"Qualified record: {qualified_record}",
                )

                st.caption(
                    "Win rate excludes pushes from the denominator; unsettled "
                    f"(pending) predictions are excluded entirely. \"Qualified\" means "
                    f"|edge| ≥ {EDGE_THRESHOLD:.1f}, the site's existing edge "
                    "threshold -- these results do not change that threshold."
                )

                history_table_rows = [
                    {
                        "Date": (r.get("game_date") or (r.get("generated_at_utc") or "")[:10]),
                        "Player": r.get("player_name"),
                        "Projection": (
                            f"{r['model_projection']:.2f}"
                            if r.get("model_projection") is not None
                            else "—"
                        ),
                        "Line": (
                            f"{r['sportsbook_line']:.1f}"
                            if r.get("sportsbook_line") is not None
                            else "—"
                        ),
                        "Edge": f"{r['edge']:+.2f}" if r.get("edge") is not None else "—",
                        "Lean": r.get("direction") or "—",
                        "Actual": (
                            f"{r['outcome_actual_points']:.1f}"
                            if r.get("outcome_actual_points") is not None
                            else "—"
                        ),
                        "Result": r.get("outcome_result_status"),
                    }
                    for r in settled_rows[:100]
                ]
                if not history_table_rows:
                    st.info("No settled predictions match this result filter.")
                else:
                    st.dataframe(
                        pd.DataFrame(history_table_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                    if len(settled_rows) > 100:
                        st.caption(f"Showing the most recent 100 of {len(settled_rows)} settled predictions.")

                st.caption(
                    "Values shown are snapshots captured at prediction time -- "
                    "sportsbook lines can move after a prediction is made, and "
                    "predictions are not guarantees of outcome."
                )
except Exception as e:
    st.info(f"Prediction history is temporarily unavailable: {e}")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-card"><div class="section-title">Player Projection</div>',
    unsafe_allow_html=True,
)

query_params = st.query_params
query_player = query_params.get("player")
query_book = query_params.get("book")

if query_player and not st.session_state.top_play_click_logged:
    write_usage_log(
        event_type="top_play_click",
        session_id=st.session_state.session_id,
        player_name=query_player,
        sportsbook=query_book or "",
        details="from_top_3_or_top_plays"
    )
    st.session_state.top_play_click_logged = True

_, player_names = get_player_lookup()

default_player = query_player or st.session_state.get("selected_player_from_top_play")
player_index = None
if default_player in player_names:
    player_index = player_names.index(default_player)

selected_player_raw = st.selectbox(
    "Search for a player",
    options=player_names,
    index=player_index,
    placeholder="Start typing a player name...",
    key="player_projection_selectbox",
)
selected_player: str | None = str(selected_player_raw) if selected_player_raw is not None else None

# The player selector is shared by both modes.
st.session_state.selected_player_from_top_play = selected_player

if OFFSEASON_MODE:
    sportsbook_line = None
    selected_book = None
    line_is_live = False

    if selected_player:
        with st.spinner("Building player forecast..."):
            result = build_prediction(selected_player, None)

        search_key = f"{selected_player}|forecast"

        if st.session_state.last_logged_search_key != search_key:
            write_usage_log(
                event_type="search",
                session_id=st.session_state.session_id,
                player_name=selected_player,
                sportsbook="",
                details="player_forecast"
            )
            st.session_state.last_logged_search_key = search_key

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
            base_predicted_points = result.get("base_predicted_points")
            live_stats = result.get("live_stats")
            projection_label = "Live Adjusted Projection" if live_stats else "Predicted Points"

            render_model_card_offseason(
                result=result,
                model_bg=model_bg,
                model_border=model_border,
                model_glow=model_glow,
                model_stat_bg=model_stat_bg,
                model_stat_border=model_stat_border,
                model_label_color=model_label_color,
                projection_label=projection_label,
                predicted_points=predicted_points,
            )

            if live_stats and base_predicted_points is not None:
                st.caption(
                    f"Pregame model: {base_predicted_points:.2f} | "
                    f"Live-adjusted projection: {predicted_points:.2f}"
                )

else:
    # ORIGINAL IN-SEASON SPORTSBOOK / LIVE-LINE WORKFLOW
    sportsbooks = get_available_sportsbooks()

    default_book = (
        query_book or
        st.session_state.get("selected_book_from_top_play") or
        "draftkings"
    ).lower()

    book_index = 0
    if default_book in sportsbooks:
        book_index = sportsbooks.index(default_book)

    selected_book_raw = st.selectbox(
        "Sportsbook",
        options=sportsbooks,
        index=book_index if sportsbooks else None,
        placeholder="Choose a sportsbook...",
        key="sportsbook_selectbox",
    )
    selected_book: str | None = (
        str(selected_book_raw) if selected_book_raw is not None else None
    )

    st.session_state.selected_book_from_top_play = selected_book

    live_line: float | None = None
    player_lines: dict[str, Any] | None = None

    if selected_player and selected_book:
        loading_placeholder = st.empty()
        loading_placeholder.markdown(
            '<div class="line-loading">Loading live sportsbook line...</div>',
            unsafe_allow_html=True,
        )

        try:
            player_lines = get_player_points_lines(selected_player, selected_book)
            if player_lines:
                raw_live_line = player_lines.get("points_line")
                if raw_live_line is not None:
                    try:
                        live_line = float(raw_live_line)
                    except (TypeError, ValueError):
                        live_line = None
        except Exception as e:
            st.warning(f"Could not load sportsbook line: {e}")
        finally:
            loading_placeholder.empty()

    manual_default = float(live_line) if live_line is not None else 25.5

    line_is_live = live_line is not None
    line_status = "live" if line_is_live else "manual_input"

    sportsbook_line = float(
        st.number_input(
            "Sportsbook points line",
            min_value=0.0,
            max_value=80.0,
            value=manual_default,
            step=0.5,
            key=f"sportsbook_line_{selected_player}_{selected_book}",
        )
    )

    manual_override = (
        line_is_live and live_line is not None and sportsbook_line != live_line
    )

    if line_status == "live" and not manual_override:
        st.caption(f"Live line • {selected_book}: {live_line:.1f}")
    elif manual_override:
        st.caption(f"Manual override • using {sportsbook_line:.1f}")
    else:
        st.caption(f"Manual input • using {sportsbook_line:.1f}")

    if selected_player:
        with st.spinner("Building projection..."):
            result = build_prediction(selected_player, sportsbook_line)

        search_key = f"{selected_player}|{selected_book}|{sportsbook_line}"

        if st.session_state.last_logged_search_key != search_key:
            write_usage_log(
                event_type="search",
                session_id=st.session_state.session_id,
                player_name=selected_player,
                sportsbook=selected_book or "",
                details=f"line={sportsbook_line}"
            )
            st.session_state.last_logged_search_key = search_key

        game_is_final = False
        result_live_stats = result.get("live_stats")
        if isinstance(result_live_stats, dict):
            game_status_text = str(result_live_stats.get("game_status", "")).upper()
            game_is_final = "FINAL" in game_status_text

        if game_is_final and not line_is_live:
            st.warning("This game is final and no live sportsbook line is available. Projection is hidden.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

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
            base_predicted_points = result.get("base_predicted_points")
            season_avg = result.get("season_avg")
            last5_avg = result.get("last5_avg")
            games_used = result.get("games_used")
            edge = result.get("edge")
            over_prob = result.get("over_prob")
            under_prob = result.get("under_prob")
            live_stats_raw = result.get("live_stats")
            live_stats = live_stats_raw if isinstance(live_stats_raw, dict) else None

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
            projection_label = "Live Adjusted Projection" if live_stats else "Predicted Points"

            render_model_card_live(
                result=result,
                model_bg=model_bg,
                model_border=model_border,
                model_glow=model_glow,
                model_stat_bg=model_stat_bg,
                model_stat_border=model_stat_border,
                model_label_color=model_label_color,
                projection_label=projection_label,
                predicted_points=predicted_points,
                sportsbook_line=sportsbook_line,
                edge_text=edge_text,
                probability_text=probability_text,
                interpretation_text=interpretation_text,
                pick_bg=pick_bg,
                pick_border=pick_border,
                pick_text_color=pick_text_color,
                pick_text=pick_text,
            )

            if live_stats and base_predicted_points is not None:
                st.caption(
                    f"Pregame model: {base_predicted_points:.2f} | "
                    f"Live-adjusted projection: {predicted_points:.2f}"
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
                        unsafe_allow_html=True,
                    )

                with prob_col2:
                    st.markdown(
                        f"""
                        <div class="mini-card">
                            <div class="mini-title">Under Probability</div>
                            <div class="mini-value">{under_prob * 100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

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
                        unsafe_allow_html=True,
                    )

                with live_col2:
                    st.markdown(
                        f"""
                        <div class="mini-card">
                            <div class="mini-title">Minutes</div>
                            <div class="mini-value">{format_minutes(live_stats.get('minutes'))}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with live_col3:
                    game_status = format_game_status_short(live_stats.get("game_status"), live_stats)
                    game_clock = format_game_clock(live_stats.get("game_clock"))

                    st.markdown(
                        f"""
                        <div class="mini-card">
                            <div class="mini-title">Game</div>
                            <div class="mini-value">{game_status} • {game_clock}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

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
                    unsafe_allow_html=True,
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
                    unsafe_allow_html=True,
                )

            with subcol3:
                st.markdown(
                    f"""
                    <div class="mini-card">
                        <div class="mini-title">Sample Size</div>
                        <div class="mini-value">{games_used}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
