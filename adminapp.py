import os
import pandas as pd
import streamlit as st

from shared_app import (
    APP_VERSION,
    get_gsheet,
    get_strong_plays_sheet,
    get_strong_plays_summary,
    get_strong_plays_health,
    update_all_pending_sheet_results,
    get_top_plays_today_df,
)


st.set_page_config(
    page_title="NBA Admin Dashboard",
    page_icon="🛠️",
    layout="wide"
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
        max-width: 1200px;
    }

    hr, div[data-testid="stDivider"] {
        display: none !important;
    }

    .hero {
        background: linear-gradient(135deg, #111827 0%, #1e293b 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    }

    .hero-title {
        font-size: 1.7rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.15rem;
    }

    .hero-sub {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
    }

    .section-card {
        background: rgba(17,24,39,0.78);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1rem 1rem 0.85rem 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.20);
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.08rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
    }

    .mini-label {
        color: #94a3b8;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }

    .mini-value {
        color: #f8fafc;
        font-size: 1.15rem;
        font-weight: 800;
    }

    .status-box {
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }

    .muted {
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


def format_last_update(value):
    if value is None:
        return "N/A"
    try:
        return pd.to_datetime(value).strftime("%b %d, %I:%M %p")
    except Exception:
        return str(value)


def load_strong_plays_df():
    sheet = get_strong_plays_sheet()
    values = sheet.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame()

    headers = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=headers)


st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">NBA Admin Dashboard</div>
        <div class="hero-sub">Internal tools, monitoring, and maintenance</div>
        <div class="hero-sub">Version: {APP_VERSION}</div>
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("Admin Login", expanded=True):
    admin_mode = False
    admin_key_input = st.text_input("Enter admin key", type="password", key="admin_key_input")

    if admin_key_input == st.secrets["admin_key"]:
        admin_mode = True
        st.success("Admin mode enabled")
    elif admin_key_input:
        st.error("Invalid admin key")

if not admin_mode:
    st.stop()


top_games_win_rate, top_games_total = get_strong_plays_summary()
health = get_strong_plays_health()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="mini-label">Win Rate for Top Games</div>
            <div class="mini-value">
                {"N/A" if top_games_win_rate is None else f"{top_games_win_rate:.1f}%"}
            </div>
            <div class="muted">
                {"No graded games yet" if top_games_win_rate is None else f"{top_games_total} graded games"}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    total_plays = health["total"] if health else 0
    st.markdown(
        f"""
        <div class="section-card">
            <div class="mini-label">Total Strong Plays</div>
            <div class="mini-value">{total_plays}</div>
            <div class="muted">Current rows in Strong Plays</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    graded_plays = health["graded"] if health else 0
    st.markdown(
        f"""
        <div class="section-card">
            <div class="mini-label">Graded Plays</div>
            <div class="mini-value">{graded_plays}</div>
            <div class="muted">WIN or LOSS only</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    pending_plays = health["pending"] if health else 0
    st.markdown(
        f"""
        <div class="section-card">
            <div class="mini-label">Pending Plays</div>
            <div class="mini-value">{pending_plays}</div>
            <div class="muted">Still waiting on result</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown('<div class="section-card"><div class="section-title">Health Check</div>', unsafe_allow_html=True)

if health:
    st.markdown(
        f"""
        <div class="status-box">
            <div><span class="muted">Last Update:</span> {format_last_update(health.get("last_update"))}</div>
            <div><span class="muted">Total Plays:</span> {health.get("total", 0)}</div>
            <div><span class="muted">Graded:</span> {health.get("graded", 0)} &nbsp; | &nbsp; <span class="muted">Pending:</span> {health.get("pending", 0)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Could not load Strong Plays health data.")

st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="section-card"><div class="section-title">Admin Tools</div>', unsafe_allow_html=True)

status_placeholder = st.empty()

tool_col1, tool_col2 = st.columns([1, 1])

with tool_col1:
    if st.button("Update Final Results", use_container_width=True):
        status_placeholder.info("Checking pending rows and updating final results...")
        try:
            updated_count, checked_count = update_all_pending_sheet_results()
            status_placeholder.success(
                f"Done. Checked {checked_count} pending rows and updated {updated_count} completed games."
            )
            st.cache_data.clear()
        except Exception as e:
            status_placeholder.error(f"Batch update failed: {e}")

with tool_col2:
    if st.button("Refresh Dashboard Data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared. Reloading admin dashboard...")
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="section-card"><div class="section-title">Strong Plays Table</div>', unsafe_allow_html=True)

try:
    strong_df = load_strong_plays_df()

    if strong_df.empty:
        st.info("Strong Plays sheet is empty.")
    else:
        display_df = strong_df.copy()

        preferred_cols = [
            "PLAYER_NAME",
            "GAME_DATE",
            "sportsbook",
            "sportsbook_line",
            "predicted_points",
            "final_points",
            "model_pick",
            "bet_status",
            "result_logged_at",
        ]

        existing_cols = [col for col in preferred_cols if col in display_df.columns]
        if existing_cols:
            display_df = display_df[existing_cols]

        st.dataframe(display_df, use_container_width=True, height=420)

except Exception as e:
    st.error(f"Could not load Strong Plays table: {e}")

st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="section-card"><div class="section-title">Pending Plays Review</div>', unsafe_allow_html=True)

try:
    strong_df = load_strong_plays_df()

    if strong_df.empty or "bet_status" not in strong_df.columns:
        st.info("No pending review data available.")
    else:
        pending_df = strong_df.copy()
        pending_df["bet_status"] = pending_df["bet_status"].astype(str).str.strip().str.upper()
        pending_df = pending_df[pending_df["bet_status"] == "PENDING"].copy()

        if pending_df.empty:
            st.success("No pending plays right now.")
        else:
            preferred_cols = [
                "PLAYER_NAME",
                "GAME_DATE",
                "sportsbook",
                "sportsbook_line",
                "predicted_points",
                "model_pick",
                "bet_status",
            ]
            existing_cols = [col for col in preferred_cols if col in pending_df.columns]
            if existing_cols:
                pending_df = pending_df[existing_cols]

            st.dataframe(pending_df, use_container_width=True, height=260)

except Exception as e:
    st.error(f"Could not load pending plays review: {e}")

st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="section-card"><div class="section-title">Top Plays Today Preview</div>', unsafe_allow_html=True)

odds_api_key = os.getenv("ODDS_API_KEY")

if not odds_api_key:
    st.warning("ODDS_API_KEY not found in environment.")
else:
    try:
        with st.spinner("Building top plays board..."):
            top_plays_df = get_top_plays_today_df(odds_api_key)

        if top_plays_df is None or top_plays_df.empty:
            st.info("No top plays available right now.")
        else:
            st.dataframe(top_plays_df, use_container_width=True, height=360)

    except Exception as e:
        st.error(f"Could not build top plays board: {e}")

st.markdown("</div>", unsafe_allow_html=True)