import sys
import os
import pandas as pd
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
from google.oauth2.service_account import Credentials
import gspread


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shared_app import (
    APP_VERSION,
    get_strong_plays_sheet,
    get_strong_plays_summary,
    get_strong_plays_health,
    update_all_pending_sheet_results,
    get_top_plays_today_df,
)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SHEET_KEY = "1uhjV_Si-qcILfNJbKZrD52y4JnT_GvqQ0hzN7POekQM"
ADMIN_LOG_SHEET_NAME = "Admin Logs"
USAGE_LOG_SHEET_NAME = "Usage Log"


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

    div.stButton > button {
        color: #f8fafc !important;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #334155 0%, #1e293b 100%) !important;
        color: #ffffff !important;
    }

    div.stButton > button:disabled {
        background: rgba(148,163,184,0.15) !important;
        color: #64748b !important;
        border: 1px solid rgba(148,163,184,0.25) !important;
        cursor: not-allowed !important;
    }

    div.stButton > button[kind="secondary"] {
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_gsheet_client():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES,
    )
    return gspread.authorize(creds)


def get_or_create_worksheet(sheet_name, rows=1000, cols=20):
    client = get_gsheet_client()
    workbook = client.open_by_key(SHEET_KEY)

    try:
        return workbook.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        return workbook.add_worksheet(title=sheet_name, rows=rows, cols=cols)


def ensure_admin_log_sheet():
    ws = get_or_create_worksheet(ADMIN_LOG_SHEET_NAME)

    existing_values = ws.get_all_values()
    if not existing_values:
        ws.update(
            "A1:E1",
            [[
                "timestamp",
                "action",
                "source",
                "status",
                "details",
            ]]
        )

    return ws


def write_admin_log(action, source, status, details=""):
    try:
        ws = ensure_admin_log_sheet()
        timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S")

        ws.append_row(
            [timestamp, action, source, status, str(details)],
            value_input_option="USER_ENTERED",
        )
    except Exception as e:
        st.warning(f"Could not write admin log: {e}")


def format_last_update(value):
    if value is None:
        return "N/A"
    try:
        return pd.to_datetime(value).strftime("%b %d, %I:%M %p")
    except Exception:
        return str(value)


@st.cache_data(ttl=30)
def get_admin_logs_df():
    ws = ensure_admin_log_sheet()
    values = ws.get_all_values()

    if not values or len(values) < 2:
        return pd.DataFrame(columns=["timestamp", "action", "source", "status", "details"])

    headers = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=headers)


def get_usage_log_sheet():
    client = get_gsheet_client()
    workbook = client.open_by_key(SHEET_KEY)
    return workbook.worksheet(USAGE_LOG_SHEET_NAME)


def get_usage_logs_df():
    ws = get_usage_log_sheet()
    values = ws.get_all_values()

    if not values or len(values) < 2:
        return pd.DataFrame(
            columns=["timestamp", "event_type", "session_id", "player_name", "sportsbook", "details"]
        )

    headers = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=headers)


def build_usage_summary(logs_df):
    if logs_df.empty:
        return {
            "page_views": 0,
            "unique_sessions": 0,
            "searches": 0,
            "top_play_clicks": 0,
            "top_players": pd.DataFrame(),
            "top_books": pd.DataFrame(),
        }

    working_df = logs_df.copy()

    if "event_type" in working_df.columns:
        working_df["event_type"] = working_df["event_type"].astype(str).str.strip()

    if "session_id" in working_df.columns:
        working_df["session_id"] = working_df["session_id"].astype(str).str.strip()

    if "player_name" in working_df.columns:
        working_df["player_name"] = working_df["player_name"].astype(str).str.strip()

    if "sportsbook" in working_df.columns:
        working_df["sportsbook"] = working_df["sportsbook"].astype(str).str.strip().str.lower()

    page_views = len(working_df[working_df["event_type"] == "page_view"])
    searches = len(working_df[working_df["event_type"] == "search"])
    top_play_clicks = len(working_df[working_df["event_type"] == "top_play_click"])

    unique_sessions = 0
    if "session_id" in working_df.columns:
        unique_sessions = working_df["session_id"].replace("", pd.NA).dropna().nunique()

    top_players = pd.DataFrame()
    search_df = working_df[working_df["event_type"] == "search"].copy()
    if not search_df.empty and "player_name" in search_df.columns:
        valid_player_df = search_df[
            (~search_df["player_name"].isin(["", "None", "none", "nan"]))
        ].copy()
        
        top_players = (
            valid_player_df
            .groupby("player_name")
            .size()
            .reset_index(name="search_count")
            .sort_values("search_count", ascending=False)
            .head(10)
        )

    top_books = pd.DataFrame()
    if not search_df.empty and "sportsbook" in search_df.columns:
        top_books = (
            search_df[search_df["sportsbook"] != ""]
            .groupby("sportsbook")
            .size()
            .reset_index(name="search_count")
            .sort_values("search_count", ascending=False)
            .head(10)
        )

    return {
        "page_views": page_views,
        "unique_sessions": unique_sessions,
        "searches": searches,
        "top_play_clicks": top_play_clicks,
        "top_players": top_players,
        "top_books": top_books,
    }


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


overview_tab, operations_tab, logs_tab, usage_tab, review_tab = st.tabs([
    "Overview",
    "Operations",
    "Logs",
    "Usage",
    "Data Review",
])


with overview_tab:
    top_games_win_rate, top_games_total = get_strong_plays_summary()
    health = get_strong_plays_health()
    usage_logs_df = get_usage_logs_df()
    usage_summary = build_usage_summary(usage_logs_df)

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

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:
        st.markdown(
            f"""
            <div class="section-card">
                <div class="mini-label">Page Views</div>
                <div class="mini-value">{usage_summary['page_views']}</div>
                <div class="muted">Total public app loads</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with summary_col2:
        st.markdown(
            f"""
            <div class="section-card">
                <div class="mini-label">Unique Sessions</div>
                <div class="mini-value">{usage_summary['unique_sessions']}</div>
                <div class="muted">Approximate visitors</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with summary_col3:
        st.markdown(
            f"""
            <div class="section-card">
                <div class="mini-label">Searches</div>
                <div class="mini-value">{usage_summary['searches']}</div>
                <div class="muted">Projection builds</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with summary_col4:
        st.markdown(
            f"""
            <div class="section-card">
                <div class="mini-label">Top Play Clicks</div>
                <div class="mini-value">{usage_summary['top_play_clicks']}</div>
                <div class="muted">Featured play engagement</div>
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


with operations_tab:
    st.markdown('<div class="section-card"><div class="section-title">Admin Tools</div>', unsafe_allow_html=True)

    status_placeholder = st.empty()

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        if st.button("Update Final Results", use_container_width=True):
            status_placeholder.info("Checking pending rows and updating final results...")
            try:
                updated_count, checked_count = update_all_pending_sheet_results()

                write_admin_log(
                    action="update_final_results",
                    source="admin_manual",
                    status="success",
                    details=f"Checked {checked_count} pending rows and updated {updated_count} completed games."
                )

                status_placeholder.success(
                    f"Done. Checked {checked_count} pending rows and updated {updated_count} completed games."
                )
                st.cache_data.clear()

            except Exception as e:
                write_admin_log(
                    action="update_final_results",
                    source="admin_manual",
                    status="failed",
                    details=str(e)
                )
                status_placeholder.error(f"Batch update failed: {e}")

    with row1_col2:
        if st.button("Retry Pending Results", use_container_width=True):
            status_placeholder.info("Retrying pending or ungraded rows...")
            try:
                updated_count, checked_count = update_all_pending_sheet_results()

                write_admin_log(
                    action="retry_pending_results",
                    source="admin_manual",
                    status="success",
                    details=f"Retried {checked_count} pending rows and updated {updated_count} completed games."
                )

                status_placeholder.success(
                    f"Retry complete. Checked {checked_count} pending rows and updated {updated_count} completed games."
                )
                st.cache_data.clear()

            except Exception as e:
                write_admin_log(
                    action="retry_pending_results",
                    source="admin_manual",
                    status="failed",
                    details=str(e)
                )
                status_placeholder.error(f"Retry failed: {e}")

    with row2_col1:
        if st.button("Rebuild Top Plays Live", use_container_width=True):
            status_placeholder.info("Top Plays Live rebuild is not wired yet.")
            write_admin_log(
                action="rebuild_top_plays_live",
                source="admin_manual",
                status="pending",
                details="Button clicked, backend function not wired yet."
            )

    with row2_col2:
        if st.button("🔄 Refresh App State", use_container_width=True):
            try:
                st.cache_data.clear()
                st.cache_resource.clear()

                write_admin_log(
                    action="refresh_app_state",
                    source="admin_manual",
                    status="success",
                    details="Cache cleared and app rerun triggered."
                )

                st.success("App state refreshed. Reloading...")
                st.rerun()

            except Exception as e:
                write_admin_log(
                    action="refresh_app_state",
                    source="admin_manual",
                    status="failed",
                    details=str(e)
                )
                st.error(f"Failed to refresh app: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


with logs_tab:
    st.markdown('<div class="section-card"><div class="section-title">Admin Logs</div>', unsafe_allow_html=True)

    logs_df = get_admin_logs_df()

    if logs_df.empty:
        st.info("No admin logs yet.")
    else:
        st.dataframe(logs_df.tail(25).iloc[::-1], use_container_width=True, hide_index=True)

    if st.button("Test Admin Log"):
        write_admin_log(
            action="test_log",
            source="admin_manual",
            status="success",
            details="Test button clicked"
        )
        st.cache_data.clear()
        st.success("Test log written.")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


with usage_tab:
    st.markdown('<div class="section-card"><div class="section-title">Usage Metrics</div>', unsafe_allow_html=True)

    try:
        usage_logs_df = get_usage_logs_df()
        usage_summary = build_usage_summary(usage_logs_df)

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.markdown(
                f"""
                <div class="status-box">
                    <div class="mini-label">Total Page Views</div>
                    <div class="mini-value">{usage_summary['page_views']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with metric_col2:
            st.markdown(
                f"""
                <div class="status-box">
                    <div class="mini-label">Unique Sessions</div>
                    <div class="mini-value">{usage_summary['unique_sessions']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with metric_col3:
            st.markdown(
                f"""
                <div class="status-box">
                    <div class="mini-label">Total Searches</div>
                    <div class="mini-value">{usage_summary['searches']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with metric_col4:
            st.markdown(
                f"""
                <div class="status-box">
                    <div class="mini-label">Top Play Clicks</div>
                    <div class="mini-value">{usage_summary['top_play_clicks']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        subcol1, subcol2 = st.columns(2)

        with subcol1:
            st.markdown("##### Most Searched Players")
            if usage_summary["top_players"].empty:
                st.info("No search data yet.")
            else:
                st.dataframe(
                    usage_summary["top_players"],
                    use_container_width=True,
                    hide_index=True,
                    height=260
                )

        with subcol2:
            st.markdown("##### Most Used Sportsbooks")
            if usage_summary["top_books"].empty:
                st.info("No sportsbook data yet.")
            else:
                st.dataframe(
                    usage_summary["top_books"],
                    use_container_width=True,
                    hide_index=True,
                    height=260
                )

        st.markdown("##### Recent Usage Events")
        if usage_logs_df.empty:
            st.info("No usage log events yet.")
        else:
            recent_usage_df = usage_logs_df.copy()

            preferred_cols = [
                "timestamp",
                "event_type",
                "session_id",
                "player_name",
                "sportsbook",
                "details",
            ]
            existing_cols = [col for col in preferred_cols if col in recent_usage_df.columns]
            if existing_cols:
                recent_usage_df = recent_usage_df[existing_cols]

            st.dataframe(
                recent_usage_df.tail(50).iloc[::-1],
                use_container_width=True,
                hide_index=True,
                height=320
            )

    except Exception as e:
        st.error(f"Could not load usage metrics: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


with review_tab:
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
            status_box = st.empty()

            with status_box.container():
                st.markdown(
                    """
                    <div class="status-box">
                        <div><span class="muted">Top Plays Status:</span> Starting build</div>
                        <div><span class="muted">Step 1:</span> Loading odds feed and candidate props</div>
                        <div><span class="muted">Step 2:</span> Scoring players against current lines</div>
                        <div><span class="muted">Step 3:</span> Ranking strongest edges</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            top_plays_df = get_top_plays_today_df(odds_api_key, debug=True)

            status_box.empty()

            if top_plays_df is None or top_plays_df.empty:
                st.info("No top plays available right now.")
            else:
                st.success(f"Top plays board loaded: {len(top_plays_df)} rows")
                st.dataframe(top_plays_df, use_container_width=True, height=360)

        except Exception as e:
            st.error(f"Could not build top plays board: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
