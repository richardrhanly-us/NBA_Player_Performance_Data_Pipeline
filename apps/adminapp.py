import sys
import os
import time
import pandas as pd
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
from google.oauth2.service_account import Credentials
import gspread
import unicodedata

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.shared_app as shared_app

from src.shared_app import (
    APP_VERSION,
    get_strong_plays_sheet,
    get_strong_plays_summary,
    get_strong_plays_health,
    update_all_pending_sheet_results,
    get_top_plays_today_df,
    get_available_sportsbooks,
    load_active_players,
)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SHEET_KEY = "1uhjV_Si-qcILfNJbKZrD52y4JnT_GvqQ0hzN7POekQM"
ADMIN_LOG_SHEET_NAME = "Admin Logs"
USAGE_LOG_SHEET_NAME = "Usage Log"
HISTORICAL_LINES_SHEET_NAME = "Historical Lines"


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


@st.cache_data(ttl=5)
def get_admin_logs_df():
    try:
        ws = ensure_admin_log_sheet()
        values = ws.get_all_values()

        expected_cols = ["timestamp", "action", "source", "status", "details"]

        if not values:
            return pd.DataFrame(columns=expected_cols)

        headers = [str(h).strip() for h in values[0]]

        if len(values) == 1:
            return pd.DataFrame(columns=headers if headers else expected_cols)

        rows = values[1:]

        cleaned_rows = []
        for row in rows:
            row = list(row)
            if len(row) < len(headers):
                row = row + [""] * (len(headers) - len(row))
            elif len(row) > len(headers):
                row = row[:len(headers)]
            cleaned_rows.append(row)

        df = pd.DataFrame(cleaned_rows, columns=headers)

        for col in expected_cols:
            if col not in df.columns:
                df[col] = ""

        df = df[expected_cols].copy()

        df = df[
            ~(df["timestamp"].astype(str).str.strip() == "")
        ].copy()

        return df

    except Exception as e:
        print(f"[ERROR] get_admin_logs_df failed: {e}")
        return pd.DataFrame(columns=["timestamp", "action", "source", "status", "details"])


def get_usage_log_sheet():
    client = get_gsheet_client()
    workbook = client.open_by_key(SHEET_KEY)
    return workbook.worksheet(USAGE_LOG_SHEET_NAME)


def get_usage_logs_df():
    expected_cols = [
        "timestamp",
        "event_type",
        "session_id",
        "player_name",
        "sportsbook",
        "details",
    ]

    try:
        ws = get_usage_log_sheet()
        values = ws.get_all_values()

        if not values or len(values) < 2:
            return pd.DataFrame(columns=expected_cols)

        headers = [str(h).strip() for h in values[0]]
        rows = values[1:]
        df = pd.DataFrame(rows, columns=headers)

        for col in expected_cols:
            if col not in df.columns:
                df[col] = ""

        return df[expected_cols]

    except Exception as e:
        print(f"[ERROR] get_usage_logs_df failed: {e}")
        return pd.DataFrame(columns=expected_cols)

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    return name.strip().lower()

def get_historical_lines_sheet():
    client = get_gsheet_client()
    workbook = client.open_by_key(SHEET_KEY)
    return workbook.worksheet(HISTORICAL_LINES_SHEET_NAME)

def build_usage_summary(logs_df):
    expected_cols = [
        "timestamp",
        "event_type",
        "session_id",
        "player_name",
        "sportsbook",
        "details",
    ]

    if logs_df is None or logs_df.empty:
        return {
            "page_views": 0,
            "unique_sessions": 0,
            "searches": 0,
            "top_play_clicks": 0,
            "top_players": pd.DataFrame(),
            "top_books": pd.DataFrame(),
        }

    working_df = logs_df.copy()

    for col in expected_cols:
        if col not in working_df.columns:
            working_df[col] = ""

    working_df["event_type"] = working_df["event_type"].astype(str).str.strip()
    working_df["session_id"] = working_df["session_id"].astype(str).str.strip()
    working_df["player_name"] = working_df["player_name"].astype(str).str.strip()
    working_df["sportsbook"] = working_df["sportsbook"].astype(str).str.strip().str.lower()

    page_views = len(working_df[working_df["event_type"] == "page_view"])
    searches = len(working_df[working_df["event_type"] == "search"])
    top_play_clicks = len(working_df[working_df["event_type"] == "top_play_click"])

    unique_sessions = (
        working_df["session_id"]
        .replace(["", "None", "none", "nan"], pd.NA)
        .dropna()
        .nunique()
    )

    valid_player_df = working_df[
        (working_df["event_type"] == "search") &
        (~working_df["player_name"].isin(["", "None", "none", "nan"]))
    ].copy()

    top_players = pd.DataFrame()
    if not valid_player_df.empty:
        top_players = (
            valid_player_df.groupby("player_name")
            .size()
            .reset_index(name="search_count")
            .sort_values("search_count", ascending=False)
            .head(10)
        )

    valid_book_df = working_df[
        (working_df["event_type"] == "search") &
        (~working_df["sportsbook"].isin(["", "None", "none", "nan"]))
    ].copy()

    top_books = pd.DataFrame()
    if not valid_book_df.empty:
        top_books = (
            valid_book_df.groupby("sportsbook")
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


def get_sheet1_df():
    try:
        client = get_gsheet_client()
        ws = client.open_by_key(SHEET_KEY).worksheet("Sheet1")
        values = ws.get_all_values()

        if not values or len(values) < 2:
            return pd.DataFrame()

        headers = [str(h).strip() for h in values[0]]
        rows = values[1:]
        return pd.DataFrame(rows, columns=headers)

    except Exception as e:
        print(f"[ERROR] get_sheet1_df failed: {e}")
        return pd.DataFrame()


def build_sheet1_debug_summary(df):
    if df.empty:
        return {
            "total_rows": 0,
            "pending_rows": 0,
            "completed_rows": 0,
            "pending_df": pd.DataFrame(),
        }

    working_df = df.copy()
    working_df.columns = [str(c).strip() for c in working_df.columns]

    for col in ["final_points", "line_result", "model_result", "bet_status", "PLAYER_NAME", "GAME_DATE"]:
        if col not in working_df.columns:
            working_df[col] = ""

    pending_mask = (
        working_df["bet_status"].astype(str).str.strip().str.upper().eq("PENDING") |
        working_df["final_points"].astype(str).str.strip().eq("") |
        working_df["line_result"].astype(str).str.strip().eq("") |
        working_df["model_result"].astype(str).str.strip().eq("")
    )

    pending_df = working_df[pending_mask].copy()
    completed_df = working_df[~pending_mask].copy()

    debug_cols = [
        col for col in [
            "PLAYER_NAME",
            "GAME_DATE",
            "sportsbook_line",
            "sportsbook",
            "predicted_points",
            "final_points",
            "model_pick",
            "line_result",
            "model_result",
            "bet_status",
            "result_logged_at",
        ]
        if col in working_df.columns
    ]

    if debug_cols:
        pending_df = pending_df[debug_cols]

    return {
        "total_rows": len(working_df),
        "pending_rows": len(pending_df),
        "completed_rows": len(completed_df),
        "pending_df": pending_df,
    }


def load_strong_plays_df():
    try:
        sheet = get_strong_plays_sheet()
        values = sheet.get_all_values()

        if not values or len(values) < 2:
            return pd.DataFrame()

        headers = [str(h).strip() for h in values[0]]
        rows = values[1:]
        return pd.DataFrame(rows, columns=headers)

    except Exception as e:
        print(f"[ERROR] load_strong_plays_df failed: {e}")
        return pd.DataFrame()


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
    try:
        top_games_win_rate, top_games_total = get_strong_plays_summary()
    except Exception as e:
        top_games_win_rate, top_games_total = None, 0
        st.warning(f"Could not load Strong Plays summary: {e}")

    try:
        health = get_strong_plays_health()
    except Exception as e:
        health = None
        st.warning(f"Could not load Strong Plays health data: {e}")

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

    if "manual_add_queue" not in st.session_state:
        st.session_state.manual_add_queue = []

    if "last_operations_debug" not in st.session_state:
        st.session_state.last_operations_debug = None

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        if st.button("📊 Update Final Results", use_container_width=True):
            status_placeholder.info("Scanning pending rows and writing results back to Sheet1...")
            try:
                debug_result = update_all_pending_sheet_results(debug=True)
                st.session_state.last_operations_debug = debug_result
                st.json(debug_result)

                write_admin_log(
                    action="update_final_results",
                    source="admin_manual",
                    status="success",
                    details=(
                        f"Source {debug_result.get('source_sheet', 'unknown')} | "
                        f"Loaded {debug_result.get('total_data_rows_loaded', 0)} rows | "
                        f"Scanned {debug_result.get('rows_scanned', 0)} rows | "
                        f"Pending {debug_result.get('pending_rows_found', 0)} | "
                        f"Skipped not final {debug_result.get('rows_skipped_not_final', 0)} | "
                        f"Skipped missing player/date {debug_result.get('rows_skipped_missing_player_date', 0)} | "
                        f"Updated {debug_result.get('rows_updated', 0)}"
                    )
                )

                status_placeholder.success(
                    f"Done. Source: {debug_result.get('source_sheet', 'unknown')} | "
                    f"Loaded: {debug_result.get('total_data_rows_loaded', 0)} | "
                    f"Pending: {debug_result.get('pending_rows_found', 0)} | "
                    f"Updated: {debug_result.get('rows_updated', 0)}"
                )
                st.cache_data.clear()

            except Exception as e:
                write_admin_log(
                    action="update_final_results",
                    source="admin_manual",
                    status="failed",
                    details=str(e)
                )
                status_placeholder.error(f"Update failed: {e}")

    with row1_col2:
        if st.button("🛠️ Retry Pending Results", use_container_width=True):
            status_placeholder.info("Retrying pending rows from Sheet1...")
            try:
                debug_result = update_all_pending_sheet_results(debug=True)
                st.session_state.last_operations_debug = debug_result

                write_admin_log(
                    action="retry_pending_results",
                    source="admin_manual",
                    status="success",
                    details=(
                        f"Source {debug_result.get('source_sheet', 'unknown')} | "
                        f"Loaded {debug_result.get('total_data_rows_loaded', 0)} rows | "
                        f"Scanned {debug_result.get('rows_scanned', 0)} rows | "
                        f"Pending {debug_result.get('pending_rows_found', 0)} | "
                        f"Skipped not final {debug_result.get('rows_skipped_not_final', 0)} | "
                        f"Skipped missing player/date {debug_result.get('rows_skipped_missing_player_date', 0)} | "
                        f"Updated {debug_result.get('rows_updated', 0)}"
                    )
                )

                status_placeholder.success(
                    f"Retry complete. Source: {debug_result.get('source_sheet', 'unknown')} | "
                    f"Pending: {debug_result.get('pending_rows_found', 0)} | "
                    f"Updated: {debug_result.get('rows_updated', 0)}"
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
        if st.button("📈 Rebuild Top Plays Live", use_container_width=True):
            status_placeholder.info("Rebuilding Top Plays Live from current odds feed...")
    
            try:
                odds_api_key = st.secrets["ODDS_API_KEY"]
    
                top_plays_df = get_top_plays_today_df(odds_api_key, debug=True)
    
                if top_plays_df is None or top_plays_df.empty:
                    status_placeholder.warning("Rebuild finished, but no qualifying top plays were found.")
    
                    write_admin_log(
                        action="rebuild_top_plays_live",
                        source="admin_manual",
                        status="success",
                        details="Manual rebuild ran successfully, but no qualifying top plays were found."
                    )
                else:
                    top_sheet = get_or_create_worksheet("Top Plays Live", rows=1000, cols=20)
    
                    output_df = top_plays_df.copy()
    
                    preferred_cols = [
                        "PLAYER_NAME",
                        "sportsbook",
                        "sportsbook_line",
                        "predicted_points",
                        "edge",
                        "model_pick",
                        "home_team",
                        "away_team",
                        "commence_time",
                    ]
    
                    output_cols = [col for col in preferred_cols if col in output_df.columns]
                    output_df = output_df[output_cols]
    
                    data = [output_df.columns.tolist()] + output_df.values.tolist()
    
                    top_sheet.clear()
                    top_sheet.update("A1", data)
    
                    status_placeholder.success(
                        f"Top Plays Live rebuild complete: {len(output_df)} rows written."
                    )
    
                    write_admin_log(
                        action="rebuild_top_plays_live",
                        source="admin_manual",
                        status="success",
                        details=f"Manual rebuild completed and wrote {len(output_df)} rows to Top Plays Live."
                    )
    
                st.cache_data.clear()
    
            except Exception as e:
                write_admin_log(
                    action="rebuild_top_plays_live",
                    source="admin_manual",
                    status="failed",
                    details=str(e)
                )
                status_placeholder.error(f"Top Plays rebuild failed: {e}")

   
    
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

    st.markdown("### Operation Debug Summary")

    debug_result = st.session_state.get("last_operations_debug")

    if not debug_result:
        st.info("Run one of the result update actions above to see debug output.")
    else:
        source_col, loaded_col = st.columns(2)

        with source_col:
            st.markdown(
                f"""
                <div class="status-box">
                    <div class="mini-label">Source Sheet</div>
                    <div class="mini-value">{debug_result.get('source_sheet', 'unknown')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with loaded_col:
            st.markdown(
                f"""
                <div class="status-box">
                    <div class="mini-label">Rows Loaded</div>
                    <div class="mini-value">{debug_result.get('total_data_rows_loaded', 0)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        dbg1, dbg2, dbg3, dbg4 = st.columns(4)

        with dbg1:
            st.markdown(
                f"""
                <div class="status-box">
                    <div class="mini-label">Rows Scanned</div>
                    <div class="mini-value">{debug_result.get('rows_scanned', 0)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with dbg2:
            st.markdown(
                f"""
                <div class="status-box">
                    <div class="mini-label">Skipped Not Final</div>
                    <div class="mini-value">{debug_result.get('rows_skipped_not_final', 0)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with dbg3:
            st.markdown(
                f"""
                <div class="status-box">
                    <div class="mini-label">Skipped Missing Player/Date</div>
                    <div class="mini-value">{debug_result.get('rows_skipped_missing_player_date', 0)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with dbg4:
            st.markdown(
                f"""
                <div class="status-box">
                    <div class="mini-label">Rows Updated</div>
                    <div class="mini-value">{debug_result.get('rows_updated', 0)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        dbg5, dbg6 = st.columns(2)

        with dbg5:
            st.markdown(
                f"""
                <div class="status-box">
                    <div class="mini-label">Pending Rows Found</div>
                    <div class="mini-value">{debug_result.get('pending_rows_found', 0)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with dbg6:
            st.markdown(
                f"""
                <div class="status-box">
                    <div class="mini-label">Other Skips</div>
                    <div class="mini-value">{debug_result.get('rows_skipped_other', 0)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with st.expander("Show row-level debug details"):
            row_debug_df = pd.DataFrame(debug_result.get("row_debug", []))
            if row_debug_df.empty:
                st.info("No row-level debug details returned.")
            else:
                st.dataframe(row_debug_df, use_container_width=True, hide_index=True, height=350)

    
    st.markdown("### Manual Add to Sheet1")

    sportsbook_options = get_available_sportsbooks()

    scan_col1, scan_col2, scan_col3 = st.columns([1, 1, 1])

    with scan_col1:
        st.markdown("Sportsbook")
        queue_sportsbook = st.selectbox(
            "Sportsbook",
            options=sportsbook_options,
            index=0,
            key="queue_sportsbook",
            label_visibility="collapsed"
        )

    with scan_col2:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("🔎 Scan Today's Lines to Queue", use_container_width=True):
            try:
                odds_api_key = st.secrets["ODDS_API_KEY"]
                props_df = shared_app.fetch_all_today_player_props(odds_api_key, queue_sportsbook)

                if props_df is None or props_df.empty:
                    status_placeholder.warning("No players with lines found for today's games.")
                else:
                    scan_df = props_df.copy()

                    scan_df = scan_df.dropna(subset=["player_name_raw", "line"]).copy()
                    scan_df["player_name_raw"] = scan_df["player_name_raw"].astype(str).str.strip()

                    # Save today's lines to Historical Lines sheet
                    historical_ws = get_historical_lines_sheet()
                    
                    existing_values = historical_ws.get_all_values()
                    
                    existing_keys = set()
                    if len(existing_values) > 1:
                        for row in existing_values[1:]:
                            if len(row) >= 4:
                                existing_player = str(row[0]).strip()
                                existing_date = str(row[1]).strip()
                                existing_line = str(row[2]).strip()
                                existing_book = str(row[3]).strip().lower()
                    
                                if (
                                    existing_player
                                    and existing_date
                                    and existing_line
                                    and existing_book
                                ):
                                    existing_keys.add((
                                        existing_player,
                                        existing_date,
                                        existing_line,
                                        existing_book
                                    ))
                    
                    today_date = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d")
                    captured_at = pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M:%S")
                    
                    rows_to_append = []
                    
                    for _, row in scan_df.iterrows():
                        player_name = normalize_name(str(row["player_name_raw"]).strip())
                        sportsbook_line = str(row["line"]).strip()
                        sportsbook_name = str(queue_sportsbook).strip().lower()
                    
                        history_key = (
                            player_name,
                            today_date,
                            sportsbook_line,
                            sportsbook_name
                        )
                    
                        if history_key not in existing_keys:
                            rows_to_append.append([
                                player_name,
                                today_date,
                                sportsbook_line,
                                sportsbook_name,
                                captured_at
                            ])
                            existing_keys.add(history_key)
                    
                    if rows_to_append:
                        historical_ws.append_rows(
                            rows_to_append,
                            value_input_option="USER_ENTERED"
                        )
                        print(f"Historical Lines updated: {len(rows_to_append)} new rows")
                    else:
                        print("Historical Lines already up to date")

                    
                    
                    scan_df = scan_df.sort_values(
                        by=["player_name_raw", "last_update"],
                        ascending=[True, False]
                    ).drop_duplicates(
                        subset=["player_name_raw"],
                        keep="first"
                    ).reset_index(drop=True)

                    sheet1_df = get_sheet1_df()

                    existing_sheet_keys = set()
                    if not sheet1_df.empty:
                        working_sheet1 = sheet1_df.copy()

                        for col in ["PLAYER_NAME", "sportsbook", "sportsbook_line"]:
                            if col not in working_sheet1.columns:
                                working_sheet1[col] = ""

                        for _, existing_row in working_sheet1.iterrows():
                            existing_player = str(existing_row.get("PLAYER_NAME", "")).strip()
                            existing_book = str(existing_row.get("sportsbook", "")).strip().lower()
                            existing_line = str(existing_row.get("sportsbook_line", "")).strip()

                            if existing_player and existing_book and existing_line:
                                try:
                                    existing_sheet_keys.add((
                                        existing_player,
                                        existing_book,
                                        float(existing_line)
                                    ))
                                except Exception:
                                    pass

                    existing_queue_keys = {
                        (
                            str(item["player_name"]).strip(),
                            str(item["sportsbook"]).strip().lower(),
                            float(item["sportsbook_line"]),
                        )
                        for item in st.session_state.manual_add_queue
                    }

                    added_count = 0

                    for _, row in scan_df.iterrows():
                        player_name = str(row["player_name_raw"]).strip()
                        sportsbook_line = float(row["line"])
                        sportsbook_key = str(queue_sportsbook).strip().lower()
                        last_update = str(row.get("last_update", "") or "")

                        queue_key = (player_name, sportsbook_key, sportsbook_line)

                        if queue_key in existing_queue_keys or queue_key in existing_sheet_keys:
                            continue

                        st.session_state.manual_add_queue.append({
                            "player_name": player_name,
                            "sportsbook": sportsbook_key,
                            "sportsbook_line": sportsbook_line,
                            "last_update": last_update,
                        })

                        existing_queue_keys.add(queue_key)
                        added_count += 1

                    if added_count == 0:
                        status_placeholder.info("0 lines added to queue. Queue already has today's lines.")
                    else:
                        status_placeholder.success(f"{added_count} lines added to queue.")

            except Exception as e:
                status_placeholder.error(f"Scan failed: {e}")

    with scan_col3:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("🧹 Clear Queue", use_container_width=True):
            st.session_state.manual_add_queue = []
            status_placeholder.info("Queue cleared.")

    queue_df = pd.DataFrame(st.session_state.manual_add_queue)

    st.markdown("#### Queue")
    if queue_df.empty:
        st.info("No players queued yet.")
    else:
        display_queue_df = queue_df.copy()

        preferred_cols = ["player_name", "sportsbook", "sportsbook_line", "last_update"]
        existing_cols = [col for col in preferred_cols if col in display_queue_df.columns]
        if existing_cols:
            display_queue_df = display_queue_df[existing_cols]

        st.dataframe(
            display_queue_df,
            use_container_width=True,
            hide_index=True,
            height=300
        )

    if st.button("📥 Load Queue to Sheet1", use_container_width=True):
        queue_items = st.session_state.manual_add_queue

        if not queue_items:
            status_placeholder.warning("Queue is empty.")
        else:
            loaded_count = 0
            failed_items = []
            progress_bar = st.progress(0)

            total_items = len(queue_items)

            for idx, item in enumerate(queue_items, start=1):
                try:
                    status_placeholder.info(
                        f"Writing {idx}/{total_items}: "
                        f"{item['player_name']} | {item['sportsbook']} {item['sportsbook_line']}"
                    )

                    result = shared_app.append_manual_play_to_sheet1(
                        player_name=item["player_name"],
                        sportsbook_key=item["sportsbook"],
                        sportsbook_line=item["sportsbook_line"],
                    )

                    loaded_count += 1

                except Exception as e:
                    error_msg = f"{item['player_name']} | {item['sportsbook']} {item['sportsbook_line']} | {e}"
                    failed_items.append(error_msg)
                    status_placeholder.error(error_msg)

                progress_bar.progress(idx / total_items)
                time.sleep(1.25)

            st.session_state.manual_add_queue = []
            st.cache_data.clear()
            write_admin_log(
                action="manual_queue_load",
                source="admin_manual",
                status="success" if not failed_items else "partial",
                details=(
                    f"Queued load finished | loaded={loaded_count} | "
                    f"failed={len(failed_items)} | total={total_items}"
                )
            )

            if failed_items:
                status_placeholder.warning(
                    f"Loaded {loaded_count} players. {len(failed_items)} failed."
                )
                with st.expander("Show queue load failures"):
                    for msg in failed_items:
                        st.write(msg)
            else:
                status_placeholder.success(f"Loaded {loaded_count} queued players into Sheet1.")

    
    
    st.markdown("### Sheet1 Pending Rows Snapshot")

    try:
        sheet1_df = get_sheet1_df()
        debug_summary = build_sheet1_debug_summary(sheet1_df)

        s1, s2, s3 = st.columns(3)

        with s1:
            st.markdown(
                f"""
                <div class="status-box">
                    <div class="mini-label">Total Sheet1 Rows</div>
                    <div class="mini-value">{debug_summary['total_rows']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with s2:
            st.markdown(
                f"""
                <div class="status-box">
                    <div class="mini-label">Pending Rows</div>
                    <div class="mini-value">{debug_summary['pending_rows']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with s3:
            st.markdown(
                f"""
                <div class="status-box">
                    <div class="mini-label">Completed Rows</div>
                    <div class="mini-value">{debug_summary['completed_rows']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with st.expander("Show rows that still look pending in Sheet1"):
            if debug_summary["pending_df"].empty:
                st.success("No pending rows detected from Sheet1 snapshot.")
            else:
                st.dataframe(
                    debug_summary["pending_df"],
                    use_container_width=True,
                    hide_index=True,
                    height=350
                )

    except Exception as e:
        st.error(f"Could not build Sheet1 debug summary: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


with logs_tab:
    st.markdown('<div class="section-card"><div class="section-title">Admin Logs</div>', unsafe_allow_html=True)

    logs_df = get_admin_logs_df()

    if st.button("Refresh Admin Logs"):
        st.cache_data.clear()
        st.rerun()
    
    if logs_df.empty:
        st.info("No admin logs yet.")
    else:
        display_logs_df = logs_df.copy()
        st.dataframe(
            display_logs_df.tail(50).iloc[::-1],
            use_container_width=True,
            hide_index=True,
            height=420
        )

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
