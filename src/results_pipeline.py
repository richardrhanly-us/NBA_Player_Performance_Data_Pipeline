import time
import pandas as pd
import streamlit as st

from src.sheets_utils import (
    RESULTS_SHEET_NAME,
    clear_app_caches,
    get_historical_lines_sheet,
    get_worksheet_with_df,
    column_letter_from_index,
    build_header_index_map,
)

def normalize_sheet_date(value):
    if value is None:
        return ""
    try:
        return pd.to_datetime(value).strftime("%B %d, %Y")
    except Exception:
        return str(value).strip()


def get_final_points_from_gamelog(player_name, game_date, load_active_players, normalize_name, get_player_gamelog_df, CURRENT_SEASON, safe_float):
    actual_name_to_id, normalized_to_actual = load_active_players()
    normalized_player = normalize_name(player_name)
    actual_name = normalized_to_actual.get(normalized_player, player_name)
    player_id = actual_name_to_id.get(actual_name)
    if not player_id:
        return None
    print(f"[PIPELINE] Player mapping -> raw: {raw_name} | actual: {actual_name}", flush=True)
    
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


def is_blank_cell(value):
    return str(value).strip() == ""


def is_pending_result_row(row):
    bet_status = str(row.get("bet_status", "")).strip().upper()
    final_points_blank = is_blank_cell(row.get("final_points", ""))
    line_result_blank = is_blank_cell(row.get("line_result", ""))
    model_result_blank = is_blank_cell(row.get("model_result", ""))

    return (
        bet_status == "PENDING" or
        final_points_blank or
        line_result_blank or
        model_result_blank
    )


def update_sheet_with_final_result(
    worksheet,
    header_index_map,
    row_number,
    final_points,
    sportsbook_line,
    model_pick,
    safe_float,
    closing_line="",
    clv=""
):
    line = safe_float(sportsbook_line)
    points = safe_float(final_points)
    pick = str(model_pick).strip().upper()

    if line is None or points is None:
        return False

    if points > line:
        line_result = "OVER"
    elif points < line:
        line_result = "UNDER"
    else:
        line_result = "PUSH"

    if line_result == "PUSH":
        model_result = "PUSH"
        profit = 0
        bet_status = "PUSH"
    elif pick == line_result:
        model_result = "WIN"
        profit = 0.91
        bet_status = "WIN"
    else:
        model_result = "LOSS"
        profit = -1
        bet_status = "LOSS"

    timestamp = pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M:%S")

    updates = {
        "final_points": points,
        "line_result": line_result,
        "model_result": model_result,
        "result_logged_at": timestamp,
        "profit": profit,
        "bet_status": bet_status,
        "closing_line": closing_line,
        "clv": clv,
    }

    batch_payload = []
    for col_name, value in updates.items():
        if col_name not in header_index_map:
            continue
        col_letter = column_letter_from_index(header_index_map[col_name])
        batch_payload.append({
            "range": f"{col_letter}{row_number}",
            "values": [[value]]
        })

    if not batch_payload:
        return False

    worksheet.batch_update(batch_payload)
    clear_app_caches()
    return True


def populate_closing_lines_and_clv(strong_df, historical_df, normalize_name):
    if strong_df is None or strong_df.empty:
        return strong_df

    if historical_df is None or historical_df.empty:
        return strong_df

    strong_df = strong_df.copy()
    historical_df = historical_df.copy()

    for col in ["PLAYER_NAME", "GAME_DATE", "sportsbook", "sportsbook_line", "model_pick", "closing_line", "clv"]:
        if col not in strong_df.columns:
            strong_df[col] = ""

    for col in ["PLAYER_NAME", "GAME_DATE", "sportsbook", "sportsbook_line", "captured_at"]:
        if col not in historical_df.columns:
            historical_df[col] = ""

    strong_df["PLAYER_NAME"] = strong_df["PLAYER_NAME"].astype(str).apply(normalize_name)
    strong_df["sportsbook"] = strong_df["sportsbook"].astype(str).str.strip().str.lower()
    strong_df["model_pick"] = strong_df["model_pick"].astype(str).str.strip().str.upper()
    strong_df["sportsbook_line"] = pd.to_numeric(strong_df["sportsbook_line"], errors="coerce")
    strong_df["GAME_DATE"] = pd.to_datetime(strong_df["GAME_DATE"], errors="coerce").dt.date

    historical_df["PLAYER_NAME"] = historical_df["PLAYER_NAME"].astype(str).apply(normalize_name)
    historical_df["sportsbook"] = historical_df["sportsbook"].astype(str).str.strip().str.lower()
    historical_df["sportsbook_line"] = pd.to_numeric(historical_df["sportsbook_line"], errors="coerce")
    historical_df["GAME_DATE"] = pd.to_datetime(historical_df["GAME_DATE"], errors="coerce").dt.date
    historical_df["captured_at"] = pd.to_datetime(historical_df["captured_at"], errors="coerce")

    for idx, row in strong_df.iterrows():
        player_name = row["PLAYER_NAME"]
        game_date = row["GAME_DATE"]
        sportsbook = row["sportsbook"]
        bet_line = row["sportsbook_line"]
        model_pick = row["model_pick"]

        if pd.isna(game_date) or pd.isna(bet_line) or not player_name or not sportsbook:
            continue

        matches = historical_df[
            (historical_df["PLAYER_NAME"] == player_name) &
            (historical_df["GAME_DATE"] == game_date) &
            (historical_df["sportsbook"] == sportsbook)
        ].copy()

        if matches.empty:
            continue

        matches = matches.dropna(subset=["captured_at", "sportsbook_line"])
        if matches.empty:
            continue

        latest_row = matches.sort_values("captured_at", ascending=True).iloc[-1]
        closing_line = latest_row["sportsbook_line"]

        strong_df.at[idx, "closing_line"] = closing_line

        if model_pick == "OVER":
            strong_df.at[idx, "clv"] = round(closing_line - bet_line, 2)
        elif model_pick == "UNDER":
            strong_df.at[idx, "clv"] = round(bet_line - closing_line, 2)

    return strong_df


def update_all_pending_sheet_results(
    load_active_players,
    normalize_name,
    get_player_gamelog_df,
    CURRENT_SEASON,
    safe_float,
    debug=False
):
    source_sheet_name = RESULTS_SHEET_NAME
    worksheet, df, headers = get_worksheet_with_df(source_sheet_name)

    status_box = None
    progress_bar = None

    if debug:
        status_box = st.empty()
        progress_bar = st.progress(0)

    if df.empty:
        empty_result = {
            "source_sheet": source_sheet_name,
            "total_data_rows_loaded": 0,
            "rows_scanned": 0,
            "pending_rows_found": 0,
            "rows_skipped_not_final": 0,
            "rows_skipped_missing_player_date": 0,
            "rows_skipped_other": 0,
            "rows_updated": 0,
            "row_debug": [],
        }
        if debug and status_box is not None:
            status_box.info(f"No data rows found in {source_sheet_name}.")
            progress_bar.progress(1.0)
        return empty_result if debug else (0, 0)

    required_cols = [
        "PLAYER_NAME",
        "GAME_DATE",
        "sportsbook_line",
        "model_pick",
        "final_points",
        "line_result",
        "model_result",
        "bet_status",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{source_sheet_name} is missing required columns: {', '.join(missing_cols)}"
        )

    header_index_map = build_header_index_map(headers)

    historical_ws = get_historical_lines_sheet()
    historical_values = historical_ws.get_all_values()

    if historical_values and len(historical_values) > 1:
        historical_headers = [str(h).strip() for h in historical_values[0]]
        historical_rows = historical_values[1:]
        historical_df = pd.DataFrame(historical_rows, columns=historical_headers)
    else:
        historical_df = pd.DataFrame()

    df = populate_closing_lines_and_clv(df, historical_df, normalize_name)

    updated_count = 0
    checked_count = 0
    rows_scanned = 0
    pending_rows_found = 0
    rows_skipped_not_final = 0
    rows_skipped_missing_player_date = 0
    rows_skipped_other = 0
    row_debug = []

    today = pd.Timestamp.now(tz="America/Chicago").date()
    total_rows = len(df)

    for idx, row in df.iterrows():
        rows_scanned += 1
        sheet_row_number = idx + 2

        if not debug and (rows_scanned % 10 == 0 or rows_scanned == total_rows):
            print(
                f"[UPDATE RESULTS] Scanned {rows_scanned}/{total_rows} rows | "
                f"pending found={pending_rows_found} | updated={updated_count}",
                flush=True,
            )

        player_name = str(row.get("PLAYER_NAME", "")).strip()
        game_date = str(row.get("GAME_DATE", "")).strip()
        sportsbook_line = row.get("sportsbook_line", "")
        model_pick = row.get("model_pick", "")

        if debug and status_box is not None:
            status_box.markdown(
                f"""
                <div class="status-box">
                    <div><span class="muted">Update Final Results:</span> scanning {source_sheet_name}</div>
                    <div><span class="muted">Row:</span> {sheet_row_number} ({rows_scanned} of {total_rows})</div>
                    <div><span class="muted">Player:</span> {player_name or 'N/A'}</div>
                    <div><span class="muted">Game Date:</span> {game_date or 'N/A'}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            progress_bar.progress(min(rows_scanned / total_rows, 1.0))

        if not is_pending_result_row(row):
            if debug:
                row_debug.append({
                    "row_number": sheet_row_number,
                    "player_name": player_name,
                    "game_date": game_date,
                    "status": "not_pending",
                    "details": "Row already has result data",
                })
            continue

        pending_rows_found += 1
        checked_count += 1

        if not player_name or not game_date:
            rows_skipped_missing_player_date += 1
            if debug:
                row_debug.append({
                    "row_number": sheet_row_number,
                    "player_name": player_name,
                    "game_date": game_date,
                    "status": "skipped_missing_player_date",
                    "details": "Missing player name or game date",
                })
            continue

        try:
            parsed_game_date = pd.to_datetime(game_date).date()
        except Exception:
            rows_skipped_missing_player_date += 1
            if debug:
                row_debug.append({
                    "row_number": sheet_row_number,
                    "player_name": player_name,
                    "game_date": game_date,
                    "status": "skipped_bad_game_date",
                    "details": "Could not parse game date",
                })
            continue

        if parsed_game_date > today:
            rows_skipped_not_final += 1
            if debug:
                row_debug.append({
                    "row_number": sheet_row_number,
                    "player_name": player_name,
                    "game_date": game_date,
                    "status": "skipped_not_final",
                    "details": "Game date is in the future",
                })
            continue

        final_points = get_final_points_from_gamelog(
            player_name=player_name,
            game_date=game_date,
            load_active_players=load_active_players,
            normalize_name=normalize_name,
            get_player_gamelog_df=get_player_gamelog_df,
            CURRENT_SEASON=CURRENT_SEASON,
            safe_float=safe_float,
        )

        if final_points is None:
            rows_skipped_not_final += 1
            if debug:
                row_debug.append({
                    "row_number": sheet_row_number,
                    "player_name": player_name,
                    "game_date": game_date,
                    "status": "skipped_not_found_in_gamelog",
                    "details": "No final points found in player gamelog",
                })
            continue

        closing_line = row.get("closing_line", "")
        clv = row.get("clv", "")

        try:
            success = update_sheet_with_final_result(
                worksheet=worksheet,
                header_index_map=header_index_map,
                row_number=sheet_row_number,
                final_points=final_points,
                sportsbook_line=sportsbook_line,
                model_pick=model_pick,
                safe_float=safe_float,
                closing_line=closing_line,
                clv=clv,
            )

            if success:
                updated_count += 1
                if debug:
                    row_debug.append({
                        "row_number": sheet_row_number,
                        "player_name": player_name,
                        "game_date": game_date,
                        "status": "updated",
                        "details": f"Updated in {source_sheet_name} with final_points={final_points}",
                    })
                time.sleep(0.15)
            else:
                rows_skipped_other += 1
                if debug:
                    row_debug.append({
                        "row_number": sheet_row_number,
                        "player_name": player_name,
                        "game_date": game_date,
                        "status": "skipped_other",
                        "details": "No writable result columns found for this sheet",
                    })

        except Exception as e:
            rows_skipped_other += 1
            if debug:
                row_debug.append({
                    "row_number": sheet_row_number,
                    "player_name": player_name,
                    "game_date": game_date,
                    "status": "skipped_other",
                    "details": str(e),
                })

    clear_app_caches()

    result = {
        "source_sheet": source_sheet_name,
        "total_data_rows_loaded": total_rows,
        "rows_scanned": rows_scanned,
        "pending_rows_found": pending_rows_found,
        "rows_skipped_not_final": rows_skipped_not_final,
        "rows_skipped_missing_player_date": rows_skipped_missing_player_date,
        "rows_skipped_other": rows_skipped_other,
        "rows_updated": updated_count,
        "row_debug": row_debug,
    }

    if debug and status_box is not None:
        status_box.markdown(
            f"""
            <div class="status-box">
                <div><span class="muted">Update Final Results:</span> complete</div>
                <div><span class="muted">Source Sheet:</span> {source_sheet_name}</div>
                <div><span class="muted">Rows Scanned:</span> {rows_scanned}</div>
                <div><span class="muted">Pending Found:</span> {pending_rows_found}</div>
                <div><span class="muted">Rows Updated:</span> {updated_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        progress_bar.progress(1.0)

    if debug:
        return result

    if not debug:
        print(
            f"[UPDATE RESULTS] DONE | scanned={rows_scanned} | "
            f"pending={pending_rows_found} | updated={updated_count} | "
            f"skipped_not_final={rows_skipped_not_final} | "
            f"skipped_missing_player_date={rows_skipped_missing_player_date} | "
            f"skipped_other={rows_skipped_other}",
            flush=True,
        )

    if debug:
        return result

    return updated_count, checked_count
