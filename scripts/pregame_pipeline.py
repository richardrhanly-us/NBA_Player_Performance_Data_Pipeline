import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shared_app import (
    fetch_all_today_player_props,
    get_sheet_records_df,
    get_historical_lines_sheet,
    normalize_name,
    format_event_game_date,
    BOOKMAKER_KEY,
)

from src.write_ops import append_manual_play_to_sheet1


def log(msg):
    print(msg, flush=True)


def get_db_connection():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not found in environment")
    return psycopg.connect(database_url)


def insert_line_snapshot(player_name, game_date, line, sportsbook):
    insert_sql = """
    INSERT INTO line_snapshots (
        player_name,
        game_date,
        sportsbook_line,
        sportsbook,
        captured_at
    ) VALUES (
        %s, %s, %s, %s, %s
    );
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                insert_sql,
                (
                    player_name,
                    game_date,
                    line,
                    sportsbook,
                    datetime.now(ZoneInfo("America/Chicago")),
                ),
            )
        conn.commit()


def normalize_last_update_for_sort(value):
    try:
        return pd.to_datetime(value, errors="coerce")
    except Exception:
        return pd.NaT


def build_scan_df(api_key, sportsbook_key):
    log(f"[PREGAME] Fetching props for sportsbook={sportsbook_key} ...")
    props_df = fetch_all_today_player_props(api_key, sportsbook_key)

    if props_df is None or props_df.empty:
        return pd.DataFrame()

    scan_df = props_df.copy()

    scan_df = scan_df.dropna(subset=["player_name_raw", "line"]).copy()
    scan_df["player_name_raw"] = scan_df["player_name_raw"].astype(str).str.strip()
    scan_df["normalized_name"] = scan_df["player_name_raw"].apply(normalize_name)
    scan_df["line"] = pd.to_numeric(scan_df["line"], errors="coerce")
    scan_df = scan_df.dropna(subset=["line"]).copy()

    if "last_update" not in scan_df.columns:
        scan_df["last_update"] = ""
    scan_df["last_update_sort"] = scan_df["last_update"].apply(normalize_last_update_for_sort)

    scan_df = scan_df.sort_values(
        by=["normalized_name", "last_update_sort"],
        ascending=[True, False]
    ).drop_duplicates(
        subset=["normalized_name"],
        keep="first"
    ).reset_index(drop=True)

    return scan_df


def append_historical_lines(scan_df, sportsbook_key):
    if scan_df is None or scan_df.empty:
        log("[PREGAME] Historical Lines skipped: no scan rows")
        return 0

    today_date = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d")
    inserted_count = 0

    for _, row in scan_df.iterrows():
        player_name = normalize_name(str(row["player_name_raw"]).strip())
        sportsbook_line = float(row["line"])
        sportsbook_name = str(sportsbook_key).strip().lower()

        try:
            insert_line_snapshot(
                player_name=player_name,
                game_date=today_date,
                line=sportsbook_line,
                sportsbook=sportsbook_name,
            )
            inserted_count += 1
        except Exception as e:
            log(f"[PREGAME] Neon line_snapshot insert failed -> {player_name} | {sportsbook_line} | {e}")

    log(f"[PREGAME] Neon line_snapshots inserted: {inserted_count}")

    # OLD GOOGLE SHEETS VERSION LEFT HERE ON PURPOSE FOR REFERENCE
    # historical_ws = get_historical_lines_sheet()
    # existing_values = historical_ws.get_all_values()
    #
    # existing_keys = set()
    # if len(existing_values) > 1:
    #     for row in existing_values[1:]:
    #         if len(row) >= 4:
    #             existing_player = str(row[0]).strip()
    #             existing_date = str(row[1]).strip()
    #             existing_line = str(row[2]).strip()
    #             existing_book = str(row[3]).strip().lower()
    #
    #             if existing_player and existing_date and existing_line and existing_book:
    #                 existing_keys.add((
    #                     existing_player,
    #                     existing_date,
    #                     existing_line,
    #                     existing_book
    #                 ))
    #
    # captured_at = pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M:%S")
    # rows_to_append = []
    #
    # for _, row in scan_df.iterrows():
    #     player_name = normalize_name(str(row["player_name_raw"]).strip())
    #     sportsbook_line = str(row["line"]).strip()
    #     sportsbook_name = str(sportsbook_key).strip().lower()
    #
    #     history_key = (
    #         player_name,
    #         today_date,
    #         sportsbook_line,
    #         sportsbook_name
    #     )
    #
    #     if history_key not in existing_keys:
    #         rows_to_append.append([
    #             player_name,
    #             today_date,
    #             sportsbook_line,
    #             sportsbook_name,
    #             captured_at
    #         ])
    #         existing_keys.add(history_key)
    #
    # if rows_to_append:
    #     historical_ws.append_rows(rows_to_append, value_input_option="USER_ENTERED")
    #     log(f"[PREGAME] Historical Lines appended: {len(rows_to_append)}")
    #     return len(rows_to_append)

    return inserted_count


def build_existing_sheet1_keys():
    sheet1_df = get_sheet_records_df()

    existing_sheet_keys = set()
    if sheet1_df is None or sheet1_df.empty:
        return existing_sheet_keys

    working_df = sheet1_df.copy()

    for col in ["PLAYER_NAME", "sportsbook", "sportsbook_line"]:
        if col not in working_df.columns:
            working_df[col] = ""

    for _, existing_row in working_df.iterrows():
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

    return existing_sheet_keys


def append_new_sheet1_rows(scan_df, sportsbook_key):
    if scan_df is None or scan_df.empty:
        log("[PREGAME] Sheet1 skipped: no scan rows")
        return {"loaded": 0, "failed": 0, "skipped_existing": 0}

    existing_sheet_keys = build_existing_sheet1_keys()

    loaded_count = 0
    failed_count = 0
    skipped_existing = 0
    total_items = len(scan_df)

    for idx, (_, row) in enumerate(scan_df.iterrows(), start=1):
        player_name = str(row["player_name_raw"]).strip()
        sportsbook_line = float(row["line"])
        sportsbook = str(sportsbook_key).strip().lower()
        last_update = str(row.get("last_update", "") or "")

        sheet_key = (player_name, sportsbook, sportsbook_line)

        if sheet_key in existing_sheet_keys:
            skipped_existing += 1
            log(f"[PREGAME] Sheet1 skip existing {idx}/{total_items} -> {player_name} | {sportsbook} | {sportsbook_line}")
            continue

        log(f"[PREGAME] Sheet1 append {idx}/{total_items} -> {player_name} | {sportsbook} | {sportsbook_line}")

        game_date = format_event_game_date(row.get("commence_time", ""))

        try:
            append_manual_play_to_sheet1(
                player_name=player_name,
                game_date=game_date,
                sportsbook_key=sportsbook,
                sportsbook_line=sportsbook_line,
                last_update=last_update,
            )
            loaded_count += 1
            existing_sheet_keys.add(sheet_key)

        except Exception as e:
            failed_count += 1
            log(f"[PREGAME] Sheet1 append failed -> {player_name} | {sportsbook_line} | {e}")

        time.sleep(1.25)

    return {
        "loaded": loaded_count,
        "failed": failed_count,
        "skipped_existing": skipped_existing,
    }


def main():
    start_time = time.time()
    log("[PREGAME] ===== START WORKFLOW =====")

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise ValueError("ODDS_API_KEY not found")

    sportsbook_key = BOOKMAKER_KEY

    scan_df = build_scan_df(api_key, sportsbook_key)

    if scan_df.empty:
        log("[PREGAME] No player props found. Exiting.")
        return

    log(f"[PREGAME] Scan rows after dedupe: {len(scan_df)}")

    historical_added = append_historical_lines(scan_df, sportsbook_key)
    sheet1_summary = append_new_sheet1_rows(scan_df, sportsbook_key)

    log(
        "[PREGAME] DONE | "
        f"historical_added={historical_added} | "
        f"sheet1_loaded={sheet1_summary['loaded']} | "
        f"sheet1_failed={sheet1_summary['failed']} | "
        f"sheet1_skipped_existing={sheet1_summary['skipped_existing']}"
    )
    log(f"[PREGAME] Runtime: {round(time.time() - start_time, 2)} seconds")
    log("[PREGAME] ===== END WORKFLOW =====")


if __name__ == "__main__":
    main()
