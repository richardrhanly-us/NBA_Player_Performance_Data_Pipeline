import sys
import os
import time
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shared_app import (
    get_strong_plays_df,
    get_top_plays_today_df,
    get_gsheet_client,
    SHEET_KEY,
)

from src.write_ops import append_play_to_strong_plays

def log(msg):
    print(msg, flush=True)


def already_logged(records_df, player_name, game_date, sportsbook, line):
    if records_df is None or records_df.empty:
        return False

    for _, row in records_df.iterrows():
        row_player = str(row.get("PLAYER_NAME", "")).strip()
        row_date = str(row.get("GAME_DATE", "")).strip()
        row_book = str(row.get("sportsbook", "")).strip().lower()

        try:
            row_line = float(row.get("sportsbook_line", ""))
        except Exception:
            row_line = None

        try:
            target_line = float(line)
        except Exception:
            target_line = None

        if (
            row_player == str(player_name).strip()
            and row_date == str(game_date).strip()
            and row_book == str(sportsbook).strip().lower()
            and row_line == target_line
        ):
            return True

    return False


def get_top_plays_live_sheet():
    client = get_gsheet_client()
    return client.open_by_key(SHEET_KEY).worksheet("Top Plays Live")


def update_top_plays_live_sheet(df):
    sheet = get_top_plays_live_sheet()

    if df is None or df.empty:
        log("[TOP PLAYS] No data available -> writing placeholder")
        sheet.clear()
        sheet.update(range_name="A1", values=[["No data available"]])
        return 0

    log(f"[TOP PLAYS] Writing {len(df)} rows to Top Plays Live")

    sheet.clear()
    sheet.update(
        range_name="A1",
        values=[df.columns.values.tolist()] + df.values.tolist()
    )

    return len(df)


def main():
    start_time = time.time()
    log("[TOP PLAYS] ===== START WORKFLOW =====")
    log("[TOP PLAYS] Calling get_top_plays_today_df()...")
    
    odds_api_key = os.environ["ODDS_API_KEY"]
    
    top_df = get_top_plays_today_df(api_key=odds_api_key, debug=True)
    
    log("[TOP PLAYS] Returned from get_top_plays_today_df()")

    if top_df is None or top_df.empty:
        log("[TOP PLAYS] No qualifying top plays returned from shared pipeline.")
        rows_written = update_top_plays_live_sheet(top_df)
        log(f"[TOP PLAYS] DONE | final={rows_written} | appended=0")
        log(f"[TOP PLAYS] Runtime: {round(time.time() - start_time, 2)} seconds")
        return

    log(f"[TOP PLAYS] Final top plays: {len(top_df)}")
    rows_written = update_top_plays_live_sheet(top_df)

    records_df = get_strong_plays_df()
    appended_count = 0
    total_rows = len(top_df)

    for i, (_, row) in enumerate(top_df.iterrows(), start=1):
        player_name = row.get("PLAYER_NAME", "")
        sportsbook = str(row.get("sportsbook", "")).lower()
        line = row.get("sportsbook_line", "")
        predicted_points = row.get("predicted_points", "")
        model_pick = row.get("model_pick", "")
        last_update = row.get("last_update", "")
        edge = row.get("edge", 0)

        game_date = row.get("GAME_DATE", "")
        if not game_date:
            try:
                game_date = (
                    pd.to_datetime(row.get("commence_time", ""), utc=True)
                    .tz_convert("US/Central")
                    .strftime("%B %d, %Y")
                )
            except Exception:
                game_date = ""

        log(f"[TOP PLAYS] Processing {i}/{total_rows} -> {player_name} | {sportsbook} | {line}")

        if already_logged(records_df, player_name, game_date, sportsbook, line):
            log(f"[TOP PLAYS] SKIP (already exists) -> {player_name} | {sportsbook} | {line}")
            continue

        append_play_to_strong_plays(
            player_name=player_name,
            game_date=game_date,
            sportsbook_line=line,
            sportsbook=sportsbook,
            predicted_points=predicted_points,
            model_pick=model_pick,
            last_update=last_update,
            edge=edge,
        )

        appended_count += 1
        log(f"[TOP PLAYS] APPENDED -> {player_name} | {sportsbook} | {line} | edge={edge}")
        time.sleep(0.5)

    log(f"[TOP PLAYS] DONE | final={rows_written} | appended={appended_count}")
    log(f"[TOP PLAYS] Runtime: {round(time.time() - start_time, 2)} seconds")
    log("[TOP PLAYS] ===== END WORKFLOW =====")


if __name__ == "__main__":
    main()
