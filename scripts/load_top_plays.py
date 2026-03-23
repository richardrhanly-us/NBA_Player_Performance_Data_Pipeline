import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time

from src.shared_app import (
    append_manual_play_to_sheet1,
    get_strong_plays_df,
    get_top_plays_today_df,
    get_gsheet_client,
    SHEET_KEY,
)


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
        print("[TOP PLAYS] No data available -> writing placeholder", flush=True)
        sheet.clear()
        sheet.update(range_name="A1", values=[["No data available"]])
        return 0

    print(f"[TOP PLAYS] Writing {len(df)} rows to Top Plays Live", flush=True)

    sheet.clear()
    sheet.update(
        range_name="A1",
        values=[df.columns.values.tolist()] + df.values.tolist()
    )

    return len(df)

def main():
    print("[TOP PLAYS] ===== START WORKFLOW =====", flush=True)
    print("[TOP PLAYS] Building top plays from shared pipeline...", flush=True)

    top_df = get_top_plays_today_df(debug=False)

    if top_df is None or top_df.empty:
        print("[TOP PLAYS] No qualifying top plays returned from shared pipeline.", flush=True)
        rows_written = update_top_plays_live_sheet(top_df)
        print(
            f"[TOP PLAYS] DONE | final={rows_written} | appended=0",
            flush=True,
        )
        return

    print(f"[TOP PLAYS] Final top plays: {len(top_df)}", flush=True)
    rows_written = update_top_plays_live_sheet(top_df)

    records_df = get_strong_plays_df()
    appended_count = 0

    for _, row in top_df.iterrows():
        player_name = row.get("PLAYER_NAME", "")
        sportsbook = str(row.get("sportsbook", "")).lower()
        line = row.get("sportsbook_line", "")
        predicted_points = row.get("predicted_points", "")
        model_pick = row.get("model_pick", "")
        game_date = row.get("GAME_DATE", "")
        if not game_date:
            try:
                game_date = pd.to_datetime(row.get("commence_time", ""), utc=True)\
                    .tz_convert("US/Central")\
                    .strftime("%B %d, %Y")
            except Exception:
                game_date = ""

        if already_logged(records_df, player_name, game_date, sportsbook, line):
            print(
                f"[TOP PLAYS] Already in Strong Plays, skipping append: "
                f"{player_name} | {sportsbook} | {line}",
                flush=True,
            )
            continue

        append_manual_play_to_sheet1(
            player_name=player_name,
            game_date=game_date,
            sportsbook_line=line,
            sportsbook=sportsbook,
            predicted_points=predicted_points,
            model_pick=model_pick,
        )

        appended_count += 1
        print(
            f"[TOP PLAYS] Appended to Strong Plays: "
            f"{player_name} | {sportsbook} | {line}",
            flush=True,
        )
        time.sleep(0.5)

    print(
        f"[TOP PLAYS] DONE | final={rows_written} | appended={appended_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
