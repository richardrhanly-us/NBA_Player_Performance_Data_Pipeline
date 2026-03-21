import os
import json
import time
import pandas as pd
import gspread

from datetime import datetime
from google.oauth2.service_account import Credentials
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
CURRENT_SEASON = "2025-26"
SHEET_KEY = "1uhjV_Si-qcILfNJbKZrD52y4JnT_GvqQ0hzN7POekQM"


def get_gsheet():
    service_account_info = json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(
        service_account_info,
        scopes=SCOPES
    )
    client = gspread.authorize(creds)
    return client.open_by_key(SHEET_KEY).sheet1


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def normalize_sheet_date(value):
    try:
        return pd.to_datetime(value).strftime("%B %d, %Y")
    except Exception:
        return str(value).strip()


def get_sheet_records_df():
    sheet = get_gsheet()
    values = sheet.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame(), sheet
    headers = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=headers), sheet


def get_player_id_by_name_map():
    active_players = players.get_active_players()
    return {p["full_name"]: p["id"] for p in active_players}


def get_final_points_from_gamelog(player_id, game_date):
    try:
        df = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=CURRENT_SEASON,
            timeout=10
        ).get_data_frames()[0]
    except Exception:
        return None

    if df.empty:
        return None

    df["GAME_DATE_FMT"] = pd.to_datetime(
        df["GAME_DATE"], errors="coerce"
    ).dt.strftime("%B %d, %Y")

    match = df[df["GAME_DATE_FMT"] == game_date]
    if match.empty:
        return None

    return int(match.iloc[0]["PTS"])


def update_all_pending_sheet_results():
    records_df, sheet = get_sheet_records_df()
    if records_df.empty:
        return 0, 0

    required_cols = [
        "PLAYER_NAME", "GAME_DATE", "sportsbook_line",
        "predicted_points", "final_points",
        "line_result", "model_pick", "model_result", "result_logged_at"
    ]
    for col in required_cols:
        if col not in records_df.columns:
            raise ValueError(f"Missing required column: {col}")

    player_name_map = get_player_id_by_name_map()

    updated_count = 0
    checked_count = 0

    for idx, row in records_df.iterrows():
        if str(row["final_points"]).strip():
            continue

        player_name = str(row["PLAYER_NAME"]).strip()
        game_date = normalize_sheet_date(row["GAME_DATE"])
        line_val = safe_float(row["sportsbook_line"])
        predicted_val = safe_float(row["predicted_points"])

        if not player_name or not game_date or line_val is None or predicted_val is None:
            continue

        player_id = player_name_map.get(player_name)
        if not player_id:
            continue

        checked_count += 1

        final_points = get_final_points_from_gamelog(player_id, game_date)
        if final_points is None:
            continue

        model_pick = "OVER" if predicted_val > line_val else "UNDER"
        line_result = "OVER" if final_points > line_val else "UNDER"
        model_result = "WIN" if model_pick == line_result else "LOSS"
        logged_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        sheet_row_num = idx + 2
        sheet.update(
            range_name=f"G{sheet_row_num}:K{sheet_row_num}",
            values=[[
                str(final_points),
                line_result,
                model_pick,
                model_result,
                logged_at
            ]]
        )

        updated_count += 1
        time.sleep(1)

    return updated_count, checked_count


if __name__ == "__main__":
    updated_count, checked_count = update_all_pending_sheet_results()
    print(f"Checked {checked_count} pending rows, updated {updated_count} rows.")
