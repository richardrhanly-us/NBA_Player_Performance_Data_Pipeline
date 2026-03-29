import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shared_app import get_top_plays_today_df, get_gsheet_client, SHEET_KEY


def run_top_plays_rebuild():
    print("[TOP PLAYS] Starting rebuild...", flush=True)

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise ValueError("ODDS_API_KEY not found")

    top_df = get_top_plays_today_df(api_key, debug=False)

    client = get_gsheet_client()
    sheet = client.open_by_key(SHEET_KEY)

    try:
        top_sheet = sheet.worksheet("Top Plays Live")
    except Exception:
        top_sheet = sheet.add_worksheet(title="Top Plays Live", rows=1000, cols=20)

    top_sheet.clear()

    if top_df is None or top_df.empty:
        print("[TOP PLAYS] No top plays returned.", flush=True)
        top_sheet.update("A1", [["No data available"]])
        return

    top_sheet.update([top_df.columns.values.tolist()] + top_df.values.tolist())
    print(f"[TOP PLAYS] Wrote {len(top_df)} rows.", flush=True)


if __name__ == "__main__":
    run_top_plays_rebuild()