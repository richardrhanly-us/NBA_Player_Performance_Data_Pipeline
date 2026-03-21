import os
import json
import time
import requests
import joblib
import pandas as pd
import unicodedata
import gspread

from google.oauth2.service_account import Credentials
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
CURRENT_SEASON = "2025-26"
SHEET_KEY = "1uhjV_Si-qcILfNJbKZrD52y4JnT_GvqQ0hzN7POekQM"

BOOKMAKER_KEY = "draftkings"
EDGE_THRESHOLD = 3.0


def get_gsheet():
    service_account_info = json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(
        service_account_info,
        scopes=SCOPES
    )
    client = gspread.authorize(creds)
    return client.open_by_key(SHEET_KEY).sheet1


def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    return (
        name.lower()
        .replace(".", "")
        .replace("’", "'")
        .replace("-", " ")
        .replace(" jr", "")
        .replace(" sr", "")
        .replace(" iii", "")
        .replace(" ii", "")
        .strip()
    )


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def get_sheet_records_df():
    sheet = get_gsheet()
    values = sheet.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame(), sheet
    headers = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=headers), sheet


def fetch_upcoming_nba_events(api_key):
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    resp = requests.get(url, params={"apiKey": api_key}, timeout=20)
    resp.raise_for_status()
    return resp.json()


def fetch_player_points_market(api_key, event_id, bookmaker_key):
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
    resp = requests.get(
        url,
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "player_points",
            "bookmakers": bookmaker_key,
            "oddsFormat": "american"
        },
        timeout=20
    )
    resp.raise_for_status()
    return resp.json()


def fetch_all_today_player_props(api_key, bookmaker_key):
    print("Fetching NBA events from Odds API...", flush=True)
    events = fetch_upcoming_nba_events(api_key)
    print(f"Events found: {len(events)}", flush=True)

    rows = []

    for event_idx, event in enumerate(events, start=1):
        event_id = event.get("id")
        if not event_id:
            continue

        print(f"Reading event {event_idx}/{len(events)}: {event.get('away_team', '')} @ {event.get('home_team', '')}", flush=True)

        try:
            event_odds = fetch_player_points_market(api_key, event_id, bookmaker_key)
        except Exception as e:
            print(f"  Skipped event odds fetch: {e}", flush=True)
            continue

        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        for bookmaker in event_odds.get("bookmakers", []):
            book_title = bookmaker.get("title", "Unknown")

            for market in bookmaker.get("markets", []):
                if market.get("key") != "player_points":
                    continue

                market_last_update = market.get("last_update", "")
                grouped = {}

                for outcome in market.get("outcomes", []):
                    player_desc = outcome.get("description", "")
                    point = outcome.get("point")
                    side = outcome.get("name")

                    if not player_desc or point is None or side not in ("Over", "Under"):
                        continue

                    key = (normalize_name(player_desc), float(point))
                    if key not in grouped:
                        grouped[key] = {
                            "player_name_raw": player_desc,
                            "line": float(point),
                            "over_price": None,
                            "under_price": None
                        }

                    if side == "Over":
                        grouped[key]["over_price"] = outcome.get("price")
                    elif side == "Under":
                        grouped[key]["under_price"] = outcome.get("price")

                for _, item in grouped.items():
                    if item["over_price"] is None or item["under_price"] is None:
                        continue

                    rows.append({
                        "player_name_raw": item["player_name_raw"],
                        "line": item["line"],
                        "bookmaker": book_title,
                        "last_update": market_last_update,
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": event.get("commence_time", "")
                    })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["player_name_raw", "line", "bookmaker"]).reset_index(drop=True)
    print(f"Qualified prop rows collected before model scoring: {len(df)}", flush=True)
    return df


def load_model():
    print("Loading model...", flush=True)
    return joblib.load("models/points_regression.pkl")


def load_model_stats():
    with open("models/points_model_stats.json", "r") as f:
        return json.load(f)


def load_active_players():
    print("Loading active players map...", flush=True)
    active_players = players.get_active_players()
    actual_name_to_id = {p["full_name"]: p["id"] for p in active_players}
    normalized_to_actual = {}
    for actual_name in actual_name_to_id.keys():
        normalized_to_actual[normalize_name(actual_name)] = actual_name
    print(f"Active players mapped: {len(actual_name_to_id)}", flush=True)
    return actual_name_to_id, normalized_to_actual


def get_player_gamelog_df(player_id, season):
    for attempt in range(2):
        try:
            return playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                timeout=12
            ).get_data_frames()[0]
        except Exception as e:
            print(f"    Gamelog attempt {attempt + 1} failed: {e}", flush=True)
            if attempt == 1:
                return pd.DataFrame()
            time.sleep(2)


def build_player_feature_row(df, player_name):
    df = df.copy()
    df["PLAYER_NAME"] = player_name
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    numeric_cols = [
        "PTS", "FGM", "FGA", "FTA", "FTM", "OREB", "DREB",
        "STL", "AST", "BLK", "PF", "TOV", "MIN"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "FG3A" in df.columns:
        df["FG3A"] = pd.to_numeric(df["FG3A"], errors="coerce")

    df["gmsc"] = (
        df["PTS"]
        + 0.4 * df["FGM"]
        - 0.7 * df["FGA"]
        - 0.4 * (df["FTA"] - df["FTM"])
        + 0.7 * df["OREB"]
        + 0.3 * df["DREB"]
        + df["STL"]
        + 0.7 * df["AST"]
        + 0.7 * df["BLK"]
        - 0.4 * df["PF"]
        - df["TOV"]
    )

    df["player_avg_pts"] = df.groupby("PLAYER_NAME")["PTS"].transform(lambda x: x.shift(1).expanding().mean())
    df["player_avg_pts_sq"] = df["player_avg_pts"] ** 2
    df["last3_pts"] = df.groupby("PLAYER_NAME")["PTS"].transform(lambda x: x.shift(1).rolling(3).mean())
    df["last5_pts"] = df.groupby("PLAYER_NAME")["PTS"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["last10_pts"] = df.groupby("PLAYER_NAME")["PTS"].transform(lambda x: x.shift(1).rolling(10).mean())
    df["last20_pts"] = df.groupby("PLAYER_NAME")["PTS"].transform(lambda x: x.shift(1).rolling(20).mean())
    df["last5_fga"] = df.groupby("PLAYER_NAME")["FGA"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["last5_fta"] = df.groupby("PLAYER_NAME")["FTA"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["last5_minutes"] = df.groupby("PLAYER_NAME")["MIN"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["last5_gmsc"] = df.groupby("PLAYER_NAME")["gmsc"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["home_game"] = df["MATCHUP"].str.contains("vs").astype(int)
    df["days_rest"] = df.groupby("PLAYER_NAME")["GAME_DATE"].diff().dt.days.fillna(3)
    df["is_back_to_back"] = (df["days_rest"] == 1).astype(int)
    df["usage_proxy"] = df["FGA"] + 0.44 * df["FTA"] + df["TOV"]
    df["last5_usage_proxy"] = df.groupby("PLAYER_NAME")["usage_proxy"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["season_minutes_avg"] = df.groupby("PLAYER_NAME")["MIN"].transform(lambda x: x.shift(1).expanding().mean())
    df["minutes_volatility"] = df.groupby("PLAYER_NAME")["MIN"].transform(lambda x: x.shift(1).rolling(5).std())
    df["points_volatility"] = df.groupby("PLAYER_NAME")["PTS"].transform(lambda x: x.shift(1).rolling(5).std())

    if "FG3A" in df.columns:
        df["last5_3pa"] = df.groupby("PLAYER_NAME")["FG3A"].transform(lambda x: x.shift(1).rolling(5).mean())

    required_features = [
        "player_avg_pts",
        "player_avg_pts_sq",
        "season_minutes_avg",
        "home_game",
        "days_rest",
        "is_back_to_back",
        "last3_pts",
        "last5_pts",
        "last10_pts",
        "last20_pts",
        "last5_fga",
        "last5_fta",
        "last5_minutes",
        "last5_gmsc",
        "last5_usage_proxy",
        "minutes_volatility",
        "points_volatility"
    ]

    if "last5_3pa" in df.columns:
        required_features.append("last5_3pa")

    df_features = df.dropna(subset=required_features).reset_index(drop=True)
    if df_features.empty:
        return None

    latest = df_features.iloc[-1]

    feature_data = {
        "player_avg_pts": latest["player_avg_pts"],
        "player_avg_pts_sq": latest["player_avg_pts_sq"],
        "season_minutes_avg": latest["season_minutes_avg"],
        "home_game": latest["home_game"],
        "days_rest": latest["days_rest"],
        "is_back_to_back": latest["is_back_to_back"],
        "last3_pts": latest["last3_pts"],
        "last5_pts": latest["last5_pts"],
        "last10_pts": latest["last10_pts"],
        "last20_pts": latest["last20_pts"],
        "last5_fga": latest["last5_fga"],
        "last5_fta": latest["last5_fta"],
        "last5_minutes": latest["last5_minutes"],
        "last5_gmsc": latest["last5_gmsc"],
        "last5_usage_proxy": latest["last5_usage_proxy"],
        "minutes_volatility": latest["minutes_volatility"],
        "points_volatility": latest["points_volatility"]
    }

    if "last5_3pa" in df_features.columns and pd.notna(latest.get("last5_3pa", None)):
        feature_data["last5_3pa"] = latest["last5_3pa"]

    return pd.DataFrame([feature_data])


def already_logged(records_df, player_name, game_date, sportsbook, line):
    if records_df.empty:
        return False

    for _, row in records_df.iterrows():
        row_player = str(row.get("PLAYER_NAME", "")).strip()
        row_date = str(row.get("GAME_DATE", "")).strip()
        row_book = str(row.get("sportsbook", "")).strip()
        row_line = safe_float(row.get("sportsbook_line", ""))

        if (
            row_player == player_name
            and row_date == game_date
            and row_book == sportsbook
            and row_line == float(line)
        ):
            return True

    return False


def format_event_game_date(commence_time):
    try:
        return pd.to_datetime(commence_time, utc=True).tz_convert("US/Central").strftime("%B %d, %Y")
    except Exception:
        return pd.Timestamp.now(tz="US/Central").strftime("%B %d, %Y")


def append_to_sheet(sheet, player_name, game_date, line, sportsbook, last_update, predicted_points, model_pick):
    col_a = sheet.col_values(1)
    next_row = len(col_a) + 1

    values = [[
        player_name,
        str(game_date),
        float(line),
        sportsbook,
        last_update if last_update else "",
        f"{predicted_points:.2f}",
        "",
        "",
        model_pick,
        "",
        ""
    ]]

    sheet.update(range_name=f"A{next_row}:K{next_row}", values=values)


def main():
    odds_api_key = os.environ["ODDS_API_KEY"]
    model = load_model()
    model_stats = load_model_stats()
    actual_name_to_id, normalized_to_actual = load_active_players()
    model_feature_names = list(getattr(model, "feature_names_in_", []))

    props_df = fetch_all_today_player_props(odds_api_key, BOOKMAKER_KEY)
    if props_df.empty:
        print("No props found.", flush=True)
        return

    print(f"Props found for scoring: {len(props_df)}", flush=True)

    records_df, sheet = get_sheet_records_df()
    logged_count = 0
    total_props = len(props_df)

    for i, (_, row) in enumerate(props_df.iterrows(), start=1):
        print(f"Evaluating prop {i}/{total_props}: {row['player_name_raw']} | {row['bookmaker']} | {row['line']}", flush=True)

        normalized = normalize_name(row["player_name_raw"])
        actual_name = normalized_to_actual.get(normalized)

        if not actual_name:
            print("  Skipped: no active player match", flush=True)
            continue

        player_id = actual_name_to_id.get(actual_name)
        if not player_id:
            print("  Skipped: no player id found", flush=True)
            continue

        print(f"  Pulling gamelog for {actual_name}", flush=True)
        df = get_player_gamelog_df(player_id, CURRENT_SEASON)
        if df.empty:
            print("  Skipped: gamelog unavailable", flush=True)
            continue

        X = build_player_feature_row(df, actual_name)
        if X is None or X.empty:
            print("  Skipped: not enough games to build features", flush=True)
            continue

        if model_feature_names:
            X = X.reindex(columns=model_feature_names, fill_value=0)

        predicted_points = float(model.predict(X)[0])
        line = safe_float(row["line"])
        if line is None:
            print("  Skipped: invalid line", flush=True)
            continue

        edge = predicted_points - line
        if abs(edge) < EDGE_THRESHOLD:
            print(f"  Skipped: edge {edge:.2f} below threshold {EDGE_THRESHOLD}", flush=True)
            continue

        model_pick = "OVER" if predicted_points > line else "UNDER"
        game_date = format_event_game_date(row["commence_time"])

        if already_logged(records_df, actual_name, game_date, row["bookmaker"], line):
            print("  Skipped: already logged", flush=True)
            continue

        append_to_sheet(
            sheet=sheet,
            player_name=actual_name,
            game_date=game_date,
            line=line,
            sportsbook=row["bookmaker"],
            last_update=row["last_update"],
            predicted_points=predicted_points,
            model_pick=model_pick
        )

        logged_count += 1
        print(f"  Logged: {actual_name} | {row['bookmaker']} | {line} | edge={edge:.2f}", flush=True)
        time.sleep(0.5)

    print(f"Done. Logged {logged_count} top plays.", flush=True)


if __name__ == "__main__":
    main()
