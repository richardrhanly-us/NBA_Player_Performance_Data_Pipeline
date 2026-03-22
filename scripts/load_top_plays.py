import os
import json
import time
import requests
import joblib
import pandas as pd
import unicodedata
import gspread
import random

from google.oauth2.service_account import Credentials
from nba_api.stats.static import players
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
CURRENT_SEASON = "2025-26"
SHEET_KEY = "1uhjV_Si-qcILfNJbKZrD52y4JnT_GvqQ0hzN7POekQM"

BOOKMAKER_KEY = "draftkings"
EDGE_THRESHOLD = 3.0
TOP_PLAYS_LIMIT = 15
NBA_STATS_HEADERS = {
    "Host": "stats.nba.com",
    "Connection": "keep-alive",
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def get_gsheet_client():
    service_account_info = json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(
        service_account_info,
        scopes=SCOPES
    )
    return gspread.authorize(creds)


def get_gsheet():
    client = get_gsheet_client()
    return client.open_by_key(SHEET_KEY).sheet1


def get_top_plays_live_sheet():
    client = get_gsheet_client()
    return client.open_by_key(SHEET_KEY).worksheet("Top Plays Live")

def update_top_plays_live_sheet(df):
    sheet = get_top_plays_live_sheet()

    if df is None or df.empty:
        print("[TOP PLAYS] No data available -> writing placeholder", flush=True)
        sheet.clear()
        sheet.update("A1", [["No data available"]])
        return 0

    print(f"[TOP PLAYS] Writing {len(df)} rows to Top Plays Live", flush=True)

    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

    return len(df)

def build_nba_session():
    session = requests.Session()

    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(NBA_STATS_HEADERS)

    return session

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
            "oddsFormat": "american",
        },
        timeout=20,
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

        print(
            f"Reading event {event_idx}/{len(events)}: "
            f"{event.get('away_team', '')} @ {event.get('home_team', '')}",
            flush=True,
        )

        try:
            event_odds = fetch_player_points_market(api_key, event_id, bookmaker_key)
        except Exception as e:
            print(f"  Skipped event odds fetch: {e}", flush=True)
            continue

        time.sleep(0.3)

        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        commence_time = event.get("commence_time", "")

        for bookmaker in event_odds.get("bookmakers", []):
            book_title = bookmaker.get("title", "Unknown")
            book_key = bookmaker.get("key", bookmaker_key)

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
                            "under_price": None,
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
                        "bookmaker_key": str(book_key).lower(),
                        "last_update": market_last_update,
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": commence_time,
                    })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(
        subset=["player_name_raw", "line", "bookmaker_key", "commence_time"]
    ).reset_index(drop=True)

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
    session = build_nba_session()

    for attempt in range(4):
        try:
            print(
                f"Pulling gamelog for player_id={player_id} season={season} "
                f"(attempt {attempt + 1}/4)",
                flush=True,
            )

            url = "https://stats.nba.com/stats/playergamelog"
            params = {
                "DateFrom": "",
                "DateTo": "",
                "LeagueID": "00",
                "PlayerID": str(player_id),
                "Season": season,
                "SeasonType": "Regular Season",
            }

            resp = session.get(url, params=params, timeout=25)
            resp.raise_for_status()
            data = resp.json()

            result_sets = data.get("resultSets", [])
            if not result_sets:
                print("No resultSets returned from stats.nba.com", flush=True)
                return pd.DataFrame()

            headers = result_sets[0].get("headers", [])
            rows = result_sets[0].get("rowSet", [])

            if not headers or not rows:
                print("Empty gamelog returned from stats.nba.com", flush=True)
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=headers)

            if df is not None and not df.empty:
                time.sleep(random.uniform(0.4, 1.1))
                return df

        except Exception as e:
            wait_time = min(2 ** attempt + random.uniform(0.5, 1.5), 8)
            print(f"Gamelog attempt {attempt + 1} failed: {e}", flush=True)
            print(f"Waiting {wait_time:.1f}s before retry...", flush=True)
            time.sleep(wait_time)

    return pd.DataFrame()


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
        "points_volatility",
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
        "points_volatility": latest["points_volatility"],
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
        row_book = str(row.get("sportsbook", "")).strip().lower()
        row_line = safe_float(row.get("sportsbook_line", ""))

        if (
            row_player == player_name
            and row_date == game_date
            and row_book == str(sportsbook).strip().lower()
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
        "",
    ]]

    sheet.update(range_name=f"A{next_row}:K{next_row}", values=values)


def add_scored_row(scored_map, row_dict):
    key = (
        row_dict["PLAYER_NAME"],
        row_dict["sportsbook"],
        float(row_dict["sportsbook_line"]),
        row_dict["commence_time"],
    )

    existing = scored_map.get(key)
    if existing is None or abs(row_dict["edge"]) > abs(existing["edge"]):
        scored_map[key] = row_dict


def build_top_plays_live_df(scored_map):
    if not scored_map:
        return pd.DataFrame()

    top_df = pd.DataFrame(scored_map.values()).copy()

    if top_df.empty:
        return pd.DataFrame()

    top_df["sportsbook_line"] = pd.to_numeric(top_df["sportsbook_line"], errors="coerce")
    top_df["predicted_points"] = pd.to_numeric(top_df["predicted_points"], errors="coerce")
    top_df["edge"] = pd.to_numeric(top_df["edge"], errors="coerce")

    top_df = top_df.dropna(subset=["sportsbook_line", "predicted_points", "edge"])
    top_df = top_df[top_df["sportsbook_line"] > 0].copy()

    if top_df.empty:
        return pd.DataFrame()

    top_df["abs_edge"] = top_df["edge"].abs()
    top_df = top_df.sort_values("abs_edge", ascending=False).drop(columns=["abs_edge"])
    top_df = top_df.head(TOP_PLAYS_LIMIT).reset_index(drop=True)

    return top_df


def process_prop_row(
    row_data,
    model,
    model_feature_names,
    actual_name_to_id,
    normalized_to_actual,
    gamelog_cache,
):
    normalized = normalize_name(row_data["player_name_raw"])
    actual_name = normalized_to_actual.get(normalized)

    if not actual_name:
        print("  Skipped: no active player match", flush=True)
        return None, "retry_not_needed"

    player_id = actual_name_to_id.get(actual_name)
    if not player_id:
        print("  Skipped: no player id found", flush=True)
        return None, "retry_not_needed"

    if player_id in gamelog_cache:
        df = gamelog_cache[player_id]
        print(f"  Using cached gamelog for {actual_name}", flush=True)
    else:
        print(f"  Pulling gamelog for {actual_name}", flush=True)
        df = get_player_gamelog_df(player_id, CURRENT_SEASON)
        if not df.empty:
            gamelog_cache[player_id] = df

    if df.empty:
        print("  Queued for retry: gamelog unavailable", flush=True)
        return None, "retry"

    X = build_player_feature_row(df, actual_name)
    if X is None or X.empty:
        print("  Skipped: not enough games to build features", flush=True)
        return None, "retry_not_needed"

    if model_feature_names:
        X = X.reindex(columns=model_feature_names, fill_value=0)

    predicted_points = float(model.predict(X)[0])

    line = safe_float(row_data["line"])
    if line is None:
        print("  Skipped: invalid line", flush=True)
        return None, "retry_not_needed"

    edge = predicted_points - line
    if abs(edge) < EDGE_THRESHOLD:
        print(f"  Skipped: edge {edge:.2f} below threshold {EDGE_THRESHOLD}", flush=True)
        return None, "retry_not_needed"

    model_pick = "OVER" if predicted_points > line else "UNDER"

    scored_row = {
        "PLAYER_NAME": actual_name,
        "sportsbook": str(row_data.get("bookmaker_key", BOOKMAKER_KEY)).lower(),
        "sportsbook_line": float(line),
        "predicted_points": round(predicted_points, 2),
        "edge": round(edge, 2),
        "model_pick": model_pick,
        "home_team": row_data.get("home_team", ""),
        "away_team": row_data.get("away_team", ""),
        "commence_time": row_data.get("commence_time", ""),
        "last_update": row_data.get("last_update", ""),
    }

    append_payload = {
        "player_name": actual_name,
        "game_date": format_event_game_date(row_data.get("commence_time", "")),
        "line": float(line),
        "sportsbook": str(row_data.get("bookmaker_key", BOOKMAKER_KEY)).lower(),
        "last_update": row_data.get("last_update", ""),
        "predicted_points": predicted_points,
        "model_pick": model_pick,
        "edge": edge,
    }

    return (scored_row, append_payload), "success"

def main():
    print("[TOP PLAYS] ===== START WORKFLOW =====", flush=True)

    odds_api_key = os.environ["ODDS_API_KEY"]

    events = fetch_upcoming_nba_events(odds_api_key)
    event_count = len(events)
    print(f"[TOP PLAYS] Events found: {event_count}", flush=True)

    model = load_model()
    load_model_stats()
    actual_name_to_id, normalized_to_actual = load_active_players()
    model_feature_names = list(getattr(model, "feature_names_in_", []))

    props_df = fetch_all_today_player_props(odds_api_key, BOOKMAKER_KEY)
    raw_props_count = len(props_df)
    
    if not props_df.empty:
        props_df = props_df.sort_values(
            by=["player_name_raw", "last_update"],
            ascending=[True, False]
        ).drop_duplicates(
            subset=["player_name_raw"],
            keep="first"
        ).reset_index(drop=True)
    
    props_count = len(props_df)
    print(f"[TOP PLAYS] Raw props found: {raw_props_count}", flush=True)
    print(f"[TOP PLAYS] Unique player props after dedupe: {props_count}", flush=True)

    if props_df.empty:
        print("[TOP PLAYS] No props found.", flush=True)
        rows_written = update_top_plays_live_sheet(pd.DataFrame())
        print(
            f"[TOP PLAYS] DONE | events={event_count} | props={props_count} | "
            f"scored=0 | final={rows_written} | appended=0",
            flush=True,
        )
        return

    records_df, sheet = get_sheet_records_df()
    total_props = len(props_df)

    gamelog_cache = {}
    retry_rows = []
    scored_map = {}

    for i, (_, row) in enumerate(props_df.iterrows(), start=1):
        row_data = row.to_dict()

        print(
            f"Evaluating prop {i}/{total_props}: "
            f"{row_data['player_name_raw']} | {row_data['bookmaker']} | {row_data['line']}",
            flush=True,
        )

        result, status = process_prop_row(
            row_data=row_data,
            model=model,
            model_feature_names=model_feature_names,
            actual_name_to_id=actual_name_to_id,
            normalized_to_actual=normalized_to_actual,
            gamelog_cache=gamelog_cache,
        )

        if status == "retry":
            retry_rows.append(row_data)
            continue

        if not result:
            continue

        scored_row, _ = result
        add_scored_row(scored_map, scored_row)
        
        time.sleep(random.uniform(0.15, 0.45))

    if retry_rows:
        print(f"Starting retry pass for {len(retry_rows)} queued props...", flush=True)
        time.sleep(3)

        for j, row_data in enumerate(retry_rows, start=1):
            print(
                f"Retrying prop {j}/{len(retry_rows)}: "
                f"{row_data['player_name_raw']} | {row_data['bookmaker']} | {row_data['line']}",
                flush=True,
            )

            result, status = process_prop_row(
                row_data=row_data,
                model=model,
                model_feature_names=model_feature_names,
                actual_name_to_id=actual_name_to_id,
                normalized_to_actual=normalized_to_actual,
                gamelog_cache=gamelog_cache,
            )

            if not result:
                if status == "retry":
                    print("  Retry failed: gamelog still unavailable", flush=True)
                continue

            scored_row, _ = result
            add_scored_row(scored_map, scored_row)

            time.sleep(random.uniform(0.15, 0.45))

    scored_count = len(scored_map)
    print(f"[TOP PLAYS] Scored rows: {scored_count}", flush=True)

    top_df = build_top_plays_live_df(scored_map)

    if top_df.empty:
        print("[TOP PLAYS] No qualifying top plays after filters", flush=True)
        rows_written = update_top_plays_live_sheet(pd.DataFrame())
        print(
            f"[TOP PLAYS] DONE | events={event_count} | props={props_count} | "
            f"scored={scored_count} | final={rows_written} | appended=0",
            flush=True,
        )
        return

    print(f"[TOP PLAYS] Final top plays: {len(top_df)}", flush=True)
    rows_written = update_top_plays_live_sheet(top_df)

    appended_count = 0

    for _, row in top_df.iterrows():
        player_name = row["PLAYER_NAME"]
        sportsbook = row["sportsbook"]
        line = row["sportsbook_line"]
        predicted_points = row["predicted_points"]
        model_pick = row["model_pick"]
        last_update = row.get("last_update", "")
        edge = row.get("edge", 0)

        game_date = datetime.now().strftime("%B %d, %Y")

        if already_logged(records_df, player_name, game_date, sportsbook, line):
            print(
                f"[TOP PLAYS] Already in Sheet1, skipping append: "
                f"{player_name} | {sportsbook} | {line}",
                flush=True,
            )
            continue

        append_to_sheet(
            sheet=sheet,
            player_name=player_name,
            game_date=game_date,
            line=line,
            sportsbook=sportsbook,
            last_update=last_update,
            predicted_points=predicted_points,
            model_pick=model_pick,
        )

        appended_count += 1
        print(
            f"[TOP PLAYS] Appended to Sheet1: "
            f"{player_name} | {sportsbook} | {line} | edge={edge:.2f}",
            flush=True,
        )
        time.sleep(0.5)

    print(
        f"[TOP PLAYS] DONE | events={event_count} | props={props_count} | "
        f"scored={scored_count} | final={rows_written} | appended={appended_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
