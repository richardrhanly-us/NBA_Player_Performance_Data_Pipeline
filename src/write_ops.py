import pandas as pd

from src.sheets_utils import (
    get_results_sheet,
    get_strong_plays_sheet,
    clear_app_caches,
)

from src.shared_app import (
    load_active_players,
    normalize_name,
    get_player_gamelog_df,
    build_player_feature_row,
    load_model,
    get_player_points_lines,
    format_sportsbook_name,
    format_event_game_date,
    safe_float,
    CURRENT_SEASON,
    BOOKMAKER_KEY,
)


def append_manual_play_to_sheet1(
    player_name,
    game_date=None,
    sportsbook_line=None,
    sportsbook=None,
    predicted_points=None,
    model_pick=None,
    sportsbook_key=None,
    last_update=""
):
    actual_name_to_id, normalized_to_actual = load_active_players()
    normalized = normalize_name(player_name)
    actual_name = normalized_to_actual.get(normalized, player_name)

    player_id = actual_name_to_id.get(actual_name)
    if not player_id:
        raise ValueError(f"No active player id found for {player_name}")

    sportsbook_lookup = sportsbook_key or sportsbook or BOOKMAKER_KEY
    line_data = None

    if predicted_points is None or model_pick is None or sportsbook_line is None:
        df = get_player_gamelog_df(player_id, CURRENT_SEASON)
        if df is None or df.empty:
            raise ValueError(f"Could not load gamelog for {actual_name}")

        X = build_player_feature_row(df, actual_name)
        if X is None or X.empty:
            raise ValueError(f"Not enough data to build features for {actual_name}")

        model = load_model()
        model_feature_names = list(getattr(model, "feature_names_in_", []))
        if model_feature_names:
            X = X.reindex(columns=model_feature_names, fill_value=0)

        predicted_points = float(model.predict(X)[0])

        line_data = get_player_points_lines(actual_name, sportsbook_lookup)

        if sportsbook_line is None:
            if not line_data or line_data.get("points_line") is None:
                raise ValueError(f"No live {sportsbook_lookup} line found for {actual_name}")
            sportsbook_line = float(line_data["points_line"])
        else:
            sportsbook_line = float(sportsbook_line)

        model_pick = "OVER" if predicted_points > sportsbook_line else "UNDER"

    sportsbook_line = float(sportsbook_line)
    predicted_points = float(predicted_points)
    model_pick = str(model_pick).strip().upper()

    if not game_date:
        if line_data and line_data.get("commence_time"):
            game_date = format_event_game_date(line_data.get("commence_time"))
        else:
            game_date = pd.Timestamp.now(tz="America/Chicago").strftime("%B %d, %Y")

    if not sportsbook:
        if line_data and line_data.get("sportsbook"):
            sportsbook = line_data.get("sportsbook")
        else:
            sportsbook = sportsbook_lookup

    sportsbook = format_sportsbook_name(sportsbook)

    if not last_update and line_data:
        last_update = line_data.get("last_update", "") or ""

    edge = round(predicted_points - sportsbook_line, 2)

    sheet = get_results_sheet()
    values = sheet.get_all_values()
    next_row = len(values) + 1 if values else 2

    row_values = [[
        actual_name,
        str(game_date),
        sportsbook_line,
        sportsbook,
        last_update,
        round(predicted_points, 2),
        "",
        "",
        model_pick,
        "",
        "",
    ]]

    sheet.update(range_name=f"A{next_row}:K{next_row}", values=row_values)
    clear_app_caches()

    return {
        "player_name": actual_name,
        "sportsbook": sportsbook,
        "sportsbook_line": sportsbook_line,
        "predicted_points": round(predicted_points, 2),
        "edge": edge,
        "model_pick": model_pick,
        "sheet_row": next_row,
    }


def append_play_to_strong_plays(
    player_name,
    game_date,
    sportsbook_line,
    sportsbook,
    predicted_points,
    model_pick,
    last_update="",
    edge=None
):
    actual_name = str(player_name).strip()
    game_date = str(game_date).strip()
    sportsbook = format_sportsbook_name(sportsbook)
    sportsbook_line = float(sportsbook_line)
    predicted_points = float(predicted_points)
    model_pick = str(model_pick).strip().upper()

    if edge is None:
        edge = round(predicted_points - sportsbook_line, 2)
    else:
        edge = float(edge)

    captured_at = pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M:%S")

    sheet = get_strong_plays_sheet()
    values = sheet.get_all_values()
    next_row = len(values) + 1 if values else 2

    row_values = [[
        actual_name,
        game_date,
        sportsbook_line,
        sportsbook,
        last_update,
        round(predicted_points, 2),
        "",
        "",
        model_pick,
        "",
        "",
        "",
        edge,
        "PENDING",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        captured_at,
    ]]

    sheet.update(range_name=f"A{next_row}:X{next_row}", values=row_values)
    clear_app_caches()

    return {
        "player_name": actual_name,
        "game_date": game_date,
        "sportsbook": sportsbook,
        "sportsbook_line": sportsbook_line,
        "predicted_points": round(predicted_points, 2),
        "edge": edge,
        "model_pick": model_pick,
        "sheet_row": next_row,
    }
