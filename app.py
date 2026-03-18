import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import json
from scipy.stats import norm


from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, commonplayerinfo, scoreboardv2

# Build player lookup lists once
all_players = players.get_players()
player_name_map = {p["full_name"]: p["id"] for p in all_players}
player_names = list(player_name_map.keys())


st.write("NEW VERSION LOADED 1.3")

st.title("NBA Points Prop Predictor")

model = joblib.load("models/points_regression.pkl")

with open("models/points_model_stats.json", "r") as f:
    model_stats = json.load(f)

points_std = model_stats["std_dev"]

st.caption("Search for a player by name")

selected_player = st.selectbox(
    "Player",
    options=sorted(player_names),
    index=None,
    placeholder="Start typing a player name..."
)

line = st.number_input("Enter points line", min_value=0.0, value=20.5, step=0.5)

if selected_player:
    player_id = player_name_map[selected_player]

    # Current team info
    player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
    team_id = int(player_info.loc[0, "TEAM_ID"])
    team_abbr = player_info.loc[0, "TEAM_ABBREVIATION"]

    # Today's NBA schedule

    now_et = datetime.now(eastern)

    # If it's before 4 AM ET, still consider it "yesterday's slate"
    if now_et.hour < 4:
        now_et = now_et - pd.Timedelta(days=1)
    
    today_str = now_et.strftime("%m/%d/%Y")
    
    board = scoreboardv2.ScoreboardV2(game_date=today_str)
    game_header = board.game_header.get_data_frame()
    line_score = board.line_score.get_data_frame()

    todays_game = game_header[
        (game_header["HOME_TEAM_ID"] == team_id) |
        (game_header["VISITOR_TEAM_ID"] == team_id)
    ]

    st.subheader("Today's Game")

    if todays_game.empty:
        st.info("No game")
    else:
        game = todays_game.iloc[0]
        game_id = game["GAME_ID"]

        game_lines = line_score[
            line_score["GAME_ID"] == game_id
        ][["TEAM_ID", "TEAM_ABBREVIATION"]]

        if int(game["HOME_TEAM_ID"]) == team_id:
            opponent_id = int(game["VISITOR_TEAM_ID"])
            opponent_row = game_lines[game_lines["TEAM_ID"] == opponent_id]
            matchup_text = f"{team_abbr} vs {opponent_row.iloc[0]['TEAM_ABBREVIATION']}"
        else:
            opponent_id = int(game["HOME_TEAM_ID"])
            opponent_row = game_lines[game_lines["TEAM_ID"] == opponent_id]
            matchup_text = f"{team_abbr} @ {opponent_row.iloc[0]['TEAM_ABBREVIATION']}"

        game_date = pd.to_datetime(game["GAME_DATE_EST"]).strftime("%B %d, %Y")
        game_time = game["GAME_STATUS_TEXT"]

        st.write(f"Matchup: {matchup_text}")
        st.write(f"Date: {game_date}")
        st.write(f"Time: {game_time}")

    # Player game logs for model features
    gamelog = playergamelog.PlayerGameLog(
        player_id=player_id,
        season="2025-26"
    )

    df = gamelog.get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # Create Game Score
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

    # Features used by the trained points model
    df["player_avg_pts"] = df["PTS"].shift(1).expanding().mean()
    df["last5_pts"] = df["PTS"].shift(1).rolling(5).mean()
    df["last5_fga"] = df["FGA"].shift(1).rolling(5).mean()
    df["last5_fta"] = df["FTA"].shift(1).rolling(5).mean()
    df["last5_minutes"] = df["MIN"].shift(1).rolling(5).mean()
    df["last5_gmsc"] = df["gmsc"].shift(1).rolling(5).mean()

    df = df.dropna().reset_index(drop=True)

    if df.empty:
        st.warning("Not enough recent games to build features yet.")
    else:
        latest = df.iloc[-1]

        X = pd.DataFrame([{
            "player_avg_pts": latest["player_avg_pts"],
            "last5_pts": latest["last5_pts"],
            "last5_fga": latest["last5_fga"],
            "last5_fta": latest["last5_fta"],
            "last5_minutes": latest["last5_minutes"],
            "last5_gmsc": latest["last5_gmsc"]
        }])

        predicted_points = model.predict(X)[0]
        edge = predicted_points - line

        prob_over = 1 - norm.cdf(line, loc=predicted_points, scale=points_std)
        prob_under = 1 - prob_over

        st.subheader("Prediction")
        st.write("Predicted points:", round(predicted_points, 2))
        st.write("Line:", line)
        st.write("Edge:", round(edge, 2))
        st.write("Probability over:", f"{prob_over:.1%}")
        st.write("Probability under:", f"{prob_under:.1%}")

        if prob_over >= 0.60:
            st.success("Lean Over")
        elif prob_under >= 0.60:
            st.warning("Lean Under")
        else:
            st.info("No Edge")
