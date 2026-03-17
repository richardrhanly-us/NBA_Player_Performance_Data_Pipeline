import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, commonplayerinfo, scoreboardv2

st.write("NEW VERSION LOADED")

st.title("NBA Points Prop Predictor")

model = joblib.load("models/points_regression.pkl")

player_name = st.text_input("Enter player name")
line = st.number_input("Enter points line", min_value=0.0, value=20.5, step=0.5)

if player_name:
    player_list = players.find_players_by_full_name(player_name)

    if not player_list:
        st.error("Player not found")
    else:
        player_id = player_list[0]["id"]

        # Current team info
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
        team_id = int(player_info.loc[0, "TEAM_ID"])
        team_abbr = player_info.loc[0, "TEAM_ABBREVIATION"]

        # Today's NBA schedule
        today_str = datetime.today().strftime("%m/%d/%Y")
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

            game_lines = line_score[line_score["GAME_ID"] == game_id][["TEAM_ID", "TEAM_ABBREVIATION"]]

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

        df["last5_pts"] = df["PTS"].shift(1).rolling(5).mean()
        df["last10_pts"] = df["PTS"].shift(1).rolling(10).mean()
        df["last5_minutes"] = df["MIN"].shift(1).rolling(5).mean()
        df["last5_fg_pct"] = df["FG_PCT"].shift(1).rolling(5).mean()

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

        df["last5_gmsc"] = df["gmsc"].shift(1).rolling(5).mean()
        df["last10_gmsc"] = df["gmsc"].shift(1).rolling(10).mean()

        df = df.dropna().reset_index(drop=True)

        if df.empty:
            st.warning("Not enough recent games to build features yet.")
        else:
            latest = df.iloc[-1]

            X = pd.DataFrame([{
                "last5_pts": latest["last5_pts"],
                "last10_pts": latest["last10_pts"],
                "last5_gmsc": latest["last5_gmsc"],
                "last10_gmsc": latest["last10_gmsc"],
                "last5_minutes": latest["last5_minutes"],
                "last5_fg_pct": latest["last5_fg_pct"]
            }])

            predicted_points = model.predict(X)[0]
            edge = predicted_points - line

            st.subheader("Prediction")
            st.write("Predicted points:", round(predicted_points, 2))
            st.write("Line:", line)
            st.write("Edge:", round(edge, 2))

            if edge >= 1.0:
                st.success("Lean Over")
            elif edge <= -1.0:
                st.warning("Lean Under")
            else:
                st.info("No Edge")
