import streamlit as st
import pandas as pd
import joblib

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

st.title("NBA Points Prop Predictor")

model = joblib.load("models/points_regression.pkl")

player_name = st.text_input("Enter player name")
line = st.number_input("Enter points line", min_value=0.0, value=20.5, step=0.5)

if player_name:
    player_list = players.find_players_by_full_name(player_name)

    if not player_list:
        st.write("Player not found")
    else:
        player_id = player_list[0]["id"]

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

        st.write("Predicted points:", round(predicted_points, 2))
        st.write("Line:", line)
        st.write("Edge:", round(edge, 2))

        if edge >= 1.0:
            st.success("Lean Over")
        elif edge <= -1.0:
            st.warning("Lean Under")
        else:
            st.info("No Edge")
