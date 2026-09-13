"""
Canonical feature-building logic for the legacy points prediction model.

Extracted from src/shared_app.py::build_player_feature_row() as a pure,
dependency-light (pandas only) function so it can be shared by inference
(src/shared_app.py) and, later, a training pipeline, without duplicating
the formulas in two places.

This is a structural extraction only -- no formulas, fill behavior, or
output semantics were changed from the original implementation.
"""

import pandas as pd

from src.features.feature_schema import (
    LEGACY_CORE_REQUIRED_FEATURES,
    LEGACY_FEATURE_NAMES,
)


def build_player_feature_row(df, player_name, sportsbook_line=None):
    def _parse_minutes_value(val):
        if pd.isna(val):
            return None

        text = str(val).strip()
        if not text:
            return None

        try:
            if ":" in text:
                parts = text.split(":")
                if len(parts) == 2:
                    mins = float(parts[0])
                    secs = float(parts[1])
                    return mins + (secs / 60.0)

            if text.startswith("PT"):
                text = text.replace("PT", "")
                mins = 0.0
                secs = 0.0

                if "M" in text:
                    m_part = text.split("M")[0]
                    mins = float(m_part) if m_part else 0.0
                    text = text.split("M")[1]

                if "S" in text:
                    s_part = text.replace("S", "")
                    secs = float(s_part) if s_part else 0.0

                return mins + (secs / 60.0)

            return float(text)
        except Exception:
            return None

    df = df.copy()
    if df.empty:
        return None

    df["PLAYER_NAME"] = player_name
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE").reset_index(drop=True)
    if df.empty:
        return None

    numeric_cols = [
        "PTS", "FGM", "FGA", "FTA", "FTM", "OREB", "DREB",
        "STL", "AST", "BLK", "PF", "TOV"
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "MIN" not in df.columns:
        df["MIN"] = pd.NA
    df["MIN"] = df["MIN"].apply(_parse_minutes_value)

    if "FG3A" in df.columns:
        df["FG3A"] = pd.to_numeric(df["FG3A"], errors="coerce")
    else:
        df["FG3A"] = pd.NA

    if "MATCHUP" not in df.columns:
        df["MATCHUP"] = ""

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

    grouped = df.groupby("PLAYER_NAME")

    df["player_avg_pts"] = grouped["PTS"].transform(lambda x: x.shift(1).expanding().mean())
    df["player_avg_pts_sq"] = df["player_avg_pts"] ** 2
    df["last3_pts"] = grouped["PTS"].transform(lambda x: x.shift(1).rolling(3).mean())
    df["last5_pts"] = grouped["PTS"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["last10_pts"] = grouped["PTS"].transform(lambda x: x.shift(1).rolling(10).mean())
    df["last20_pts"] = grouped["PTS"].transform(lambda x: x.shift(1).rolling(20).mean())
    df["last5_fga"] = grouped["FGA"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["last5_fta"] = grouped["FTA"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["last5_minutes"] = grouped["MIN"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["last5_gmsc"] = grouped["gmsc"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["home_game"] = df["MATCHUP"].astype(str).str.contains("vs", case=False, na=False).astype(int)
    df["days_rest"] = grouped["GAME_DATE"].diff().dt.days.fillna(3)
    df["is_back_to_back"] = (df["days_rest"] == 1).astype(int)
    df["usage_proxy"] = df["FGA"] + 0.44 * df["FTA"] + df["TOV"]
    df["last5_usage_proxy"] = grouped["usage_proxy"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["season_minutes_avg"] = grouped["MIN"].transform(lambda x: x.shift(1).expanding().mean())
    df["predicted_minutes"] = df["last5_minutes"].combine_first(df["season_minutes_avg"])
    df["minutes_volatility"] = grouped["MIN"].transform(lambda x: x.shift(1).rolling(5).std())
    df["points_volatility"] = grouped["PTS"].transform(lambda x: x.shift(1).rolling(5).std())
    df["opponent"] = df["MATCHUP"].astype(str).str.split().str[-1]

    opp_grouped = df.groupby("opponent")
    df["opp_pts_allowed"] = opp_grouped["PTS"].transform(lambda x: x.shift(1).rolling(10).mean())
    df["opp_pts_allowed_last5"] = opp_grouped["PTS"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["opp_pts_volatility"] = opp_grouped["PTS"].transform(lambda x: x.shift(1).rolling(10).std())

    df["is_star"] = (df["player_avg_pts"] >= 20).astype(int)
    df["closing_line"] = float(sportsbook_line) if sportsbook_line is not None else df["player_avg_pts"]
    df["last5_3pa"] = grouped["FG3A"].transform(lambda x: x.shift(1).rolling(5).mean())

    required_features = list(LEGACY_FEATURE_NAMES)

    df_features = df.copy()

    for col in required_features:
        if col not in df_features.columns:
            df_features[col] = pd.NA

    df_features[required_features] = df_features[required_features].ffill().bfill()

    core_required = list(LEGACY_CORE_REQUIRED_FEATURES)

    df_features = df_features.dropna(subset=core_required).reset_index(drop=True)
    if df_features.empty:
        return None

    latest = df_features.iloc[-1]
    feature_data = {col: latest.get(col) for col in required_features}
    return pd.DataFrame([feature_data])
