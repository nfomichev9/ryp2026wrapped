"""
Train per-player ML models that predict which team a player would pick
given game conditions. Used by the Tab 3 predictor in the dashboard.
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

PLAYERS = [
    "ADon", "Exciting Whites", "Kevin", "MC$", "Maye Magic", "P-Otys",
    "Ripw1124", "Vegas", "Willheser", "Yianni", "b_hop", "derelicious", "mrmcwinnerson",
]

ALL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WAS",
]


def build_features(picks_df, sap_df, teams_df):
    """Build feature matrix for ML training.

    Each row = one player-game pick. Target = did they pick the home team (1) or away (0).
    Features: team one-hot encodings, spread, week, conference matchup, weather.
    """
    # merge picks with game info from scores
    game_info = sap_df[[
        "week", "game", "team_home", "team_away", "platform_spread",
        "weather_temperature", "weather_wind_mph", "weather_humidity", "weather_detail",
    ]].drop_duplicates(subset=["week", "game"])

    merged = picks_df.merge(game_info, on=["week", "game"], how="inner")

    # target: 1 if picked home team, 0 if picked away
    merged["picked_home"] = (merged["pick"] == merged["team_home"]).astype(int)

    # conference/division info
    team_meta = teams_df[["team_id", "team_conference", "team_division"]].drop_duplicates(subset=["team_id"])
    # keep only current teams (non-null division)
    team_meta = team_meta.dropna(subset=["team_division"])
    conf_map = team_meta.set_index("team_id")["team_conference"].to_dict()
    div_map = team_meta.set_index("team_id")["team_division"].to_dict()

    # features
    features = pd.DataFrame()
    features["spread"] = merged["platform_spread"].fillna(0)
    features["spread_abs"] = features["spread"].abs()
    features["week"] = merged["week"]

    # home/away team one-hot
    for team in ALL_TEAMS:
        features[f"home_{team}"] = (merged["team_home"] == team).astype(int)
        features[f"away_{team}"] = (merged["team_away"] == team).astype(int)

    # conference matchup
    merged["home_conf"] = merged["team_home"].map(conf_map)
    merged["away_conf"] = merged["team_away"].map(conf_map)
    features["cross_conference"] = (merged["home_conf"] != merged["away_conf"]).astype(int)

    # weather
    features["indoor"] = merged["weather_detail"].fillna("").str.contains("indoor|retractable", case=False).astype(int)
    features["temp"] = merged["weather_temperature"].fillna(65)
    features["wind"] = merged["weather_wind_mph"].fillna(5)
    features["rain_snow"] = merged["weather_detail"].fillna("").str.contains("rain|snow", case=False).astype(int)

    return features, merged["picked_home"], merged["player"]


def train_models():
    """Train a logistic regression model for each player. Returns dict of models."""
    picks = pd.read_csv("picks.csv")
    sap = pd.read_csv("scores_and_picks.csv")
    teams_df = pd.read_csv("nfl_teams (1).csv")

    # fix LVR -> LV in teams_df for consistency
    teams_df["team_id"] = teams_df["team_id"].replace("LVR", "LV")

    X_all, y_all, players_col = build_features(picks, sap, teams_df)
    feature_names = list(X_all.columns)

    models = {}
    for player in PLAYERS:
        mask = players_col == player
        X_p = X_all[mask]
        y_p = y_all[mask]

        if len(X_p) < 20:
            continue

        model = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
        model.fit(X_p, y_p)
        models[player] = model

    # save models and feature names
    Path("models").mkdir(exist_ok=True)
    with open("models/player_models.pkl", "wb") as f:
        pickle.dump({"models": models, "feature_names": feature_names}, f)

    print(f"Trained {len(models)} player models, saved to models/player_models.pkl")
    return models, feature_names


def load_models():
    """Load pre-trained models."""
    with open("models/player_models.pkl", "rb") as f:
        data = pickle.load(f)
    return data["models"], data["feature_names"]


def predict_pick(model, feature_names, home_team, away_team, spread, week,
                 indoor=False, temp=65, wind=5, rain_snow=False, cross_conf=False):
    """Predict probability a player picks the home team."""
    row = {name: 0 for name in feature_names}
    row["spread"] = spread
    row["spread_abs"] = abs(spread)
    row["week"] = week
    row[f"home_{home_team}"] = 1
    row[f"away_{away_team}"] = 1
    row["cross_conference"] = int(cross_conf)
    row["indoor"] = int(indoor)
    row["temp"] = temp
    row["wind"] = wind
    row["rain_snow"] = int(rain_snow)

    X = pd.DataFrame([row])[feature_names]
    prob_home = model.predict_proba(X)[0][1]
    return prob_home


if __name__ == "__main__":
    train_models()
