import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "home_team_strength",
    "away_team_strength",
    "home_recent_form",
    "away_recent_form",
    "home_goals_avg",
    "away_goals_avg",
    "home_goals_conceded_avg",
    "away_goals_conceded_avg",
    "home_advantage",
]


def clean_completed_matches(data: pd.DataFrame) -> pd.DataFrame:
    completed = data.dropna(subset=["FTHG", "FTAG", "HomeTeam", "AwayTeam"]).copy()

    completed["FTHG"] = completed["FTHG"].astype(int)
    completed["FTAG"] = completed["FTAG"].astype(int)

    completed["result"] = np.where(
        completed["FTHG"] > completed["FTAG"],
        "H",
        np.where(completed["FTHG"] < completed["FTAG"], "A", "D"),
    )

    completed = completed.rename(
        columns={
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
        }
    )

    keep_cols = [
        "season",
        "Date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
    ]
    return completed[keep_cols].copy()


def get_team_history(data: pd.DataFrame, team: str, current_index: int, games: int = 5) -> pd.DataFrame:
    team_matches = data[
        (((data["home_team"] == team) | (data["away_team"] == team)) & (data.index < current_index))
    ]
    return team_matches.tail(games)


def calculate_team_stats(history: pd.DataFrame, team: str) -> dict:
    if len(history) == 0:
        return {
            "strength": 50.0,
            "form": 5.0,
            "goals_for": 1.5,
            "goals_against": 1.5,
        }

    goals_scored = 0
    goals_conceded = 0
    points = 0

    for _, match in history.iterrows():
        if match["home_team"] == team:
            goals_scored += match["home_goals"]
            goals_conceded += match["away_goals"]
            if match["result"] == "H":
                points += 3
            elif match["result"] == "D":
                points += 1
        else:
            goals_scored += match["away_goals"]
            goals_conceded += match["home_goals"]
            if match["result"] == "A":
                points += 3
            elif match["result"] == "D":
                points += 1

    num_games = len(history)
    goals_per_game = goals_scored / num_games
    goals_conceded_per_game = goals_conceded / num_games
    strength = (points / num_games) * 20 + 20

    return {
        "strength": float(min(90, max(10, strength))),
        "form": float(points),
        "goals_for": float(goals_per_game),
        "goals_against": float(goals_conceded_per_game),
    }


def add_features(completed_matches: pd.DataFrame) -> pd.DataFrame:
    df = completed_matches.copy().reset_index(drop=True)

    for col in FEATURE_COLUMNS:
        df[col] = np.nan

    for i, match in df.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]

        home_history = get_team_history(df, home_team, i, games=5)
        away_history = get_team_history(df, away_team, i, games=5)

        home_stats = calculate_team_stats(home_history, home_team)
        away_stats = calculate_team_stats(away_history, away_team)

        df.loc[i, "home_team_strength"] = home_stats["strength"]
        df.loc[i, "away_team_strength"] = away_stats["strength"]
        df.loc[i, "home_recent_form"] = home_stats["form"]
        df.loc[i, "away_recent_form"] = away_stats["form"]
        df.loc[i, "home_goals_avg"] = home_stats["goals_for"]
        df.loc[i, "away_goals_avg"] = away_stats["goals_for"]
        df.loc[i, "home_goals_conceded_avg"] = home_stats["goals_against"]
        df.loc[i, "away_goals_conceded_avg"] = away_stats["goals_against"]
        df.loc[i, "home_advantage"] = 1.0

    return df


def make_match_features(history_df: pd.DataFrame, home_team: str, away_team: str) -> pd.DataFrame:
    home_history = history_df[
        (history_df["home_team"] == home_team) | (history_df["away_team"] == home_team)
    ].tail(5)

    away_history = history_df[
        (history_df["home_team"] == away_team) | (history_df["away_team"] == away_team)
    ].tail(5)

    home_stats = calculate_team_stats(home_history, home_team)
    away_stats = calculate_team_stats(away_history, away_team)

    return pd.DataFrame(
        {
            "home_team_strength": [home_stats["strength"]],
            "away_team_strength": [away_stats["strength"]],
            "home_recent_form": [home_stats["form"]],
            "away_recent_form": [away_stats["form"]],
            "home_goals_avg": [home_stats["goals_for"]],
            "away_goals_avg": [away_stats["goals_for"]],
            "home_goals_conceded_avg": [home_stats["goals_against"]],
            "away_goals_conceded_avg": [away_stats["goals_against"]],
            "home_advantage": [1.0],
        }
    )
