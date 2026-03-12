import numpy as np
import pandas as pd

from .features import clean_completed_matches, make_match_features


def get_current_season_split(match_data: pd.DataFrame):
    current_raw = match_data[match_data["season"] == "2025-26"].copy()

    played = current_raw.dropna(subset=["FTHG", "FTAG"]).copy()
    played = played.rename(
        columns={
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
        }
    )

    played["home_goals"] = played["home_goals"].astype(int)
    played["away_goals"] = played["away_goals"].astype(int)

    played["result"] = np.where(
        played["home_goals"] > played["away_goals"],
        "H",
        np.where(played["home_goals"] < played["away_goals"], "A", "D"),
    )

    teams = sorted(set(played["home_team"]).union(set(played["away_team"])))

    all_fixtures = []
    for home_team in teams:
        for away_team in teams:
            if home_team != away_team:
                all_fixtures.append({"home_team": home_team, "away_team": away_team})

    all_fixtures_df = pd.DataFrame(all_fixtures)
    played_pairs = set(zip(played["home_team"], played["away_team"]))

    unplayed = all_fixtures_df[
        ~all_fixtures_df.apply(
            lambda r: (r["home_team"], r["away_team"]) in played_pairs, axis=1
        )
    ].copy()

    return played.reset_index(drop=True), unplayed.reset_index(drop=True)


def build_current_table(played: pd.DataFrame) -> pd.DataFrame:
    teams = sorted(set(played["home_team"]).union(set(played["away_team"])))
    table = pd.DataFrame(index=teams, columns=["P", "W", "D", "L", "GF", "GA", "GD", "Pts"]).fillna(0)

    for _, row in played.iterrows():
        h, a = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])

        table.loc[h, "P"] += 1
        table.loc[a, "P"] += 1
        table.loc[h, "GF"] += hg
        table.loc[h, "GA"] += ag
        table.loc[a, "GF"] += ag
        table.loc[a, "GA"] += hg

        if hg > ag:
            table.loc[h, "W"] += 1
            table.loc[a, "L"] += 1
            table.loc[h, "Pts"] += 3
        elif hg < ag:
            table.loc[a, "W"] += 1
            table.loc[h, "L"] += 1
            table.loc[a, "Pts"] += 3
        else:
            table.loc[h, "D"] += 1
            table.loc[a, "D"] += 1
            table.loc[h, "Pts"] += 1
            table.loc[a, "Pts"] += 1

    table["GD"] = table["GF"] - table["GA"]
    return table


def simulate_season(match_data: pd.DataFrame, match_model, n_sims: int = 200) -> pd.DataFrame:
    played, unplayed = get_current_season_split(match_data)
    print(f"Played matches: {len(played)}")
    print(f"Remaining fixtures to simulate: {len(unplayed)}")

    current_table = build_current_table(played)
    base_history = clean_completed_matches(match_data).reset_index(drop=True)

    title_counts = {team: 0 for team in current_table.index}
    total_points = {team: 0 for team in current_table.index}

    if len(unplayed) == 0:
        final_table = current_table.sort_values(["Pts", "GD", "GF"], ascending=False)
        champion = final_table.index[0]
        result = current_table.copy()
        result["title_probability"] = 0.0
        result.loc[champion, "title_probability"] = 1.0
        result["avg_points"] = result["Pts"]
        return result.sort_values(["Pts", "GD", "GF"], ascending=False)

    for sim in range(n_sims):
        if sim % 25 == 0:
            print(f"Running simulation {sim + 1}/{n_sims}...")

        sim_table = current_table.copy()

        for _, fixture in unplayed.iterrows():
            home_team = fixture["home_team"]
            away_team = fixture["away_team"]

            X_match = make_match_features(base_history, home_team, away_team)
            probs = match_model.predict_proba(X_match)[0]
            classes = match_model.classes_
            result = np.random.choice(classes, p=probs)

            if result == "H":
                hg, ag = 2, 1
                sim_table.loc[home_team, "Pts"] += 3
                sim_table.loc[home_team, "W"] += 1
                sim_table.loc[away_team, "L"] += 1
            elif result == "A":
                hg, ag = 1, 2
                sim_table.loc[away_team, "Pts"] += 3
                sim_table.loc[away_team, "W"] += 1
                sim_table.loc[home_team, "L"] += 1
            else:
                hg, ag = 1, 1
                sim_table.loc[home_team, "Pts"] += 1
                sim_table.loc[away_team, "Pts"] += 1
                sim_table.loc[home_team, "D"] += 1
                sim_table.loc[away_team, "D"] += 1

            sim_table.loc[home_team, "P"] += 1
            sim_table.loc[away_team, "P"] += 1
            sim_table.loc[home_team, "GF"] += hg
            sim_table.loc[home_team, "GA"] += ag
            sim_table.loc[away_team, "GF"] += ag
            sim_table.loc[away_team, "GA"] += hg
            sim_table.loc[home_team, "GD"] = sim_table.loc[home_team, "GF"] - sim_table.loc[home_team, "GA"]
            sim_table.loc[away_team, "GD"] = sim_table.loc[away_team, "GF"] - sim_table.loc[away_team, "GA"]

        final_table = sim_table.sort_values(["Pts", "GD", "GF"], ascending=False)
        champion = final_table.index[0]
        title_counts[champion] += 1

        for team in sim_table.index:
            total_points[team] += sim_table.loc[team, "Pts"]

    result = current_table.copy()
    result["avg_points"] = [total_points[t] / n_sims for t in result.index]
    result["title_probability"] = [title_counts[t] / n_sims for t in result.index]

    return result.sort_values(["title_probability", "avg_points", "GD"], ascending=False)
