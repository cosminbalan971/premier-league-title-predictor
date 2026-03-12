from io import StringIO

import pandas as pd
import requests


class FootballDataLoader:
    """Download Premier League match data from football-data.co.uk."""

    SEASON_URLS = {
        "2022-23": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
        "2023-24": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "2024-25": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "2025-26": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    }

    def fetch(self) -> pd.DataFrame:
        all_matches = []
        print("Downloading Premier League data...")

        for season, url in self.SEASON_URLS.items():
            print(f"  Getting {season}...")
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            df["season"] = season
            all_matches.append(df)
            print(f"  Loaded {len(df)} rows")

        match_data = pd.concat(all_matches, ignore_index=True)
        match_data["Date"] = pd.to_datetime(
            match_data["Date"], dayfirst=True, errors="coerce"
        )
        return match_data.sort_values(["Date", "season"]).reset_index(drop=True)
