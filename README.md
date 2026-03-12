# Premier League Title Predictor

A small machine learning project that estimates the probability of each team winning the current Premier League season.

The project downloads historical Premier League match results, engineers rolling team-form features, trains a Random Forest classifier to predict match outcomes, and then runs Monte Carlo simulations over the remaining fixtures.

## What it does

- downloads Premier League match data from Football-Data
- creates recent-form and team-strength features
- trains a match outcome model (`Home Win`, `Draw`, `Away Win`)
- simulates the remaining season many times
- estimates title probabilities and average final points
- saves a bar chart of the top title contenders

## Project structure

```text
pl-title-predictor/
├── main.py
├── requirements.txt
├── README.md
├── output/
└── src/
    └── pl_title_predictor/
        ├── __init__.py
        ├── data_loader.py
        ├── features.py
        ├── model.py
        ├── simulator.py
        └── visualization.py
```

## Setup

Create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv venv
venv\Scripts\Activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Run the project

```bash
python main.py
```

## Example output

```text
Validation accuracy: 0.498
Played matches: 291
Remaining fixtures to simulate: 89

Premier League title probabilities:
============================================================
             Pts  GD  avg_points  title_probability
Arsenal       67  37       82.14              0.53
Man City      60  32       79.80              0.30
Liverpool     48   9       72.45              0.08
```

The script also saves a chart to:

```text
output/title_probabilities.png
```

## Notes

- The simulation uses a lightweight score placeholder (`2-1`, `1-1`, `1-2`) to update goal difference.
- Remaining fixtures are inferred from the current season matchups rather than exact calendar dates.
- The model is meant as a portfolio and learning project, not a betting-grade forecasting system.

## Possible next improvements

- replace placeholder scores with a Poisson goal model
- add richer team stats from FBref
- export results to CSV automatically
- add a notebook or Streamlit app for interactive exploration

## Data source

- Football-Data: historical results and current-season EPL data
