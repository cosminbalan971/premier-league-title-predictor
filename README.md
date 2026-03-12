# Premier League Title Predictor

This project uses **machine learning and Monte Carlo simulation** to estimate the probability of each team winning the Premier League.

The model predicts match outcomes based on historical performance and then simulates the remainder of the season thousands of times to estimate title probabilities.

---

## Project Overview

Pipeline:

1. Download historical Premier League match data
2. Engineer team performance features
3. Train a Random Forest match prediction model
4. Simulate the remaining fixtures of the season
5. Estimate title probabilities using Monte Carlo simulation

---

## Data Source

Match results are downloaded from:

https://www.football-data.co.uk/

The dataset contains:

- match results
- teams
- goals scored
- season information

---

## Model

The match prediction model uses:

- Random Forest Classifier
- Rolling team performance statistics
- Recent team form
- Goal scoring averages
- Defensive performance

Validation accuracy:

~50% match outcome prediction accuracy (typical for football ML models).

---

## Season Simulation

The remaining matches of the current season are simulated many times.

89 remaining matches × 1000 simulations
Each simulation produces a final league table. 
Title probabilities are calculated as: title probability = championships / simulations

## Example Output

| Team | Avg Points | Title Probability |
|-----|-----|-----|
| Arsenal | 83.2 | 0.51 |
| Man City | 80.1 | 0.33 |
| Liverpool | 77.0 | 0.12 |

---

## Data source

Football-Data: historical results and current-season EPL data
