import warnings

from src.pl_title_predictor.data_loader import FootballDataLoader
from src.pl_title_predictor.model import MatchOutcomeModel
from src.pl_title_predictor.simulator import simulate_season
from src.pl_title_predictor.visualization import save_title_probability_chart

warnings.filterwarnings("ignore")


def main():
    loader = FootballDataLoader()
    match_data = loader.fetch()

    model = MatchOutcomeModel(random_state=42)
    accuracy = model.train(match_data)
    print(f"Validation accuracy: {accuracy:.3f}")

    title_probs = simulate_season(match_data, model, n_sims=200)

    print("\nPremier League title probabilities:")
    print("=" * 60)
    cols = [c for c in ["Pts", "GD", "avg_points", "title_probability"] if c in title_probs.columns]
    print(title_probs[cols].head(10).to_string())

    chart_path = save_title_probability_chart(title_probs)
    print(f"\nChart saved to: {chart_path}")


if __name__ == "__main__":
    main()
