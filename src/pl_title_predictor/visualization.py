from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_title_probability_chart(results: pd.DataFrame, output_path: str = "output/title_probabilities.png") -> Path:
    top = results.head(10).copy()
    top = top.sort_values("title_probability", ascending=True)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index, top["title_probability"] * 100)
    ax.set_xlabel("Title probability (%)")
    ax.set_ylabel("Team")
    ax.set_title("Premier League title probabilities")

    for i, value in enumerate(top["title_probability"] * 100):
        ax.text(value + 0.5, i, f"{value:.1f}%", va="center")

    plt.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
