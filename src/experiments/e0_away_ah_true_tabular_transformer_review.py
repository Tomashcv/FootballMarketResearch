from pathlib import Path

from src.experiments import e0_away_ah_advanced_tabular_neural_review as review


review.REPORT_PATH = Path("outputs/reports/e0_away_ah_true_tabular_transformer_review.md")
review.SUMMARY_PATH = Path("outputs/reports/e0_away_ah_true_tabular_transformer_summary.csv")
review.DETAIL_DIR = Path("outputs/E0/asian_handicap_big_home_favorite_away/true_tabular_transformer_review")
review.REPORT_TITLE = "E0 Away AH True Tabular Transformer Review"
review.REPORT_CONTEXT = (
    "Torch is installed, so this run compares the existing E0 rule candidates with logistic regression, "
    "the NumPy dropout MLP fallback from the prior run, and a true small PyTorch FT-Transformer-style tabular model."
)


if __name__ == "__main__":
    review.main()
