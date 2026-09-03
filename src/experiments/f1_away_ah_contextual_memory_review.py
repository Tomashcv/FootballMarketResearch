from pathlib import Path

from src.experiments import i1_away_ah_contextual_memory_review as review


review.LEAGUE = "F1"
review.LEAGUE_NAME = "Ligue 1"
review.REPORT_PATH = Path("outputs/reports/f1_away_ah_contextual_memory_review.md")
review.SUMMARY_PATH = Path("outputs/reports/f1_away_ah_contextual_memory_summary.csv")
review.DETAIL_DIR = Path("outputs/F1/asian_handicap_big_home_favorite_away/contextual_memory_review")


if __name__ == "__main__":
    review.main()
