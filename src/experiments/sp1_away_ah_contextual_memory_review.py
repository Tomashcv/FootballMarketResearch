from pathlib import Path

from src.experiments import i1_away_ah_contextual_memory_review as review


review.LEAGUE = "SP1"
review.LEAGUE_NAME = "La Liga"
review.REPORT_PATH = Path("outputs/reports/sp1_away_ah_contextual_memory_review.md")
review.SUMMARY_PATH = Path("outputs/reports/sp1_away_ah_contextual_memory_summary.csv")
review.DETAIL_DIR = Path("outputs/SP1/asian_handicap_big_home_favorite_away/contextual_memory_review")


if __name__ == "__main__":
    review.main()
