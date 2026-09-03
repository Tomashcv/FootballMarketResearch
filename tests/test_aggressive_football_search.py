import pandas as pd

from src.experiments.aggressive_football_search import asian_profit
from src.experiments.aggressive_football_search import split_handicap
from src.experiments.aggressive_football_search import summarize_bets


def test_split_handicap_quarter_lines():
    assert split_handicap(1.25) == [1.0, 1.5]
    assert split_handicap(-1.25) == [-1.5, -1.0]
    assert split_handicap(-1.5) == [-1.5]


def test_asian_profit_quarter_line_half_loss():
    # +0.75 splits into +0.5 and +1.0. Losing by 1 loses the +0.5 half and pushes the +1.0 half.
    assert asian_profit(team_margin=-1, handicap=0.75, odds=2.0) == -0.5


def test_asian_profit_quarter_line_half_win():
    assert asian_profit(team_margin=1, handicap=-0.75, odds=2.0) == 0.5


def test_summarize_bets_basic_drawdown():
    dataframe = pd.DataFrame(
        {
            "Date": ["2022-01-01", "2022-01-02", "2023-01-01"],
            "HomeTeam": ["A", "B", "C"],
            "AwayTeam": ["D", "E", "F"],
            "test_year": [2022, 2022, 2023],
            "profit": [1.0, -1.0, 1.0],
        }
    )
    summary = summarize_bets(dataframe)
    assert summary["bets"] == 3
    assert summary["profit"] == 1.0
    assert summary["positive_test_years"] == 1
    assert summary["negative_test_years"] == 1
    assert summary["max_drawdown"] == 1.0
