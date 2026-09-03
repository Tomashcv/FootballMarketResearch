# E0 Asian Handicap Big Home Favourite Baseline

## Strategy

Bet Away Asian Handicap when the home team is a big favourite.

Current live rule candidate:

- League: E0 / Premier League
- Market: Asian Handicap
- Variant: main line
- Side: Away AH
- Condition: AHh <= -1.25
- Stake: 1 unit paper

## Nested Backtest Main Line

- Bets: 261
- Profit: +25.525 units
- ROI: +9.78%
- Z-score: 1.79
- Max drawdown: 8.99 units

## By Test Year

- 2022: +0.515 units, ROI +1.14%
- 2023: +3.920 units, ROI +5.94%
- 2024: +4.550 units, ROI +5.83%
- 2025: +16.540 units, ROI +22.97%

## Robustness Notes

The strategy remained positive after requiring validation years to have positive minimum yearly ROI.

The signal exists both for promoted/new teams and non-new teams:

- Away not new to league: ROI +7.29%
- Away new to league: ROI +14.06%

The signal is not only one exact handicap line. Main positive lines include -1.25, -1.50, -1.75, -2.00 and -2.25.

## Current Status

Candidate for paper trading only.

No real-money betting until live paper results confirm that the edge survives with currently available odds and realistic execution.
