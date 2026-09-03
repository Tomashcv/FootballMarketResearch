# Asian Handicap Big Home Favourite Strategy - Multi-League Summary

## Hypothesis

When the home team is a big favourite on the Asian Handicap line, the away handicap may be undervalued.

The tested side is:

- Bet Away Asian Handicap
- Only when the home AH line is strongly negative
- Threshold selected by nested validation

## Official Candidate

### E0 - Premier League

Main line nested backtest:

- Bets: 261
- Profit: +25.525 units
- ROI: +9.78%
- Z-score: 1.79
- Max drawdown: 8.99 units
- Test years: 2022, 2023, 2024, 2025 all positive

Closing line also stayed positive:

- Bets: 105
- Profit: +10.65 units
- ROI: +10.14%
- Z-score: 1.19

Status: official paper-trading candidate.

## Other Leagues

### SP1 - La Liga

Overall positive, but unstable.

- Main ROI: around +10%
- Closing ROI: around +10%
- Several negative test years, including a poor 2025
- Not robust enough for official paper trading

Status: shadow only.

### D1 - Bundesliga

No valid threshold under strict validation.

Status: rejected.

### I1 - Serie A

No valid threshold under strict validation.

Status: rejected.

### F1 - Ligue 1

Closing line showed a weak positive result:

- Bets: 308
- ROI: +4.29%
- Z-score: 0.87
- Several years close to breakeven

Status: shadow only.

## Final Decision

Use only E0 / Premier League for paper trading.

Do not use real money.

The next step is to monitor live upcoming E0 Asian Handicap odds, add candidates to the paper ledger as watching, and only confirm paper bets closer to kickoff.
