# V5 Betfair BASIC price-path research

Final research decision: `v5_betfair_no_price_movement_signal`.

Strict outer test seasons: 2019, 2020, 2021, 2022, 2023, 2024. Each uses historical fit, tuning season, calibration season, then untouched test. Best observed fold: t15m, xgboost.

Aggregate selected-model MAE 0.010508 versus no-movement MAE 0.010489; bootstrap probability of lower MAE 0.020.

A pre-COVID outer-test comparison was not estimable under the minimum-history and nested tuning/calibration rules: early mapped seasons (2015–2016) had only 24 and 9 approved fixtures. This robustness analysis is explicitly skipped rather than weakened.

Targets and features are normalized inverse-LTP probability proxies and last-traded-price movements. They are not executable CLV, available back/lay quotes, verified liquidity, betting profit, or a strategy.

No confirmed edge is claimed. Betfair ADVANCED data validation is required before any strategy work.
