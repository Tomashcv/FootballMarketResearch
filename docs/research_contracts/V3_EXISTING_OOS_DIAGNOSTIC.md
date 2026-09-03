# V3 Existing OOS Diagnostic

This report uses only already out-of-sample yearly V3 predictions. It does not retrain the model or claim a confirmed edge.

## Main finding
- Predictive improvement exists across the combined sample: delta log loss -0.001470, delta Brier -0.000854, delta ECE -0.005221 on 15786 rows.
- The historical/current fixed away rule has 1237 bets, 57.43u profit, ROI 4.64%, z 1.670.
- Therefore the bottleneck is no longer only prediction. It is converting a small, consistently better probability forecast into a stable decision rule without overfitting thresholds.

## Predictive performance by season
```
 season_start_year  rows  market_log_loss  model_log_loss  delta_log_loss  market_brier  model_brier  delta_brier  market_ece  model_ece  delta_ece
              2021  3323         0.967329        0.966541       -0.000787      0.574403     0.573917    -0.000485    0.018269   0.020312   0.002044
              2022  3252         0.956680        0.954951       -0.001729      0.567050     0.566004    -0.001047    0.029485   0.023637  -0.005849
              2023  3122         0.941126        0.939474       -0.001652      0.556907     0.555973    -0.000934    0.026291   0.020560  -0.005731
              2024  3063         0.952701        0.951161       -0.001540      0.564505     0.563571    -0.000934    0.025672   0.013584  -0.012088
              2025  3026         0.964289        0.962606       -0.001683      0.573241     0.572353    -0.000888    0.026812   0.024513  -0.002299
```

## Predictive performance by league
```
league  rows  market_log_loss  model_log_loss  delta_log_loss  market_brier  model_brier  delta_brier  market_ece  model_ece  delta_ece
   SC0    52         0.841423        0.836600       -0.004823      0.489713     0.487779    -0.001934    0.169024   0.175743   0.006719
    P1  1530         0.913026        0.909800       -0.003226      0.537140     0.535368    -0.001773    0.048102   0.040632  -0.007469
    T1  1819         0.958623        0.955437       -0.003186      0.568158     0.566186    -0.001972    0.054243   0.050519  -0.003725
    I1  1899         0.972439        0.970609       -0.001830      0.578998     0.577781    -0.001217    0.020063   0.014329  -0.005733
    G1   657         0.869042        0.867396       -0.001646      0.508097     0.507472    -0.000625    0.082695   0.062048  -0.020648
   SP1  1900         0.966383        0.964848       -0.001535      0.573947     0.573030    -0.000917    0.033072   0.028110  -0.004962
    F1  1749         0.982664        0.981481       -0.001183      0.585486     0.584789    -0.000697    0.015525   0.011511  -0.004014
    N1  1530         0.940498        0.939509       -0.000990      0.556730     0.556355    -0.000374    0.029686   0.028564  -0.001122
    B1  1175         0.971974        0.971009       -0.000966      0.576556     0.575796    -0.000760    0.041560   0.041589   0.000029
    E0  1900         0.960384        0.960187       -0.000197      0.570158     0.570109    -0.000049    0.020824   0.022001   0.001177
    D1  1575         0.976000        0.976217        0.000217      0.580308     0.580412     0.000104    0.034854   0.033267  -0.001587
```

## Fixed rule by season
```
 season_start_year  bets  profit       roi   z_score  average_odds  average_edge
              2021   319   13.37  0.041912  0.688885      2.164828      0.016485
              2022   337   -4.31 -0.012789 -0.251670      1.920682      0.016276
              2023   115   13.52  0.117565  1.333119      1.890783      0.020496
              2024   218   33.57  0.153991  2.523871      1.858624      0.021399
              2025   248    1.28  0.005161  0.084041      1.931976      0.024146
```

## Fixed rule by league
```
league  bets  profit       roi   z_score  average_odds  average_edge
    I1   285   34.67  0.121649  2.105291      1.982667      0.019925
    F1   143   14.52  0.101538  1.186017      1.992657      0.018537
    P1    95   14.09  0.148316  1.625152      1.865579      0.018893
    D1    70    5.38  0.076857  0.624601      1.954857      0.019286
   SP1    86    1.92  0.022326  0.180607      2.293953      0.016365
    B1    83    1.08  0.013012  0.119668      1.982169      0.016487
   SC0     6   -1.00 -0.166667 -0.446130      1.725000      0.016983
    N1    92   -1.66 -0.018043 -0.187871      1.849239      0.019876
    E0   129   -1.94 -0.015039 -0.181835      1.954031      0.018187
    T1   201   -4.11 -0.020448 -0.300931      1.977960      0.021488
    G1    47   -5.52 -0.117447 -0.904320      1.778723      0.019321
```

## Prior-OOS-only threshold meta-test
```
 test_year                 selected_rule  edge_min  odds_min  odds_max  validation_bets  validation_profit  validation_roi  validation_z_score  validation_average_odds  validation_average_edge  validation_lcb_mean_profit  validation_positive_years  validation_positive_leagues  validation_max_positive_league_share  test_bets  test_profit  test_roi  test_z_score  test_average_odds  test_average_edge  test_lcb_mean_profit  test_positive_years  test_positive_leagues  test_max_positive_league_share
      2023  away_edge_0.02_odds_1.5_None     0.020       1.5       NaN              150              11.44        0.076267            1.020870                 1.840667                 0.023885                    0.001559                          2                            5                              0.368769         58        -3.70 -0.063793     -0.522187           1.852414           0.023246             -0.185958                    0                      4                        0.529412
      2024  away_edge_0.01_odds_1.5_2.25     0.010       1.5      2.25              707              30.24        0.042772            1.265421                 1.825064                 0.016917                    0.008971                          3                            7                              0.384868        341        38.58  0.113138      2.345131           1.820938           0.017721              0.064894                    1                      6                        0.309054
      2025 away_edge_0.015_odds_1.5_2.25     0.015       1.5      2.25              598              57.90        0.096823            2.668256                 1.816455                 0.020867                    0.060536                          3                            9                              0.242722        217        -1.15 -0.005300     -0.083900           1.863088           0.024311             -0.068464                    0                      5                        0.632974
```

## Interpretation
- Simple post-hoc odds caps/threshold changes are not a reliable solution: rules that look strong on prior OOS seasons can still fail in the next season.
- The next research should focus on model shrinkage/calibration, recency weighting, feature-block ablation and uncertainty-aware selection, all nested before the test season.
- League-specific models should not be promoted merely because I1/F1/P1 were profitable; league-specific predictive and value gates must be passed prospectively.
