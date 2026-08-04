# Feature Registry

Canonical, column-level reference for everything that passes through feature engineering: where each column comes from, whether it's safe to use pre-race, and whether it made the locked `FINAL_FEATURES` list actually shipped to the model. Companion to the [README](../README.md#stage-2--eda-and-feature-engineering) (how features are built for *training*) and [`inference_pipeline.md`](inference_pipeline.md) (assembling the same features for a single **unplayed** race). The tables below are the source of truth — `reports/eda_fe.ipynb` (Section 8) and the README's "Final feature set" table summarize the same information for their own narratives and should agree with this file, not the other way around.

## Leakage Registry

Every column reachable from the 5 raw Parquet tables (`session_info`, `driver_info`, `session_results`, `laps`, `weather`), classified before it's allowed anywhere near the model. "Pre-race available?" means knowable before lights-out for the race being predicted, not merely present in the historical dataset.

| Column | Source | Pre-race available? | Disposition | Notes |
|---|---|---|---|---|
| `GridPosition` | `session_results` | Yes | **Keep** | Locked at end of qualifying |
| `RacePosition` | `session_results.Position` | No — target | **Target** | What the model predicts |
| `RaceTime` | `session_results.Time` | No | **Drop** | Known only after the race ends |
| `ClassifiedPosition` | `session_results` | No | **Drop** | Post-race classification |
| `Points` | `session_results` | No | **Drop** | Derived from finish position |
| `Laps` | `session_results` | No | **Drop** | Total laps completed — post-race |
| `Status` | `session_results` | No | **Drop** | Finished/DNF/DSQ etc. — post-race |
| `LapTime_mean` | `laps` (aggregated to driver-race) | No | **Drop (use lag)** | Only the *previous* race's value is safe |
| `LapTime_std` | `laps` (aggregated to driver-race) | No | **Drop (use lag)** | Lagged version ships as `LapStd_lag1` |
| `PitCount` | `laps` (aggregated to driver-race) | No | **Drop** | Strategy is only known once the race is run |
| `RainRisk` | `weather` (mean `Rainfall`) | Caveat | **Keep w/ caveat, not yet wired in** | Real telemetry in training has no leakage-free pre-race equivalent — see [Weather section](inference_pipeline.md#weather--forecast-vs-telemetry-mismatch) |
| `TrackTemp` | `weather` | Caveat | **Keep w/ caveat, not yet wired in** | Same caveat as `RainRisk` |
| `Humidity` | `weather` | Caveat | **Keep w/ caveat, not yet wired in** | Same caveat as `RainRisk` |
| `Pressure` | `weather` | Caveat | **Keep w/ caveat, not yet wired in** | Same caveat as `RainRisk` |
| `AirTemp` | `weather` | Caveat | **Keep w/ caveat, not yet wired in** | Same caveat as `RainRisk` |
| `WindSpeed` | `weather` | Caveat | **Keep w/ caveat, not yet wired in** | Same caveat as `RainRisk` |
| `TeamName` | `driver_info` (normalised via `TEAM_NAME_MAP`) | Yes | **Keep** | Known at season start, mid-season swaps aside |
| `Meeting.Circuit.ShortName` | `session_info` | Yes | **Keep** | Known at season start |
| `round_number` | Season calendar (partition key) | Yes | **Keep** | Known at season start |
| `DriverFinish_lag1` | Engineered — `RacePosition.shift(1)` per `DriverId` | Yes, if a prior race exists | **Keep** | `NaN` at a driver's debut |
| `DriverFinish_ewm` | Engineered — EWMA(span=5) of shifted `RacePosition` per `DriverId` | Yes, if a prior race exists | **Keep** | Same debut caveat |
| `DriverFinish_roll3_inseason` | Engineered — rolling mean(3) of shifted `RacePosition` per `(DriverId, year)` | Yes, from the driver's 2nd race of the season | **Keep** | Asserted `NaN` at round 1 every season |
| `TeamFinish_ewm` | Engineered — team-race avg `RacePosition`, then EWMA(span=5) shifted, per `TeamName` | Yes, if the team has a prior race | **Keep** | Pre-aggregated to team-race level before shifting so a teammate's target race never leaks in |
| `TeamFinish_roll3_inseason` | Engineered — team-race avg `RacePosition`, then rolling mean(3) shifted, per `(TeamName, year)` | Yes, from the team's 2nd race of the season | **Keep** | Same in-season reset and no-teammate-leak guarantee |
| `LapStd_lag1` | Engineered — `LapTime_std.shift(1)` per `DriverId` | Yes, if the prior race has lap data | **Keep** | `NaN` if the previous race ended before a full lap was timed (e.g. a lap-1 DNF) |

> **Weather caveat**: none of the 6 weather columns are actually joined into `build_features()`'s output today (see the README's [TODO](../README.md#todo)) — they're classified here because `reports/eda_fe.ipynb` audits them as part of the full illustrative frame, not because they reach `FINAL_FEATURES`. Training telemetry (`weather.parquet`) is leakage-free in principle (both it and the target are post-race facts about the same session), but inference has no equivalent without a forecast API, and a forecast's error distribution doesn't match a measurement's — see [`inference_pipeline.md`](inference_pipeline.md#weather--forecast-vs-telemetry-mismatch) for why that blocks adding them.
>
> Everything marked **Drop** above is never materialized by `build_features()` in the first place — it selects only `FINAL_FEATURES` + id columns + target before writing `data/features.parquet`, so these columns are excluded by construction, not by a separate filter step.

## Locked Feature Registry (`FINAL_FEATURES`)

The 10 columns actually fitted on, in the order defined in [`feature_engineering.py`](feature_engineering/feature_engineering.py). All 8 non-categorical features are median-imputed then `StandardScaler`-transformed; `TeamName` and `Meeting.Circuit.ShortName` are `LabelEncoder`-transformed. Both transforms are fit once on the full dataset and persisted to `data/features_preprocessors.pkl` — training uses `.fit_transform`, inference uses `.transform` only (see [`inference_pipeline.md`](inference_pipeline.md)).

| # | Feature | Description | Encoding | Pre-race availability |
|---|---|---|---|---|
| 1 | `GridPosition` | Qualifying grid position | Numeric, scaled | Known once qualifying is finalized |
| 2 | `round_number` | Race number within the season | Numeric, scaled | Known at season start |
| 3 | `TeamName` | Constructor, post-rebrand-normalisation | Categorical, label-encoded | Known at season start |
| 4 | `Meeting.Circuit.ShortName` | Circuit identifier | Categorical, label-encoded | Known at season start |
| 5 | `DriverFinish_lag1` | Driver's previous race finish position | Numeric, scaled | Once the driver has ≥1 prior race |
| 6 | `DriverFinish_ewm` | EWMA (span=5) of the driver's finish positions, cross-season | Numeric, scaled | Once the driver has ≥1 prior race |
| 7 | `TeamFinish_ewm` | EWMA (span=5) of the team's average finish, cross-season | Numeric, scaled | Once the team has ≥1 prior race |
| 8 | `DriverFinish_roll3_inseason` | Rolling 3-race average finish, resets each season | Numeric, scaled | From the driver's 2nd race of the season |
| 9 | `TeamFinish_roll3_inseason` | Rolling 3-race team average finish, resets each season | Numeric, scaled | From the team's 2nd race of the season |
| 10 | `LapStd_lag1` | Lap time consistency (std dev) from the driver's previous race | Numeric, scaled | If the previous race has usable lap data |

**Target**: `RacePosition` (`session_results.Position`) — dropped from the feature set, predicted as a continuous value.

A `NaN` in any of rows 5–10 above (debut driver/team, or a previous race with no usable lap time) is median-imputed at training time. That per-column median is **not** persisted alongside the encoders/scaler in `features_preprocessors.pkl`, so `build_inference_features()` can't reproduce it for a single race — see the "Remaining gap" note in [`inference_pipeline.md`](inference_pipeline.md#known-gaps).

**Not the same as any one model's input**: Random Forest and LightGBM each separately drop 1–2 of these 10 after a VIF/importance check (see the README's Stage 4) for their own reduced fits. This registry documents the full, locked `FINAL_FEATURES` list — the one `build_features()`, `build_inference_features()`, and the Optuna-tuned model (`pipeline/model/inference.ipynb`) all use unreduced.
