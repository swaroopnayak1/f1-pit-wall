# F1 Pit Wall

A Formula 1 data pipeline that fetches session data via [FastF1](https://github.com/theOehrly/Fast-F1), cleans and flattens it into Hive-partitioned Parquet files, and engineers a model-ready feature matrix for finish-position prediction.

## Features

- **Two load modes** — `ml` (laps + weather, no telemetry) and `viz` (full telemetry) to control cost
- **Live or offline** — hit the network to populate the FastF1 cache, or run entirely cache-only
- **Registry-driven cleaners** — add new table types by registering a cleaner subclass; no orchestrator changes needed
- **Hive partitioning** — output lands at `data/year={Y}/round={RR}/session={TYPE}/{table}.parquet` for efficient slice queries
- **Graceful degradation** — unavailable sessions are skipped without aborting the full season run
- **Feature engineering** — builds a driver × race feature matrix with lag/rolling features and writes `data/features.parquet`

## Requirements

```
fastf1
matplotlib
numpy
pandas
scikit-learn
seaborn
pyarrow
pytest
nbconvert
lightgbm
shap
statsmodels
optuna
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

Use the `venv`
```bash
call C:\Users\<user_name>\anaconda3\Scripts\activate.bat <env_name>
```
Replace the `<user_name>` and `<env_name>` as per your configuration.

### Full pipeline (ingestion + feature engineering)

```bash
# Single year (defaults to ml mode)
python -m pipeline.pipeline 2024

# Multiple years
python -m pipeline.pipeline 2022 2023 2024

# Year range shorthand
python -m pipeline.pipeline 2021-2024

# Visualization mode (includes telemetry — much heavier)
python -m pipeline.pipeline 2024 --mode viz

# Offline mode — serve from cache only, no network calls
python -m pipeline.pipeline 2024 --offline

# Custom output directory
python -m pipeline.pipeline 2024 --out /path/to/output
```

Note: You may have to run the commands several times if the data is being collected from the FastF1 servers due to the rate limits.

### Ingestion only

```bash
python -m pipeline.pipeline 2024 --module data

# Year range, offline cache
python -m pipeline.pipeline 2021-2024 --module data --offline

# Visualization mode (includes telemetry)
python -m pipeline.pipeline 2024 --module data --mode viz
```

### Feature engineering only

Run against already-written Parquet files without re-fetching from FastF1:

```bash
python -m pipeline.pipeline 2024 --module fe

# Custom output directory
python -m pipeline.pipeline 2024 --module fe --out /path/to/data

# Tune hyperparameters (via the feature_engineering module directly)
python -m pipeline.feature_engineering --ewm-span 8 --roll-window 5
```

## Output Structure

```
data/
├── year=2024/
│   └── round=01/
│       └── session=R/
│           ├── session_info.parquet
│           ├── driver_info.parquet
│           ├── session_results.parquet
│           ├── laps.parquet
│           └── weather.parquet
├── features.parquet              # full preprocessed feature matrix (all years)
├── train.parquet                 # training split (all seasons before test year)
├── test.parquet                  # held-out test split (test year, e.g. 2025)
└── features_preprocessors.pkl   # fitted LabelEncoders + StandardScaler
```

Session types follow FastF1 conventions: `FP1`, `FP2`, `FP3`, `Q`, `SQ`, `S`, `R`.

Each Parquet file includes injected partition key columns (`year`, `round_number`, `session_type`) so files can be read independently without path parsing.

### Tables

| Table | Rows | Key columns |
|---|---|---|
| `session_info` | 1 per session | `Meeting.Name`, `Meeting.Circuit.ShortName`, `StartDate`, `EndDate`, `GmtOffset` |
| `driver_info` | 1 per driver | `DriverNumber`, `Abbreviation`, `FullName`, `TeamName`, `CountryCode` |
| `session_results` | 1 per driver | `Position`, `ClassifiedPosition`, `GridPosition`, `Points`, `Status`, `Time`, `Q1`/`Q2`/`Q3`, `Laps` |
| `laps` | 1 per lap | `LapNumber`, `LapTime`, `Sector1-3Time`, `Compound`, `TyreLife`, `Stint`, `SpeedI1`/`I2`/`FL`/`ST`, `IsAccurate`, `Deleted` |
| `weather` | 1 per sample (~1 min intervals) | `Time`, `AirTemp`, `TrackTemp`, `Humidity`, `Pressure`, `Rainfall`, `WindDirection`, `WindSpeed` |

**Timing columns** (`LapTime`, `Sector1-3Time`, `PitOutTime`, `PitInTime`, `Q1`/`Q2`/`Q3`, `Time`) are stored as **float64 seconds** — Parquet has no timedelta type, and seconds are directly usable as model features.

For model training, filter `laps` to `IsAccurate == True` to exclude in/out laps and laps with deleted times.

All files use Snappy compression.

## Project Structure

```
f1-pit-wall/
├── pipeline/
│   ├── pipeline.py               # Orchestrator and CLI entry point
│   ├── loader/
│   │   ├── loader.py             # F1SessionLoader, LoadedSession, build_loader()
│   │   └── strategies.py         # LoadStrategy and SessionSource hierarchies
│   ├── cleaner/
│   │   ├── base.py               # BaseCleaner — clean() + Parquet write
│   │   ├── registry.py           # CleanerRegistry
│   │   ├── session_info.py       # SessionInfoCleaner
│   │   ├── driver_info.py        # DriverInfoCleaner
│   │   ├── session_results.py    # SessionResultsCleaner
│   │   ├── laps.py               # LapsCleaner
│   │   └── weather.py            # WeatherCleaner
│   ├── feature_engineering/
│   │   └── feature_engineering.py  # build_features(), run_feature_engineering(), CLI
│   └── model/
│       ├── baseline.ipynb          # GridPosition and DummyRegressor baselines
│       ├── random_forest.ipynb     # RandomForestRegressor, temporal CV + test evaluation
│       ├── lightgbm.ipynb          # LGBMRegressor, temporal CV + test evaluation, Optuna comparison
│       └── tune_lightgbm.py        # Optuna hyperparameter search (CLI)
├── tests/
│   ├── pipeline/
│   │   ├── conftest.py           # Shared mock FastF1 fixtures
│   │   ├── test_cleaners.py      # Per-table cleaner tests
│   │   ├── test_loader.py        # Loader tests
│   │   ├── test_pipeline.py      # Orchestrator tests
│   │   ├── test_registry.py      # Registry tests
│   │   └── test_smoke.py         # Smoke tests
│   └── feature_engineering/
│       ├── conftest.py           # Parquet fixture builder
│       └── test_feature_engineering.py
├── reports/                  # EDA, feature engineering, SHAP, and error analysis notebooks + saved model results
├── sandbox/                  # Jupyter notebooks for ad-hoc exploration
├── .cache/                   # FastF1 cache (git-ignored)
├── data/                     # Pipeline output (git-ignored)
└── requirements.txt
```

## Architecture

The pipeline runs in two sequential stages:

### Stage 1 — Ingestion

Fetches and cleans raw FastF1 data into Hive-partitioned Parquet files. Split into two independent strategy hierarchies:

**Load strategies** control *what* data FastF1 fetches:
- `MLLoadStrategy` — laps + weather only
- `VizLoadStrategy` — laps + telemetry + weather

**Session sources** control *where* data comes from:
- `LiveF1Source` — network fetch, populates cache
- `OfflineF1Source` — cache-only, deterministic

`build_loader(mode, offline)` composes the right pair. The orchestrator in `pipeline.py` iterates every session of each requested season, runs all registered cleaners, and writes one Parquet file per (session, table) pair.

### Stage 2 — EDA and Feature engineering

Reads the race-session (`session=R`) Parquet partitions, audits the data, and builds a flat driver × race feature matrix written to `data/features.parquet`.

#### EDA

1. **Load and join** — reads all 5 Parquet sources for every `session=R` partition and assembles a single frame at the driver × race grain (1 row per driver per race)
   - Weather aggregated to session level (mean `RainRisk`, `TrackTemp`, `Humidity`, `Pressure`, `AirTemp`, `WindSpeed`)
   - Laps aggregated to driver-race level (`LapTime_mean`, `LapTime_std`, `PitCount`)
2. **Team name normalisation** — maps historical constructor names to their current form so rolling features treat rebrands as one continuous entity (e.g. AlphaTauri → Racing Bulls, Alfa Romeo → Kick Sauber, Racing Point → Aston Martin, Renault → Alpine)
3. **Schema and null audit** — reviews dtype, null percentage, and cardinality for every column; `Q1`/`Q2`/`Q3` are 100 % null for race sessions (qualifying times are not in the race partition)
4. **Coverage check** — confirms rounds per season to catch missing partitions before modelling
5. **Target distribution** — `RacePosition` counts and per-year boxplots to verify a balanced ordinal target across seasons
6. **Univariate distributions** — histograms for all numeric features to catch skew, outliers, or degenerate columns
7. **Spearman correlation with target** — ranks all numeric features by |ρ| against `RacePosition`; `GridPosition` is the strongest pre-race signal
8. **Feature × feature correlation heatmap** — flags pairs with |ρ| > 0.85 as potentially multicollinear
9. **Leakage registry** — classifies every column as pre-race (safe), post-race (drop or lag), or target; weather features are kept with the caveat that telemetry is used during training and a forecast API must be substituted at inference

#### Feature engineering

1. Compute lag and rolling features sorted by `(DriverId, year, round_number)`; all shifts use `shift(1)` so no current-race data leaks in
2. Cross-season driver features (`DriverFinish_lag1`, `DriverFinish_ewm`) — grouped by `DriverId` so features carry across season boundaries
3. Within-season driver feature (`DriverFinish_roll3_inseason`) — grouped by `(DriverId, year)` so the window resets at round 1 each year
4. Team features (`TeamFinish_ewm`, `TeamFinish_roll3_inseason`) — pre-aggregated to the team-race level before rolling to avoid cross-driver leakage
5. Season-boundary sanity check: asserts all within-season rolling features are `NaN` at round 1
6. Spearman correlation of each engineered feature against `RacePosition` with significance markers (p < 0.05 / 0.01 / 0.001)
7. **Train / test split on season boundary** — 2024 → train, 2025 → test; year overlap asserted to be empty
8. Lock `FINAL_FEATURES` as the single source of truth imported by the modelling notebook

#### Preprocessing

Applied after feature construction, in this order:

1. **Drop target-null rows** — rows where `RacePosition` is NaN (DNF/DNS/DSQ with no classified position) are dropped before any fitting step
2. **Label-encode categoricals** — `TeamName` and `Meeting.Circuit.ShortName` are encoded to integer codes using `LabelEncoder` (alphabetical class ordering); encoders are fit on the full dataset and saved to the preprocessors artifact
3. **Median imputation** — remaining NaN in numeric features (e.g. `DriverFinish_lag1` at round 1, `LapStd_lag1`) are filled with each column's median; imputation happens on the full dataset before splitting
4. **StandardScaler** — all numeric features are zero-centred and unit-variance scaled; the fitted scaler is saved to the preprocessors artifact

The fitted `LabelEncoder` instances and `StandardScaler` are persisted together to `data/features_preprocessors.pkl` so the same transforms can be reapplied at inference without re-fitting.

#### Final feature set

| Feature | Description |
|---|---|
| `GridPosition` | Qualifying grid position |
| `round_number` | Race number within the season |
| `TeamName` | Constructor (normalised) |
| `Meeting.Circuit.ShortName` | Circuit identifier |
| `DriverFinish_lag1` | Previous race finish position |
| `DriverFinish_ewm` | EWMA of finish positions (span=5, cross-season) |
| `TeamFinish_ewm` | EWMA of team avg finish (span=5, cross-season) |
| `DriverFinish_roll3_inseason` | Rolling 3-race finish avg (within-season) |
| `TeamFinish_roll3_inseason` | Rolling 3-race team avg (within-season) |
| `LapStd_lag1` | Lap time consistency from previous race |

**Target**: `RacePosition` (finish position)

The EWMA span and rolling window are treated as hyperparameters and can be overridden via CLI flags or the `build_features()` API.

## Models

### Stage 3 — Baseline Models

Establishes lower-bound benchmarks in `pipeline/model/baseline.ipynb` that every subsequent model must beat. Loads `data/train.parquet` and `data/test.parquet` produced by the feature engineering pipeline.

#### Metrics

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — average position error in race positions |
| **Spearman ρ** | Rank-order correlation between predicted and actual finishing order |
| **Macro F1** | Continuous predictions rounded to the nearest position (1–20); F1 averaged equally over all 20 classes regardless of frequency |

#### Baselines

| Model | Strategy | Rationale |
|---|---|---|
| **GridPosition** | Rule-based: predict finish = grid position | Strong real-world heuristic in F1; sets a meaningful performance bar |
| **DummyRegressor** | Always predict training-set mean position | Statistical floor; establishes minimum MAE from zero-signal prediction |

#### Results (test set)

| Model | MAE | Spearman ρ | Macro F1 |
|---|---|---|---|
| GridPosition | 10.44 | 0.652 | 0.005 |
| DummyRegressor | 4.99 | 0.000 | 0.005 |

A trained model must beat **both** baselines on **all three** metrics to be considered useful. Note that DummyRegressor achieves a lower MAE (predicts near the middle of the 1–20 range) but has no ranking or classification power — Spearman ρ and Macro F1 near zero confirm this.

### Stage 4 — Random Forest and LightGBM

Trains a `RandomForestRegressor` (`pipeline/model/random_forest.ipynb`) and an `LGBMRegressor` (`pipeline/model/lightgbm.ipynb`), both with default hyperparameters (only `random_state=42` fixed for reproducibility), starting from the same `FINAL_FEATURES` and metrics as the baseline notebook for direct comparison.

Both are evaluated with **temporal cross-validation** — an expanding-window walk-forward split over the 6 training seasons (2019–2024), so every fold trains only on seasons strictly before its validation season, mirroring real deployment (predict an upcoming season from past ones only):

| Fold | Train years | Validation year |
|---|---|---|
| 1 | 2019 | 2020 |
| 2 | 2019–2020 | 2021 |
| 3 | 2019–2021 | 2022 |
| 4 | 2019–2022 | 2023 |
| 5 | 2019–2023 | 2024 |

Each notebook also refits on the full 2019–2024 training set, scores once on the held-out 2025 test set, and computes native feature importance (Gini/impurity-based for Random Forest, split-count-based for LightGBM — not directly comparable across the two models).

**Feature selection (VIF + importance)**: after the first-pass fit on all 10 `FINAL_FEATURES`, each notebook computes Variance Inflation Factor (VIF) on the 8 continuous/rolling features (the 2 label-encoded categoricals are excluded — a linear-redundancy statistic isn't meaningful for an arbitrary numeric encoding) and drops a feature only if it's *both* near-zero importance (impurity/split-count share below the 10% uniform baseline) *and* highly collinear (VIF > 5). Both models flag the same 4 EWM/rolling-mean feature pairs as collinear (VIF 18–25), but differ on what to drop because their importance rankings differ:
- **Random Forest** drops `TeamFinish_roll3_inseason` and `DriverFinish_roll3_inseason` (8 features retained) — their `_ewm` counterparts carry far more importance.
- **LightGBM** drops only `DriverFinish_roll3_inseason` (9 features retained) — `TeamFinish_roll3_inseason` importance (11.3%) stays just above the near-zero bar.

Each notebook reruns temporal CV and the 2025 test evaluation on its reduced feature set and confirms no regression against the all-features run (deltas are within each model's fold-to-fold CV std). CV results are saved to `reports/random_forest_cv_results.csv` / `reports/lightgbm_cv_results.csv`; test-set comparisons (cumulative, including baselines) to `reports/random_forest_test_results.csv` / `reports/lightgbm_test_results.csv` — both reflect the reduced feature sets, not the diagnostic all-10-feature fit.

#### Results (test set, 2025, reduced feature sets)

| Model | MAE | Spearman ρ | Macro F1 |
|---|---|---|---|
| GridPosition | 10.44 | 0.652 | 0.005 |
| DummyRegressor | 4.99 | 0.000 | 0.005 |
| RandomForest (8 features) | 3.52 | 0.614 | 0.080 |
| LightGBM (9 features) | 3.54 | 0.598 | 0.092 |

Both models clear both baselines on MAE and Macro F1, and stay close to `GridPosition` on Spearman ρ. Neither is tuned — these are first-pass, default-hyperparameter numbers.

#### Takeaways

- **Overfitting is severe for Random Forest with default params**: train MAE is 1.26 vs. 3.54 (CV mean) and 3.53 (test, all-features) — a ~2.3-point gap, with train Spearman ρ 0.97 vs. CV 0.61. Unbounded tree depth lets the forest memorize training seasons; `max_depth`, `min_samples_leaf`, and `max_features` are the highest-priority regularization targets.
- **LightGBM overfits less out of the box**: train MAE is 2.11 vs. 3.64 (CV mean) and 3.52 (test, all-features) — a ~1.5-point gap, narrower than Random Forest's. Boosted, shallower trees generalize somewhat better by default, though `num_leaves`, `min_child_samples`, and early stopping on `n_estimators` are still worth tuning.
- **Feature importance concentration differs sharply between the two models**: for Random Forest, `GridPosition` alone accounts for ~30% of impurity-based importance, followed by `TeamFinish_ewm` (~23%) and `DriverFinish_ewm` (~11%). For LightGBM, split-count importance is more evenly spread — `DriverFinish_ewm`, `TeamFinish_ewm`, and `LapStd_lag1` lead, while `GridPosition` ranks only 6th of 10 — LightGBM's boosting spreads splits across engineered rolling/EWM form features rather than concentrating on qualifying position.
- **Trimming collinear, near-zero-importance features costs nothing**: dropping the redundant `_roll3_inseason` feature(s) moves CV MAE by ≤0.01 and test MAE by ≤0.02 for both models — well inside CV fold-to-fold std (~0.24 for Random Forest, ~0.31 for LightGBM) — so the smaller, less redundant feature set is kept as the model actually saved.

### Stage 5 — Optuna Hyperparameter Tuning

`pipeline/model/tune_lightgbm.py` runs an Optuna search over LightGBM hyperparameters (TPE sampler, `MedianPruner` with 2 warm-up steps), minimizing mean MAE across the same 5-fold expanding-window temporal CV as Stage 4. Only `train.parquet` (2019–2024) is loaded during the search — the 2025 test season is never read, and trials are ranked purely on out-of-sample CV MAE.

```bash
python pipeline/model/tune_lightgbm.py --n-trials 50

# Resumable study backed by persistent storage
python pipeline/model/tune_lightgbm.py --n-trials 100 --storage sqlite:///reports/lightgbm_optuna.db
```

Search space (widened from an initial pass after best trials pinned at the original floors — cross-checked against `optuna_integration.lightgbm.LightGBMTuner`'s default ranges):

| Param | Range |
|---|---|
| `n_estimators` | 50–1000 |
| `num_leaves` | 7–63 |
| `max_depth` | 2–8 |
| `learning_rate` | 0.003–0.3 (log scale) |
| `min_child_samples` | 5–50 |
| `reg_alpha`, `reg_lambda` | 1e-8–10.0 (log scale) |
| `subsample`, `colsample_bytree` | 0.4–1.0 |

The 50-trial search lands on a much slower, shallower model than the untuned defaults: `n_estimators=614`, `max_depth=3`, `learning_rate≈0.0084` (full params in `reports/lightgbm_best_params.json`; per-trial log in `reports/lightgbm_optuna_trials.csv`). `lightgbm.ipynb` Section 11a refits this configuration on the full 2019–2024 training set and scores once on 2025:

| Model | MAE | Spearman ρ | Macro F1 |
|---|---|---|---|
| LightGBM (default, all features) | 3.52 | 0.61 | 0.092 |
| LightGBM (Optuna-tuned, all features) | 3.34 | 0.65 | 0.058 |

Saved to `reports/lightgbm_tuned_cv_results.csv` and `reports/lightgbm_tuned_test_results.csv` — kept separate from `lightgbm_cv_results.csv`/`lightgbm_test_results.csv`, which still reflect the untuned, reduced-feature model referenced elsewhere (SHAP analysis, main results table).

#### Takeaways

- **MAE and rank correlation improve, but Macro F1 gets worse**: tuning cuts test MAE from 3.52 to 3.34 and lifts Spearman ρ from 0.61 to 0.65, but Macro F1 drops from 0.092 to 0.058. The objective was MAE only, so the search had no incentive to keep predictions spread across all 20 position classes — the low learning rate likely produces smoother, more conservative predictions clustered in a narrower band, which helps "how far off" but hurts "exact position hit rate."
- Whether that tradeoff is worth taking depends on the downstream use case — it isn't a strict win, so the tuned model is reported alongside the untuned one rather than replacing it as the project's reference LightGBM result.

### Stage 6 — SHAP Analysis

`reports/shap_analysis.ipynb` explains the LightGBM model's predictions using **SHAP (SHapley Additive exPlanations)**, computed with `shap.TreeExplainer` (exact for tree ensembles) on the same held-out 2025 test set used for final evaluation. Unlike native feature importance, SHAP attributes each individual prediction to its input features in units of the target (finishing positions), with a sign — showing not just which features matter, but how and in which direction they push a given prediction.

Views produced, in order: global importance (mean |SHAP| bar plot), a beeswarm plot (per-instance magnitude and direction), waterfall and force plots for the best- and worst-predicted 2025 rows, dependence/scatter plots for the top 3 features by mean |SHAP|, a heatmap across all test instances, and a decision plot comparing the best vs. worst predictions feature-by-feature.

**Note on feature scale**: `FINAL_FEATURES` are post-preprocessing — the 8 continuous features are standardised (zero mean, unit variance) and the 2 categorical features (`TeamName`, `Meeting.Circuit.ShortName`) are label-encoded integers, so plot axes show z-scores/codes rather than raw grid positions or team names.

#### Takeaways

- The mean |SHAP| ranking mostly agrees with, but refines, LightGBM's native split-count importance — SHAP is denominated in actual output impact (finishing positions) rather than split frequency, so features split on often but with small, offsetting effects rank lower here.
- The beeswarm plot exposes direction, not just magnitude: a consistent colour gradient for a feature (e.g. `GridPosition`) indicates a monotonic relationship the model has learned reliably, while a mixed-colour spread indicates the effect depends on interactions with other features.
- Best- vs. worst-prediction waterfall/force/decision plots show qualitatively different shapes — the best prediction typically has a small number of features agreeing on a confident push toward the correct position, while the worst prediction shows large, conflicting contributions, suggesting the model was working from misleading or unusually noisy inputs.
- The heatmap is the most useful single view for spotting systematic subgroups — clusters of test rows with similar SHAP patterns often correspond to real-world groupings (e.g. a team's mid-season form spike) that aren't explicit features in the model.
- **Caveat**: SHAP explains the model's *learned* behaviour, not ground-truth causality — a feature with high SHAP impact is only as trustworthy as the model itself, which still shows a meaningful train/test generalisation gap (see Stage 4 takeaways).

### Stage 7 — Error Analysis

`reports/error_analysis.ipynb` goes beyond the single headline test MAE (Stage 5, 3.34) to check whether that error is spread evenly across the 2025 held-out set or concentrated in specific, actionable slices. **Model**: the Optuna-tuned `LGBMRegressor` (`reports/lightgbm_best_params.json`), refit on the full 2019–2024 training set — the best model by test MAE in `reports/lightgbm_tuned_test_results.csv`.

None of the four segmentation dimensions exist as columns in `data/test.parquet` — `Meeting.Circuit.ShortName` is label-encoded, `round_number` is standardised, there's no rookie/debut flag, and weather was never joined into `FINAL_FEATURES` (see [TODO](#todo)). Each is reconstructed from the saved preprocessors (`data/features_preprocessors.pkl`) and raw partitions before segmenting:

1. **Street vs. permanent circuits** — manually classified from F1 knowledge: Baku, Jeddah, Las Vegas, Miami, Monte Carlo, Singapore, and Melbourne (a temporary street layout, not a purpose-built facility) are "street"; every other 2025 circuit is "permanent."
2. **Rain vs. dry** — `RainRisk` (mean `Rainfall` reading per race) reconstructed directly from the raw `weather.parquet` partitions for each of the 24 2025 rounds. Two thresholds compared: any rain detected (`RainRisk > 0`, 6/24 races) vs. meaningfully wet (`RainRisk > 0.05`, 3/24 races), since the metric is heavily zero-inflated with a few marginal single-digit-percent blips.
3. **Rookie vs. veteran** — proxy: a driver with fewer than 22 prior races (~1 season) in the full 2019–2025 ingested history is tagged "rookie."
4. **Finishing-position tier** — podium (1–3), points (4–10), midfield (11–15), back (16–20), using the untransformed target — no reconstruction needed.

Some slices are small (as few as ~60 rows) — treat magnitudes as directional, not precise.

#### Results (2025 test set, n=479)

| Segment | Group | n | MAE | Bias (pred − actual) |
|---|---|---|---|---|
| Circuit type | Permanent | 339 | 3.43 | −0.075 |
| Circuit type | Street | 140 | 3.13 | −0.071 |
| Rain (`>0`) | Dry | 359 | 3.37 | −0.071 |
| Rain (`>0`) | Rain | 120 | 3.26 | −0.082 |
| Rain (`>0.05`) | Dry | 419 | 3.29 | −0.082 |
| Rain (`>0.05`) | Rain | 60 | 3.75 | −0.013 |
| Experience | Rookie | 115 | 3.17 | −0.540 |
| Experience | Veteran | 364 | 3.40 | +0.074 |
| Position tier | Podium (1–3) | 72 | 3.17 | +3.153 |
| Position tier | Points (4–10) | 168 | 3.05 | +2.703 |
| Position tier | Midfield (11–15) | 120 | 1.59 | −0.379 |
| Position tier | Back (16–20) | 119 | 5.64 | −5.637 |

Full data: `reports/error_analysis_segments.csv`. Sign convention: `signed_error = pred - actual`; positive bias means the model predicted a *worse* (higher-numbered) finish than actually happened.

#### Takeaways

- **Position tier is the dominant error driver — far larger than any domain segment.** Textbook regression to the mean: podium finishers are under-predicted (bias ≈ +3.15) and back-of-grid finishers are massively over-predicted (bias ≈ −5.64). Back-tier MAE (≈5.6) is more than **3x** the midfield tier's MAE (≈1.6), which is itself nearly unbiased. The overall MAE of 3.34 hides this — the model is considerably more trustworthy for midfield predictions than for either extreme.
- **Rookies get a lower MAE but a much larger, more one-sided bias than veterans.** Rookie MAE (≈3.17) is actually *below* veteran MAE (≈3.40), but rookie bias (≈−0.54) is far more negative than veteran bias (≈+0.07) — the model is systematically over-optimistic about rookies. Plausible cause: a rookie's lag/EWM form features are median-imputed or based on very few real races, so the model leans more on `GridPosition`/`TeamFinish_*`, which doesn't capture a typically rougher rookie season.
- **Rain only hurts in genuinely wet races, not any detectable rain.** At the strict `RainRisk > 0.05` threshold, MAE is clearly worse in the rain (≈3.75 vs ≈3.29 dry). At the loose `> 0` threshold — which folds in marginal, barely-wet rounds — that gap nearly vanishes (≈3.26 vs ≈3.37). The effect is real but driven by the truly wet races, diluted by borderline ones; small sample (3–6 races), so directional rather than conclusive.
- **Street vs. permanent circuits shows no meaningful gap** (MAE ≈3.13 street vs ≈3.43 permanent, comparable bias) — contrary to the intuition that chaotic street races (safety cars, walls, Monaco's single-file track) would be harder to predict. `GridPosition` and the team/driver form features apparently transfer just as well to street tracks in this dataset.
- **Practical implication**: position-tier is the split to design around first — a separate correction/calibration step for the front and back of the field would likely help more than tuning for weather, circuit type, or experience.

## Adding a New Table

1. Create a cleaner in `pipeline/cleaner/` that subclasses `BaseCleaner` and implements `table_name` and `clean()`.
2. Register it in `pipeline/cleaner/__init__.py`:
   ```python
   registry.register("my_table", MyTableCleaner)
   ```
3. Add `"my_table"` to `ACTIVE_CLEANERS` in `pipeline/pipeline.py`.

## Running and generating reports

Run the following command with the file name in terminal.
```bash
jupyter nbconvert --to html your_notebook.ipynb
```

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests use mock FastF1 sessions — no network access or cache required.

### Useful pytest flags

| Command | Purpose |
|---|---|
| `pytest tests/ -v` | Verbose output |
| `pytest tests/pipeline/test_cleaners.py -v` | Cleaner tests only |
| `pytest tests/feature_engineering/ -v` | Feature engineering tests only |
| `pytest tests/pipeline/test_registry.py::TestRegister -v` | Single class |
| `pytest tests/ -x` | Stop on first failure |

### Reading results

- `.` / `PASSED` — test passed
- `F` / `FAILED` — assertion failed, stacktrace shown below
- `E` / `ERROR` — setup/teardown error (fixture problem)

## Future Plans
1. Race win prediction
    a. With weather support (forecasting) from Tomorrow.io
2. Pit Strategy prediction
3. Lap-time prediction
4. Live telemetry data visualization
5. F1 chatbot (maybe)

## TODO
- **Weather forecast pipeline + feature integration** — `weather.parquet` is ingested and cleaned per session, but `build_features()` never loads or joins it, so no weather column reaches `FINAL_FEATURES` today. The EDA leakage registry (see [Architecture](#stage-2--eda-and-feature-engineering)) marked `RainRisk`/`TrackTemp`/`Humidity`/`Pressure`/`AirTemp`/`WindSpeed` as "keep w/ caveat": training can use actual session telemetry (no leakage, since both target and weather are post-race), but inference needs a forecast substitute since the race hasn't happened yet. Need to: (1) join session-level weather aggregates into the race frame, (2) build a forecast-fetch step (e.g. Tomorrow.io) for pre-race prediction, and (3) reconcile the train/inference distribution mismatch before adding weather to `FINAL_FEATURES`.

## License

MIT — see [LICENSE](LICENSE).
