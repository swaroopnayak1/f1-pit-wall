# Pre-Race Inference Pipeline

Companion doc to the [README](../README.md#stage-2--eda-and-feature-engineering) — that document covers how features are engineered for *training* (all races, target known). This one covers assembling features for a **single race that hasn't happened yet** (target unknown), and what currently blocks doing that end to end. For the column-level leakage classification and per-feature encoding/availability that the assembly order below relies on, see [`feature_registry.md`](feature_registry.md).

**Status: implemented as a sample.** [`pipeline/model/inference.ipynb`](model/inference.ipynb) demonstrates this end to end for one race — persists the Optuna-tuned LightGBM to `pipeline/model/lightgbm_model.joblib`, and `build_inference_features()` in [`feature_engineering.py`](feature_engineering/feature_engineering.py) is the transform-only assembly function this doc calls for below. See [Known gaps](#known-gaps) for what's still unresolved (weather, pre-quali `GridPosition`, imputation medians).

## Pre-race feature assembly order

For one target race (`year`, `round_number`) and its driver/team lineup, in the order the values become knowable:

1. **Identify the target race and lineup** — `year`, `round_number`, and the driver/team entry list. This comes from `driver_info` + `session_info` for that round; note the race's own `session_results` doesn't fully exist until the race is run, so only the columns below are usable pre-race.
2. **Static, known-in-advance fields** — `round_number`, `TeamName` (post team-name-normalisation, see `TEAM_NAME_MAP` in [`feature_engineering.py`](feature_engineering/feature_engineering.py)), `Meeting.Circuit.ShortName`.
3. **`GridPosition`** — becomes known once qualifying is finalized (same day or day before the race), not 1 hour before, but well before any weather forecast would need to be locked. Sourced from `session_results.GridPosition` for the race session.
4. **Historical lag/EWM/rolling features** — computed the same way as training (`_add_lag_features`), using only *already-completed* races for that driver/team, sorted by `(DriverId, year, round_number)` with `shift(1)`:
   - `DriverFinish_lag1`, `DriverFinish_ewm` (cross-season, per `DriverId`)
   - `DriverFinish_roll3_inseason` (per `(DriverId, year)`, resets each season)
   - `TeamFinish_ewm`, `TeamFinish_roll3_inseason` (pre-aggregated to team-race level before shifting, so a teammate's target race never leaks in)
   - `LapStd_lag1` (from the driver's previous race's `LapTime_std`)

   These require the target race's row to be *appended* to that driver/team's history before the `shift(1)` window is computed — it contributes no value of its own (since `RacePosition` is unknown) but its position in the sort order is what lets the previous race's shifted value land on it.
5. **Weather** — not currently part of `FINAL_FEATURES` at all (see [TODO in README](../README.md#todo)). Once integrated, this is the step where training and inference diverge structurally: training reads *actual* session telemetry from `weather.parquet`; inference must substitute a forecast. See [below](#weather--forecast-vs-telemetry-mismatch).
6. **Apply saved preprocessors** — label-encode `TeamName` / `Meeting.Circuit.ShortName` and scale the numeric features using the **already-fitted** `LabelEncoder`/`StandardScaler` from `data/features_preprocessors.pkl`, *not* refit on the single target race. This is a real gap today — see [Known gaps](#known-gaps).
7. **Assemble rows** — one row per driver, columns in `FINAL_FEATURES` order.
8. **Load the persisted model and predict.**

## Feature availability map (T-1hr before race start)

| Feature | Available at T-1hr? | Source | Notes |
|---|---|---|---|
| `GridPosition` | Yes | `session_results.GridPosition` | Locked at end of qualifying, well before T-1hr |
| `round_number` | Yes | Season calendar | Known at season start |
| `TeamName` | Yes | `driver_info` | Known at season start (mid-season driver swaps aside) |
| `Meeting.Circuit.ShortName` | Yes | `session_info` | Known at season start |
| `DriverFinish_lag1` | Yes | Prior race's `session_results` | Requires driver to have a prior race; else `NaN` → median-imputed (round 1 / debuts) |
| `DriverFinish_ewm` | Yes | Prior races, cross-season | Same debut caveat as above |
| `TeamFinish_ewm` | Yes | Prior races, team-aggregated | Same caveat if it's a new team entry |
| `DriverFinish_roll3_inseason` | Yes | Prior races, current season only | `NaN` until the driver has raced at least once this season (asserted `NaN` at round 1 in training) |
| `TeamFinish_roll3_inseason` | Yes | Prior races, current season only | Same in-season reset as above |
| `LapStd_lag1` | Yes | Prior race's `laps.parquet` | Requires a prior race with lap data |
| Weather (`RainRisk`, `TrackTemp`, `Humidity`, `Pressure`, `AirTemp`, `WindSpeed`) | **No — not real telemetry** | N/A today; would need a forecast API | Session telemetry for the target race doesn't exist until the race happens; training's values (actual `weather.parquet`) have no pre-race equivalent |

Everything currently in `FINAL_FEATURES` is available well before T-1hr — the assembly order above is really gated by *qualifying* finishing (step 3), not by a 1-hour cutoff. The 1-hour boundary only becomes the binding constraint once weather is added, since a forecast issued further out is less reliable than one issued close to lights-out.

## Weather — forecast vs. telemetry mismatch

Not yet implemented; flagged here so the gap is explicit rather than silently deferred.

- **Training** uses real session telemetry from `weather.parquet` (mean `RainRisk`, `TrackTemp`, `Humidity`, `Pressure`, `AirTemp`, `WindSpeed` per race). This is leakage-free during training because both the weather reading and the target (`RacePosition`) are post-race facts about the same completed session.
- **Inference** cannot use this — the race hasn't happened, so there is no telemetry to read. It would need a forecast (e.g. Tomorrow.io, per the README's Future Plans) queried close to race start.
- **The mismatch**: a forecast is a *prediction* (e.g. probability of rain, expected temperature range) with its own error distribution, while the training feature is a *measurement*. A model trained on measured `Rainfall` and fed a forecast probability at inference is being handed a differently-distributed input for the same column name — this needs to be reconciled (e.g. training on forecast-equivalent quantities, or explicitly modeling forecast uncertainty) before weather is safe to add to `FINAL_FEATURES`.
- Until this is resolved, weather stays out of `FINAL_FEATURES` for both training and inference, per the existing README TODO.

## Known gaps

- ~~No persisted model artifact.~~ **Resolved** — `pipeline/model/inference.ipynb` fits the Optuna-tuned LightGBM (`reports/lightgbm_best_params.json`) on `train.parquet` and calls `joblib.dump`, persisting to `pipeline/model/lightgbm_model.joblib` the first time it's run (later runs just `joblib.load` it).
- ~~`build_features()` can't be reused as-is for a single unplayed race.~~ **Resolved** — `build_inference_features()` in `feature_engineering.py` handles this: it keeps the target race's row instead of dropping it for a NaN `RacePosition`, and calls `.transform` (never `.fit_transform`) using the saved `LabelEncoder`/`StandardScaler` from `data/features_preprocessors.pkl`.
  - Remaining gap: imputation medians aren't part of that saved preprocessors pickle, only the encoders/scaler. A driver with no usable prior-race history (debut, or a DNF-on-lap-1 with no lap time for `LapStd_lag1`) has no median to fill with. `build_inference_features(..., strict=False)` surfaces this as a warning and leaves the cell NaN (fine for LightGBM, which splits on NaN natively) rather than either crashing the whole race or silently imputing something training never sanctioned.
- **No pre-quali `GridPosition` estimate.** The assembly order above treats `GridPosition` as known, which is only true after qualifying. Predicting *before* qualifying (if ever needed) would require a separate grid-position estimate, out of scope here.
- **Weather absent from `FINAL_FEATURES`** — see above; both the training join and the inference forecast substitute are unbuilt.
- **`train.parquet`'s split doesn't exclude the target race by itself.** The split only holds out year 2025 as test; every other season — including an in-progress one — is "train" in full. `build_inference_features()` masks the target race out of its *own* lag/rolling computation, but does nothing about the model-fitting side: whoever fits a model on `train.parquet` must separately exclude the target race's own row(s) (and anything from the same season at or after it) or the model will have already seen the real answer. `pipeline/model/inference.ipynb` Section 3 does this before fitting, but it's the *caller's* responsibility, not something `build_inference_features()` or `train.parquet` itself guards against.
