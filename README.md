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
seaborn
pyarrow
pytest
nbconvert
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
└── features.parquet          # model-ready feature matrix (all years combined)
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
│   └── feature_engineering/
│       └── feature_engineering.py  # build_features(), run_feature_engineering(), CLI
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
├── reports/                  # EDA and feature engineering notebooks
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

## License

MIT — see [LICENSE](LICENSE).
