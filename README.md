# F1 Pit Wall

A Formula 1 data ingestion pipeline that fetches session data via [FastF1](https://github.com/theOehrly/Fast-F1), cleans and flattens it, and writes structured Parquet files in a Hive-style partitioned layout.

## Features

- **Two load modes** — `ml` (laps + weather, no telemetry) and `viz` (full telemetry) to control cost
- **Live or offline** — hit the network to populate the FastF1 cache, or run entirely cache-only
- **Registry-driven cleaners** — add new table types by registering a cleaner subclass; no orchestrator changes needed
- **Hive partitioning** — output lands at `data/year={Y}/round={RR}/session={TYPE}/{table}.parquet` for efficient slice queries
- **Graceful degradation** — unavailable sessions are skipped without aborting the full season run

## Requirements

```
fastf1
matplotlib
numpy
pandas
pyarrow
pytest
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

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

## Output Structure

```
data/
└── year=2024/
    └── round=01/
        └── session=R/
            ├── session_info.parquet
            ├── driver_info.parquet
            ├── session_results.parquet
            ├── laps.parquet
            └── weather.parquet
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
│   ├── pipeline.py          # Orchestrator and CLI entry point
│   ├── loader/
│   │   ├── loader.py        # F1SessionLoader, LoadedSession, build_loader()
│   │   └── strategies.py    # LoadStrategy and SessionSource hierarchies
│   └── cleaner/
│       ├── base.py              # BaseCleaner — clean() + Parquet write
│       ├── registry.py          # CleanerRegistry
│       ├── session_info.py      # SessionInfoCleaner
│       ├── driver_info.py       # DriverInfoCleaner
│       ├── session_results.py   # SessionResultsCleaner
│       ├── laps.py              # LapsCleaner
│       └── weather.py           # WeatherCleaner
├── tests/                   # Unit test suite (pytest, no network calls)
├── sandbox/                 # Jupyter notebooks for ad-hoc exploration
├── .cache/                  # FastF1 cache (git-ignored)
├── data/                    # Pipeline output (git-ignored)
└── requirements.txt
```

## Architecture

The pipeline is split into two independent strategy hierarchies:

**Load strategies** control *what* data FastF1 fetches:
- `MLLoadStrategy` — laps + weather only
- `VizLoadStrategy` — laps + telemetry + weather

**Session sources** control *where* data comes from:
- `LiveF1Source` — network fetch, populates cache
- `OfflineF1Source` — cache-only, deterministic

`build_loader(mode, offline)` composes the right pair. The orchestrator in `pipeline.py` iterates every session of each requested season, runs all registered cleaners, and writes one Parquet file per (session, table) pair.

## Adding a New Table

1. Create a cleaner in `pipeline/cleaner/` that subclasses `BaseCleaner` and implements `table_name` and `clean()`.
2. Register it in `pipeline/cleaner/__init__.py`:
   ```python
   registry.register("my_table", MyTableCleaner)
   ```
3. Add `"my_table"` to `ACTIVE_CLEANERS` in `pipeline/pipeline.py`.

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests use mock FastF1 sessions — no network access or cache required.

### Useful pytest flags

| Command | Purpose |
|---|---|
| `pytest tests/ -v` | Verbose output |
| `pytest tests/test_registry.py -v` | Single file |
| `pytest tests/test_registry.py::TestRegister -v` | Single class |
| `pytest tests/ -x` | Stop on first failure |

### Reading results

- `.` / `PASSED` — test passed
- `F` / `FAILED` — assertion failed, stacktrace shown below
- `E` / `ERROR` — setup/teardown error (fixture problem)

## License

MIT — see [LICENSE](LICENSE).
