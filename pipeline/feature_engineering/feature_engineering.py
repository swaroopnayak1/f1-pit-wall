"""
Feature engineering pipeline.

Reads race-level Parquet partitions produced by the ingestion pipeline and
builds the model-ready feature set for finish-position prediction.

Steps
-----
1. Load session_info, driver_info, session_results for session=R partitions
2. Join into a single driver × race frame
3. Normalise team names across historical rebrands
4. Compute lag / rolling features (sorted to prevent leakage)
5. Label-encode categorical features
6. Impute remaining nulls with column medians
7. Standardise numeric features with StandardScaler
8. Write the final feature matrix + target to a single Parquet file

Output
------
    <output_path>  (default: data/features.parquet)
    Columns: year, round_number, DriverId, DriverNumber, FINAL_FEATURES, TARGET

    <scaler_path>  (default: data/features_preprocessors.pkl)
    Pickle containing {"label_encoders": dict[str, LabelEncoder], "scaler": StandardScaler}
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..loader import parse_years

# ── Grain and join keys ────────────────────────────────────────────────────────

SESSION_FILTER = "R"
SESSION_KEYS   = ["year", "round_number", "session_type"]
DRIVER_KEYS    = SESSION_KEYS + ["DriverNumber"]

# ── Team name normalisation — historical rebrands → current constructor ────────

TEAM_NAME_MAP: dict[str, str] = {
    # Toro Rosso → AlphaTauri → RB → Racing Bulls
    "Scuderia AlphaTauri": "Racing Bulls",
    "AlphaTauri":          "Racing Bulls",
    "RB":                  "Racing Bulls",
    # Sauber → Alfa Romeo → Kick Sauber
    "Alfa Romeo":          "Kick Sauber",
    "Sauber":              "Kick Sauber",
    # Force India → Racing Point → Aston Martin
    "Racing Point":        "Aston Martin",
    "Force India":         "Aston Martin",
    # Renault → Alpine
    "Renault":             "Alpine",
}

# ── Locked feature list — single source of truth for downstream modelling ─────

FINAL_FEATURES: list[str] = [
    # Pre-race, always known
    "GridPosition",
    "round_number",
    # Categorical — label-encoded to int during feature engineering
    "TeamName",
    "Meeting.Circuit.ShortName",
    # Cross-season exponential moving average (decays older data)
    "DriverFinish_lag1",
    "DriverFinish_ewm",
    "TeamFinish_ewm",
    # Within-season rolling — NaN until race 2, reliable from race 4 onwards
    "DriverFinish_roll3_inseason",
    "TeamFinish_roll3_inseason",
    # Consistency proxy from previous race (lap time std, lagged)
    "LapStd_lag1",
]

# Categorical features that are label-encoded to integer codes
CATEGORICAL_FEATURES: list[str] = [
    "TeamName",
    "Meeting.Circuit.ShortName",
]

# Continuous numeric features — imputed with median then standardised
NUMERIC_SCALE_FEATURES: list[str] = [
    "GridPosition",
    "round_number",
    "DriverFinish_lag1",
    "DriverFinish_ewm",
    "TeamFinish_ewm",
    "DriverFinish_roll3_inseason",
    "TeamFinish_roll3_inseason",
    "LapStd_lag1",
]

TARGET = "RacePosition"

# ── Default paths ──────────────────────────────────────────────────────────────

_PIPELINE_ROOT          = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_ROOT       = _PIPELINE_ROOT / "data"
DEFAULT_OUTPUT          = DEFAULT_DATA_ROOT / "features.parquet"
DEFAULT_PREPROCESSORS   = DEFAULT_DATA_ROOT / "features_preprocessors.pkl"


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_sources(
    data_root: Path,
    years: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scan data_root for session=R partitions and concatenate the 4 core tables."""
    all_si, all_di, all_sr, all_laps = [], [], [], []

    session_dirs = sorted(data_root.glob(f"**/session={SESSION_FILTER}"))
    if years is not None:
        year_set = set(years)
        session_dirs = [
            sd for sd in session_dirs
            if int(sd.parent.parent.name.split("=")[1]) in year_set
        ]
    if not session_dirs:
        raise FileNotFoundError(
            f"No session={SESSION_FILTER} partitions found under {data_root}"
            + (f" for years {sorted(years)}" if years is not None else "")
            + ". Run the ingestion pipeline first."
        )

    for sd in session_dirs:
        all_si.append(pq.ParquetFile(sd / "session_info.parquet").read().to_pandas())
        all_di.append(pq.ParquetFile(sd / "driver_info.parquet").read().to_pandas())
        all_sr.append(pq.ParquetFile(sd / "session_results.parquet").read().to_pandas())
        all_laps.append(pq.ParquetFile(sd / "laps.parquet").read().to_pandas())

    return (
        pd.concat(all_si,   ignore_index=True),
        pd.concat(all_di,   ignore_index=True),
        pd.concat(all_sr,   ignore_index=True),
        pd.concat(all_laps, ignore_index=True),
    )


def _aggregate_laps(laps_raw: pd.DataFrame) -> pd.DataFrame:
    """Reduce per-lap data to one row per driver per race."""
    return (
        laps_raw
        .groupby(DRIVER_KEYS)
        .agg(LapTime_std=("LapTime", "std"))
        .reset_index()
    )


def _build_race_frame(
    session_info: pd.DataFrame,
    driver_info: pd.DataFrame,
    session_results: pd.DataFrame,
    laps_agg: pd.DataFrame,
) -> pd.DataFrame:
    """Join all sources into a single driver × race frame."""
    sr = (
        session_results
        .rename(columns={"Position": "RacePosition", "Time": "RaceTime"})
        .drop(columns=["FullName", "TeamName", "Abbreviation"], errors="ignore")
    )
    return (
        sr
        .merge(driver_info,  on=DRIVER_KEYS, how="left")
        .merge(session_info, on=SESSION_KEYS, how="left")
        .merge(laps_agg,     on=DRIVER_KEYS,  how="left")
    )


def _normalise_teams(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TeamName"] = df["TeamName"].replace(TEAM_NAME_MAP)
    return df


def _add_lag_features(df: pd.DataFrame, *, ewm_span: int, roll_window: int) -> pd.DataFrame:
    """
    Compute all lag / rolling features.

    Sort order: (DriverId, year, round_number) — ensures shift(1) within a
    driver group always references the chronologically previous race, crossing
    season boundaries where appropriate.
    """
    df = df.sort_values(["DriverId", "year", "round_number"]).reset_index(drop=True)

    # ── Driver-level, cross-season ─────────────────────────────────────────────
    df["DriverFinish_lag1"] = (
        df.groupby("DriverId")["RacePosition"].shift(1)
    )

    df["DriverFinish_ewm"] = (
        df.groupby("DriverId")["RacePosition"]
        .transform(lambda x: x.shift(1).ewm(span=ewm_span, min_periods=1).mean())
    )

    # ── Driver-level, within-season ────────────────────────────────────────────
    # Group by (DriverId, year) so the season resets at round 1 each year.
    df["DriverFinish_roll3_inseason"] = (
        df.groupby(["DriverId", "year"])["RacePosition"]
        .transform(lambda x: x.shift(1).rolling(roll_window, min_periods=1).mean())
    )

    df["LapStd_lag1"] = (
        df.groupby("DriverId")["LapTime_std"].shift(1)
    )

    # ── Team-level — pre-aggregate to team-race level first ───────────────────
    # Avoids driver-to-driver leakage: averaging both drivers before shifting
    # means Driver A's result never leaks into Driver B's feature via shift(1).
    team_race = (
        df.groupby(["TeamName", "year", "round_number"])["RacePosition"]
        .mean()
        .reset_index(name="TeamAvgPosition")
        .sort_values(["TeamName", "year", "round_number"])
    )

    team_race["TeamFinish_ewm"] = (
        team_race.groupby("TeamName")["TeamAvgPosition"]
        .transform(lambda x: x.shift(1).ewm(span=ewm_span, min_periods=1).mean())
    )

    team_race["TeamFinish_roll3_inseason"] = (
        team_race.groupby(["TeamName", "year"])["TeamAvgPosition"]
        .transform(lambda x: x.shift(1).rolling(roll_window, min_periods=1).mean())
    )

    df = df.merge(
        team_race[["TeamName", "year", "round_number", "TeamFinish_ewm", "TeamFinish_roll3_inseason"]],
        on=["TeamName", "year", "round_number"],
        how="left",
    )

    # ── Sanity checks — no within-season leakage at round 1 ───────────────────
    round1 = df[df["round_number"] == 1]
    assert round1["DriverFinish_roll3_inseason"].isna().all(), (
        "Season boundary leak detected in DriverFinish_roll3_inseason!"
    )
    assert round1["TeamFinish_roll3_inseason"].isna().all(), (
        "Season boundary leak detected in TeamFinish_roll3_inseason!"
    )

    return df


def _encode_categoricals(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode CATEGORICAL_FEATURES columns in place (alphabetical classes)."""
    df = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def _impute_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN in NUMERIC_SCALE_FEATURES with each column's median."""
    df = df.copy()
    for col in NUMERIC_SCALE_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def _scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardise NUMERIC_SCALE_FEATURES (zero mean, unit variance)."""
    df = df.copy()
    scaler = StandardScaler()
    df[NUMERIC_SCALE_FEATURES] = scaler.fit_transform(df[NUMERIC_SCALE_FEATURES])
    return df, scaler


# ── Public API ─────────────────────────────────────────────────────────────────

def build_features(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    *,
    years: list[int] | None = None,
    ewm_span: int = 5,
    roll_window: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """
    Build the full feature matrix from ingestion-pipeline Parquet outputs.

    Parameters
    ----------
    data_root:   Root of the Hive-partitioned Parquet dataset.
    ewm_span:    EWMA span for driver/team exponential smoothing features.
    roll_window: Rolling window size for within-season averaging features.

    Returns
    -------
    (df, preprocessors) where df has columns: year, round_number, DriverId,
    DriverNumber, FINAL_FEATURES, TARGET; and preprocessors is a dict with
    keys "label_encoders" (dict[str, LabelEncoder]) and "scaler" (StandardScaler).
    """
    data_root = Path(data_root)

    print(f"Loading parquet sources from {data_root} ...")
    session_info, driver_info, session_results, laps_raw = _load_sources(data_root, years)

    laps_agg = _aggregate_laps(laps_raw)
    df = _build_race_frame(session_info, driver_info, session_results, laps_agg)
    print(f"Race frame: {df.shape[0]} rows × {df.shape[1]} columns  "
          f"(years: {sorted(df['year'].unique())})")

    df = _normalise_teams(df)
    df = _add_lag_features(df, ewm_span=ewm_span, roll_window=roll_window)
    df, label_encoders = _encode_categoricals(df)

    id_cols = ["year", "round_number", "DriverId", "DriverNumber"]
    keep    = list(dict.fromkeys(id_cols + FINAL_FEATURES + [TARGET]))
    df = df[[c for c in keep if c in df.columns]]

    before = len(df)
    df = df.dropna(subset=[TARGET])
    dropped = before - len(df)
    if dropped:
        print(f"[features] dropped {dropped} rows with NaN {TARGET} (DNF/DNS/DSQ)")

    df = _impute_nulls(df)
    df, scaler = _scale_features(df)
    return df, {"label_encoders": label_encoders, "scaler": scaler}


def run_feature_engineering(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    years: list[int] | None = None,
    ewm_span: int = 5,
    roll_window: int = 3,
    preprocessors_path: str | Path | None = DEFAULT_PREPROCESSORS,
) -> Path:
    """
    Build features and write to a single Parquet file.

    Parameters
    ----------
    data_root:          Root of the Hive-partitioned Parquet dataset.
    output_path:        Destination Parquet file (created with snappy compression).
    ewm_span:           EWMA span (default 5; candidates: 3, 5, 8, 10).
    roll_window:        Rolling window (default 3; candidates: 3, 5, 7).
    preprocessors_path: Destination pickle for label encoders + scaler (pass None
                        to skip). Default: data/features_preprocessors.pkl.

    Returns
    -------
    Path to the written Parquet file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df, preprocessors = build_features(
        data_root, years=years, ewm_span=ewm_span, roll_window=roll_window
    )

    df.to_parquet(output_path, index=False, compression="snappy")
    print(f"[features] wrote {len(df)} rows ({len(FINAL_FEATURES)} features + target) -> {output_path}")

    if preprocessors_path is not None:
        preprocessors_path = Path(preprocessors_path)
        preprocessors_path.parent.mkdir(parents=True, exist_ok=True)
        with open(preprocessors_path, "wb") as fh:
            pickle.dump(preprocessors, fh)
        print(f"[features] wrote preprocessors -> {preprocessors_path}")

    return output_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feature_engineering",
        description="Build feature matrix from ingestion-pipeline Parquet outputs.",
        epilog=(
            "Examples:\n"
            "  python -m pipeline.feature_engineering\n"
            "  python -m pipeline.feature_engineering --out data/features.parquet\n"
            "  python -m pipeline.feature_engineering --ewm-span 8 --roll-window 5\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_ROOT),
        help=f"Parquet data root directory (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT),
        help=f"Output Parquet file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--ewm-span",
        type=int,
        default=5,
        help="EWMA span for driver/team finish smoothing (default: 5; candidates: 3, 5, 8, 10)",
    )
    parser.add_argument(
        "--roll-window",
        type=int,
        default=3,
        help="Rolling window for within-season features (default: 3; candidates: 3, 5, 7)",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=None,
        help="Year(s) or range to process, e.g. 2024 / 2021 2022 / 2021-2024. Default: all years in data-root.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    years: list[int] | None = None
    if args.years is not None:
        try:
            years = parse_years(args.years)
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
    try:
        run_feature_engineering(
            data_root=args.data,
            output_path=args.out,
            years=years,
            ewm_span=args.ewm_span,
            roll_window=args.roll_window,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
