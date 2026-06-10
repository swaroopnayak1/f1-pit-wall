"""Fixtures shared across feature-engineering tests."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# ── Shared test data ───────────────────────────────────────────────────────────

DRIVERS = [("ham", 44, "Mercedes"), ("ver", 1, "Red Bull Racing")]
YEARS   = [2024, 2025]
ROUNDS  = [1, 2, 3]

# Deterministic finish positions: hamilton 1,2,3 per round | verstappen 4,5,6
_FINISH = {
    ("ham", 1): 1.0, ("ham", 2): 2.0, ("ham", 3): 3.0,
    ("ver", 1): 4.0, ("ver", 2): 5.0, ("ver", 3): 6.0,
}


# ── In-memory race frame (already joined, for testing internal functions) ──────

@pytest.fixture
def race_frame() -> pd.DataFrame:
    """Minimal driver × race DataFrame for testing lag feature computation.
    2 drivers × 2 years × 3 rounds = 12 rows. RacePosition and LapTime_std
    already present — mirrors the output of _build_race_frame.
    """
    rows = []
    for year in YEARS:
        for rnd in ROUNDS:
            for driver_id, driver_num, team in DRIVERS:
                rows.append({
                    "year":                      year,
                    "round_number":              rnd,
                    "session_type":              "R",
                    "DriverId":                  driver_id,
                    "DriverNumber":              driver_num,
                    "TeamName":                  team,
                    "RacePosition":              _FINISH[(driver_id, rnd)],
                    "LapTime_std":               2.0 + rnd * 0.1,
                    "GridPosition":              float(rnd),
                    "Meeting.Circuit.ShortName": f"Circuit{rnd}",
                })
    df = pd.DataFrame(rows)
    df["year"]         = df["year"].astype("int32")
    df["round_number"] = df["round_number"].astype("int8")
    df["DriverId"]     = df["DriverId"].astype("string")
    df["TeamName"]     = df["TeamName"].astype("string")
    return df


# ── Hive-partitioned parquet fixture (for testing disk I/O functions) ──────────

def _write_session_parquets(root: Path, year: int, rnd: int) -> None:
    sd = root / f"year={year}" / f"round={rnd:02d}" / "session=R"
    sd.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "year":                    pd.array([year], dtype="int32"),
        "round_number":            pd.array([rnd],  dtype="int8"),
        "session_type":            ["R"],
        "Meeting.Circuit.ShortName": pd.array([f"Circuit{rnd}"], dtype="string"),
    }).to_parquet(sd / "session_info.parquet", index=False)

    di = pd.DataFrame([
        {"year": year, "round_number": rnd, "session_type": "R",
         "DriverNumber": 44, "DriverId": "ham", "TeamName": "Mercedes"},
        {"year": year, "round_number": rnd, "session_type": "R",
         "DriverNumber": 1,  "DriverId": "ver", "TeamName": "Red Bull Racing"},
    ])
    di["year"]         = di["year"].astype("int32")
    di["round_number"] = di["round_number"].astype("int8")
    di["DriverNumber"] = di["DriverNumber"].astype("Int8")
    di["DriverId"]     = di["DriverId"].astype("string")
    di["TeamName"]     = di["TeamName"].astype("string")
    di.to_parquet(sd / "driver_info.parquet", index=False)

    sr = pd.DataFrame([
        {"year": year, "round_number": rnd, "session_type": "R",
         "DriverNumber": 44, "Position": _FINISH[("ham", rnd)], "GridPosition": float(rnd)},
        {"year": year, "round_number": rnd, "session_type": "R",
         "DriverNumber": 1,  "Position": _FINISH[("ver", rnd)], "GridPosition": float(rnd + 1)},
    ])
    sr["year"]         = sr["year"].astype("int32")
    sr["round_number"] = sr["round_number"].astype("int8")
    sr["DriverNumber"] = sr["DriverNumber"].astype("Int8")
    sr["Position"]     = sr["Position"].astype("Float32")
    sr["GridPosition"] = sr["GridPosition"].astype("Float32")
    sr.to_parquet(sd / "session_results.parquet", index=False)

    # Two laps per driver — gives a non-trivial LapTime_std
    laps = pd.DataFrame([
        {"year": year, "round_number": rnd, "session_type": "R",
         "DriverNumber": drv, "LapTime": lap_t, "PitInTime": None if lap_t == 90.0 else 100.0}
        for drv in [44, 1]
        for lap_t in [90.0, 91.0]
    ])
    laps["year"]         = laps["year"].astype("int32")
    laps["round_number"] = laps["round_number"].astype("int8")
    laps["DriverNumber"] = laps["DriverNumber"].astype("Int8")
    laps.to_parquet(sd / "laps.parquet", index=False)


@pytest.fixture
def data_root(tmp_path) -> Path:
    """Write minimal hive-partitioned parquet files for 2 years × 3 rounds."""
    for year in YEARS:
        for rnd in ROUNDS:
            _write_session_parquets(tmp_path, year, rnd)
    return tmp_path
