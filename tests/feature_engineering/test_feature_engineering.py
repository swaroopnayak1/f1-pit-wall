"""Tests for pipeline.feature_engineering.feature_engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.feature_engineering.feature_engineering import (
    FINAL_FEATURES,
    TARGET,
    TEAM_NAME_MAP,
    _add_lag_features,
    _aggregate_laps,
    _build_race_frame,
    _load_sources,
    _normalise_teams,
    build_features,
    run_feature_engineering,
)


class TestConstants:
    def test_target_is_race_position(self):
        assert TARGET == "RacePosition"

    def test_final_features_count(self):
        assert len(FINAL_FEATURES) == 10

    def test_final_features_contains_expected_columns(self):
        expected = {
            "GridPosition", "round_number", "TeamName",
            "Meeting.Circuit.ShortName", "DriverFinish_lag1",
            "DriverFinish_ewm", "TeamFinish_ewm",
            "DriverFinish_roll3_inseason", "TeamFinish_roll3_inseason",
            "LapStd_lag1",
        }
        assert set(FINAL_FEATURES) == expected


class TestLoadSources:
    def test_raises_when_no_partitions(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="session=R"):
            _load_sources(tmp_path)

    def test_returns_four_dataframes(self, data_root):
        result = _load_sources(data_root)
        assert len(result) == 4
        assert all(isinstance(df, pd.DataFrame) for df in result)

    def test_session_info_row_count(self, data_root):
        si, *_ = _load_sources(data_root)
        assert len(si) == 6  # 2 years × 3 rounds

    def test_driver_info_row_count(self, data_root):
        _, di, *_ = _load_sources(data_root)
        assert len(di) == 12  # 2 years × 3 rounds × 2 drivers

    def test_session_results_row_count(self, data_root):
        _, _, sr, _ = _load_sources(data_root)
        assert len(sr) == 12

    def test_laps_row_count(self, data_root):
        *_, laps = _load_sources(data_root)
        assert len(laps) == 24  # 2 years × 3 rounds × 2 drivers × 2 laps

    def test_year_filter_restricts_to_requested_year(self, data_root):
        si, di, sr, laps = _load_sources(data_root, years=[2024])
        assert len(si) == 3    # 1 year × 3 rounds
        assert len(di) == 6    # 1 year × 3 rounds × 2 drivers
        assert len(sr) == 6
        assert len(laps) == 12  # 1 year × 3 rounds × 2 drivers × 2 laps

    def test_year_filter_absent_year_raises(self, data_root):
        with pytest.raises(FileNotFoundError, match="session=R"):
            _load_sources(data_root, years=[1999])


class TestAggregateLaps:
    def test_one_row_per_driver_per_session(self, data_root):
        *_, laps_raw = _load_sources(data_root)
        laps_agg = _aggregate_laps(laps_raw)
        assert len(laps_agg) == 12  # 2 years × 3 rounds × 2 drivers

    def test_laptime_std_computed(self, data_root):
        *_, laps_raw = _load_sources(data_root)
        laps_agg = _aggregate_laps(laps_raw)
        # Each driver has laps [90.0, 91.0] → std(ddof=1) = sqrt(0.5)
        expected = np.std([90.0, 91.0], ddof=1)
        assert laps_agg["LapTime_std"].notna().all()
        assert pytest.approx(laps_agg["LapTime_std"].iloc[0], abs=1e-4) == expected

    def test_output_columns(self, data_root):
        *_, laps_raw = _load_sources(data_root)
        laps_agg = _aggregate_laps(laps_raw)
        assert "LapTime_std" in laps_agg.columns
        assert "DriverNumber" in laps_agg.columns


class TestBuildRaceFrame:
    def test_position_renamed_to_race_position(self, data_root):
        si, di, sr, laps_raw = _load_sources(data_root)
        df = _build_race_frame(si, di, sr, _aggregate_laps(laps_raw))
        assert "RacePosition" in df.columns
        assert "Position" not in df.columns

    def test_driver_id_present_from_driver_info(self, data_root):
        si, di, sr, laps_raw = _load_sources(data_root)
        df = _build_race_frame(si, di, sr, _aggregate_laps(laps_raw))
        assert "DriverId" in df.columns
        assert set(df["DriverId"].dropna()) == {"ham", "ver"}

    def test_circuit_name_present_from_session_info(self, data_root):
        si, di, sr, laps_raw = _load_sources(data_root)
        df = _build_race_frame(si, di, sr, _aggregate_laps(laps_raw))
        assert "Meeting.Circuit.ShortName" in df.columns

    def test_laptime_std_present_from_laps(self, data_root):
        si, di, sr, laps_raw = _load_sources(data_root)
        df = _build_race_frame(si, di, sr, _aggregate_laps(laps_raw))
        assert "LapTime_std" in df.columns

    def test_row_count(self, data_root):
        si, di, sr, laps_raw = _load_sources(data_root)
        df = _build_race_frame(si, di, sr, _aggregate_laps(laps_raw))
        assert len(df) == 12  # 2 years × 3 rounds × 2 drivers

    def test_no_rows_dropped_on_left_join(self, data_root):
        si, di, sr, laps_raw = _load_sources(data_root)
        # Even with an empty laps_agg, no rows should be lost
        empty_laps = _aggregate_laps(laps_raw).iloc[0:0]
        df = _build_race_frame(si, di, sr, empty_laps)
        assert len(df) == 12


class TestNormaliseTeams:
    def test_known_aliases_mapped_to_canonical_names(self):
        df = pd.DataFrame({"TeamName": list(TEAM_NAME_MAP.keys())})
        result = _normalise_teams(df)
        assert set(result["TeamName"]).issubset(set(TEAM_NAME_MAP.values()))

    def test_unknown_teams_left_unchanged(self):
        df = pd.DataFrame({"TeamName": ["McLaren", "Ferrari"]})
        result = _normalise_teams(df)
        assert list(result["TeamName"]) == ["McLaren", "Ferrari"]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"TeamName": ["RB"]})
        _normalise_teams(df)
        assert df["TeamName"].iloc[0] == "RB"

    @pytest.mark.parametrize("alias,canonical", [
        ("RB",                  "Racing Bulls"),
        ("AlphaTauri",          "Racing Bulls"),
        ("Scuderia AlphaTauri", "Racing Bulls"),
        ("Alfa Romeo",          "Kick Sauber"),
        ("Sauber",              "Kick Sauber"),
        ("Racing Point",        "Aston Martin"),
        ("Force India",         "Aston Martin"),
        ("Renault",             "Alpine"),
    ])
    def test_specific_alias(self, alias, canonical):
        df = pd.DataFrame({"TeamName": [alias]})
        result = _normalise_teams(df)
        assert result["TeamName"].iloc[0] == canonical


class TestAddLagFeatures:
    def test_all_engineered_columns_present(self, race_frame):
        df = _add_lag_features(race_frame, ewm_span=5, roll_window=3)
        for col in ["DriverFinish_lag1", "DriverFinish_ewm", "LapStd_lag1",
                    "DriverFinish_roll3_inseason", "TeamFinish_ewm", "TeamFinish_roll3_inseason"]:
            assert col in df.columns, f"Missing column: {col}"

    # ── DriverFinish_lag1 ──────────────────────────────────────────────────────

    def test_driver_finish_lag1_nan_at_first_career_race(self, race_frame):
        df = _add_lag_features(race_frame, ewm_span=5, roll_window=3)
        first = df[(df["DriverId"] == "ham") & (df["year"] == 2024) & (df["round_number"] == 1)]
        assert first["DriverFinish_lag1"].isna().all()

    def test_driver_finish_lag1_correct_value(self, race_frame):
        df = _add_lag_features(race_frame, ewm_span=5, roll_window=3)
        row = df[(df["DriverId"] == "ham") & (df["year"] == 2024) & (df["round_number"] == 2)]
        # Hamilton R1 2024 finish = 1.0
        assert pytest.approx(row["DriverFinish_lag1"].iloc[0]) == 1.0

    def test_driver_finish_lag1_crosses_season_boundary(self, race_frame):
        df = _add_lag_features(race_frame, ewm_span=5, roll_window=3)
        row = df[(df["DriverId"] == "ham") & (df["year"] == 2025) & (df["round_number"] == 1)]
        # Hamilton R3 2024 finish = 3.0 → lag1 at R1 2025 = 3.0
        assert pytest.approx(row["DriverFinish_lag1"].iloc[0]) == 3.0

    # ── DriverFinish_ewm ───────────────────────────────────────────────────────

    def test_driver_finish_ewm_nan_at_first_career_race(self, race_frame):
        df = _add_lag_features(race_frame, ewm_span=5, roll_window=3)
        first = df[(df["DriverId"] == "ham") & (df["year"] == 2024) & (df["round_number"] == 1)]
        assert first["DriverFinish_ewm"].isna().all()

    def test_driver_finish_ewm_present_from_second_race(self, race_frame):
        df = _add_lag_features(race_frame, ewm_span=5, roll_window=3)
        row = df[(df["DriverId"] == "ham") & (df["year"] == 2024) & (df["round_number"] == 2)]
        assert row["DriverFinish_ewm"].notna().all()

    # ── DriverFinish_roll3_inseason ────────────────────────────────────────────

    def test_driver_finish_roll3_inseason_nan_at_round1(self, race_frame):
        df = _add_lag_features(race_frame, ewm_span=5, roll_window=3)
        round1 = df[df["round_number"] == 1]
        assert round1["DriverFinish_roll3_inseason"].isna().all()

    def test_driver_finish_roll3_inseason_resets_each_season(self, race_frame):
        df = _add_lag_features(race_frame, ewm_span=5, roll_window=3)
        row = df[(df["DriverId"] == "ham") & (df["year"] == 2025) & (df["round_number"] == 1)]
        assert row["DriverFinish_roll3_inseason"].isna().all()

    def test_driver_finish_roll3_inseason_value_at_round2(self, race_frame):
        df = _add_lag_features(race_frame, ewm_span=5, roll_window=3)
        row = df[(df["DriverId"] == "ham") & (df["year"] == 2024) & (df["round_number"] == 2)]
        # Only R1 (finish=1.0) available, min_periods=1
        assert pytest.approx(row["DriverFinish_roll3_inseason"].iloc[0]) == 1.0

    def test_driver_finish_roll3_inseason_value_at_round3(self, race_frame):
        df = _add_lag_features(race_frame, ewm_span=5, roll_window=3)
        row = df[(df["DriverId"] == "ham") & (df["year"] == 2024) & (df["round_number"] == 3)]
        # R1=1.0, R2=2.0 → mean = 1.5
        assert pytest.approx(row["DriverFinish_roll3_inseason"].iloc[0]) == 1.5

    # ── Team features ──────────────────────────────────────────────────────────

    def test_team_finish_roll3_inseason_nan_at_round1(self, race_frame):
        df = _add_lag_features(race_frame, ewm_span=5, roll_window=3)
        round1 = df[df["round_number"] == 1]
        assert round1["TeamFinish_roll3_inseason"].isna().all()

    def test_team_finish_ewm_nan_at_first_team_appearance(self, race_frame):
        df = _add_lag_features(race_frame, ewm_span=5, roll_window=3)
        first = df[(df["TeamName"] == "Mercedes") & (df["year"] == 2024) & (df["round_number"] == 1)]
        assert first["TeamFinish_ewm"].isna().all()

    def test_team_finish_ewm_present_from_second_race(self, race_frame):
        df = _add_lag_features(race_frame, ewm_span=5, roll_window=3)
        row = df[(df["TeamName"] == "Mercedes") & (df["year"] == 2024) & (df["round_number"] == 2)]
        assert row["TeamFinish_ewm"].notna().all()

    # ── LapStd_lag1 ───────────────────────────────────────────────────────────

    def test_lap_std_lag1_nan_at_first_career_race(self, race_frame):
        df = _add_lag_features(race_frame, ewm_span=5, roll_window=3)
        first = df[(df["DriverId"] == "ham") & (df["year"] == 2024) & (df["round_number"] == 1)]
        assert first["LapStd_lag1"].isna().all()

    def test_lap_std_lag1_correct_value(self, race_frame):
        df = _add_lag_features(race_frame, ewm_span=5, roll_window=3)
        row = df[(df["DriverId"] == "ham") & (df["year"] == 2024) & (df["round_number"] == 2)]
        # race_frame: LapTime_std = 2.0 + round * 0.1, so R1 = 2.1
        assert pytest.approx(row["LapStd_lag1"].iloc[0], abs=1e-6) == 2.1


class TestBuildFeatures:
    def test_returns_dataframe(self, data_root):
        df = build_features(data_root)
        assert isinstance(df, pd.DataFrame)

    def test_contains_all_final_features(self, data_root):
        df = build_features(data_root)
        for col in FINAL_FEATURES:
            assert col in df.columns, f"Missing feature: {col}"

    def test_contains_target(self, data_root):
        df = build_features(data_root)
        assert TARGET in df.columns

    def test_row_count(self, data_root):
        df = build_features(data_root)
        assert len(df) == 12  # 2 years × 3 rounds × 2 drivers

    def test_identifier_columns_present(self, data_root):
        df = build_features(data_root)
        for col in ["year", "round_number", "DriverId", "DriverNumber"]:
            assert col in df.columns

    def test_raises_when_data_root_empty(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_features(tmp_path)

    def test_custom_ewm_span_accepted(self, data_root):
        df = build_features(data_root, ewm_span=3)
        assert "DriverFinish_ewm" in df.columns

    def test_custom_roll_window_accepted(self, data_root):
        df = build_features(data_root, roll_window=5)
        assert "DriverFinish_roll3_inseason" in df.columns

    def test_year_filter_returns_only_requested_year(self, data_root):
        df = build_features(data_root, years=[2024])
        assert set(df["year"].unique()) == {2024}
        assert len(df) == 6  # 1 year × 3 rounds × 2 drivers


class TestRunFeatureEngineering:
    def test_writes_parquet_file(self, data_root, tmp_path):
        out = tmp_path / "features.parquet"
        run_feature_engineering(data_root, out)
        assert out.exists()

    def test_parquet_is_readable(self, data_root, tmp_path):
        out = tmp_path / "features.parquet"
        run_feature_engineering(data_root, out)
        df = pd.read_parquet(out)
        assert len(df) > 0

    def test_parquet_contains_final_features_and_target(self, data_root, tmp_path):
        out = tmp_path / "features.parquet"
        run_feature_engineering(data_root, out)
        df = pd.read_parquet(out)
        for col in FINAL_FEATURES:
            assert col in df.columns, f"Missing feature in parquet: {col}"
        assert TARGET in df.columns

    def test_creates_output_directory(self, data_root, tmp_path):
        out = tmp_path / "nested" / "dir" / "features.parquet"
        run_feature_engineering(data_root, out)
        assert out.exists()

    def test_returns_output_path(self, data_root, tmp_path):
        out = tmp_path / "features.parquet"
        result = run_feature_engineering(data_root, out)
        assert result == out

    def test_year_filter_writes_only_requested_year(self, data_root, tmp_path):
        out = tmp_path / "features.parquet"
        run_feature_engineering(data_root, out, years=[2024])
        df = pd.read_parquet(out)
        assert set(df["year"].unique()) == {2024}
        assert len(df) == 6  # 1 year × 3 rounds × 2 drivers
