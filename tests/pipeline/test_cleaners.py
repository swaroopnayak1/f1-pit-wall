"""Unit tests for BaseCleaner and all cleaner implementations."""
from pathlib import Path

import pandas as pd
import pytest

from pipeline.cleaner.base import BaseCleaner
from pipeline.cleaner.driver_info import DriverInfoCleaner
from pipeline.cleaner.laps import LapsCleaner
from pipeline.cleaner.session_info import SessionInfoCleaner
from pipeline.cleaner.session_results import SessionResultsCleaner
from pipeline.cleaner.weather import WeatherCleaner


# --------------------------------------------------------------------------- #
# table_name
# --------------------------------------------------------------------------- #
class TestTableName:
    def test_session_info_table_name(self):
        assert SessionInfoCleaner.__name__  # sanity: class accessible
        # Instantiating needs a session; check via the class directly via instance
        # (use the session_info_session fixture implicitly through a helper)

    def test_session_info_table_name_on_instance(self, session_info_session):
        cleaner = SessionInfoCleaner(session_info_session, 2023, 1, "R")
        assert cleaner.table_name == "session_info"

    def test_driver_info_table_name_on_instance(self, driver_info_session):
        cleaner = DriverInfoCleaner(driver_info_session, 2023, 1, "R")
        assert cleaner.table_name == "driver_info"

    def test_session_results_table_name_on_instance(self, session_results_session):
        cleaner = SessionResultsCleaner(session_results_session, 2023, 1, "R")
        assert cleaner.table_name == "session_results"

    def test_laps_table_name_on_instance(self, laps_session):
        cleaner = LapsCleaner(laps_session, 2023, 1, "R")
        assert cleaner.table_name == "laps"

    def test_weather_table_name_on_instance(self, weather_session):
        cleaner = WeatherCleaner(weather_session, 2023, 1, "R")
        assert cleaner.table_name == "weather"


# --------------------------------------------------------------------------- #
# SessionInfoCleaner.clean()
# --------------------------------------------------------------------------- #
class TestSessionInfoClean:
    def test_returns_dataframe(self, session_info_session):
        df = SessionInfoCleaner(session_info_session, 2023, 1, "R").clean()
        assert isinstance(df, pd.DataFrame)

    def test_has_one_row(self, session_info_session):
        df = SessionInfoCleaner(session_info_session, 2023, 1, "R").clean()
        assert len(df) == 1

    def test_contains_expected_columns(self, session_info_session):
        df = SessionInfoCleaner(session_info_session, 2023, 1, "R").clean()
        for col in ("Meeting.Name", "Meeting.Country.Name", "Type", "StartDate"):
            assert col in df.columns, f"missing column: {col}"

    def test_partition_keys_injected(self, session_info_session):
        df = SessionInfoCleaner(session_info_session, 2023, 1, "R").clean()
        assert df["year"].iloc[0] == 2023
        assert df["round_number"].iloc[0] == 1
        assert df["session_type"].iloc[0] == "R"

    def test_partition_key_dtypes(self, session_info_session):
        df = SessionInfoCleaner(session_info_session, 2023, 1, "R").clean()
        assert df["year"].dtype == "int32"
        assert df["round_number"].dtype == "int8"

    def test_start_end_date_are_datetime(self, session_info_session):
        df = SessionInfoCleaner(session_info_session, 2023, 1, "R").clean()
        assert pd.api.types.is_datetime64_any_dtype(df["StartDate"])
        assert pd.api.types.is_datetime64_any_dtype(df["EndDate"])

    def test_string_columns_have_string_dtype(self, session_info_session):
        df = SessionInfoCleaner(session_info_session, 2023, 1, "R").clean()
        assert df["Meeting.Name"].dtype == pd.StringDtype()
        assert df["Type"].dtype == pd.StringDtype()

    def test_missing_optional_columns_tolerated(self):
        """clean() should not raise when optional columns are absent."""
        from types import SimpleNamespace
        sess = SimpleNamespace(
            session_info={
                "Meeting": {"Name": "Bahrain Grand Prix"},
                "Type": "Race",
                "StartDate": "2023-03-05T15:00:00",
                "EndDate": "2023-03-05T17:00:00",
                # GmtOffset and Circuit.ShortName absent
            }
        )
        df = SessionInfoCleaner(sess, 2023, 1, "R").clean()
        assert "Type" in df.columns
        assert "GmtOffset" not in df.columns


# --------------------------------------------------------------------------- #
# DriverInfoCleaner.clean()
# --------------------------------------------------------------------------- #
class TestDriverInfoClean:
    def test_returns_dataframe(self, driver_info_session):
        df = DriverInfoCleaner(driver_info_session, 2023, 1, "R").clean()
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_driver_count(self, driver_info_session):
        df = DriverInfoCleaner(driver_info_session, 2023, 1, "R").clean()
        assert len(df) == len(driver_info_session.drivers)

    def test_contains_expected_columns(self, driver_info_session):
        df = DriverInfoCleaner(driver_info_session, 2023, 1, "R").clean()
        for col in ("DriverNumber", "FullName", "Abbreviation", "TeamName"):
            assert col in df.columns, f"missing column: {col}"

    def test_partition_keys_injected(self, driver_info_session):
        df = DriverInfoCleaner(driver_info_session, 2023, 3, "Q").clean()
        assert all(df["year"] == 2023)
        assert all(df["round_number"] == 3)
        assert all(df["session_type"] == "Q")

    def test_driver_number_is_numeric(self, driver_info_session):
        df = DriverInfoCleaner(driver_info_session, 2023, 1, "R").clean()
        assert pd.api.types.is_integer_dtype(df["DriverNumber"])

    def test_string_columns_have_string_dtype(self, driver_info_session):
        df = DriverInfoCleaner(driver_info_session, 2023, 1, "R").clean()
        assert df["FullName"].dtype == pd.StringDtype()
        assert df["TeamName"].dtype == pd.StringDtype()


# --------------------------------------------------------------------------- #
# BaseCleaner.run()
# --------------------------------------------------------------------------- #
class TestBaseCleanerRun:
    def test_creates_output_directory(self, session_info_session, tmp_path):
        out = tmp_path / "year=2023" / "round=01" / "session=R"
        SessionInfoCleaner(session_info_session, 2023, 1, "R").run(out)
        assert out.is_dir()

    def test_writes_parquet_file(self, session_info_session, tmp_path):
        path = SessionInfoCleaner(session_info_session, 2023, 1, "R").run(tmp_path)
        assert path.exists()
        assert path.suffix == ".parquet"

    def test_parquet_filename_matches_table_name(self, session_info_session, tmp_path):
        path = SessionInfoCleaner(session_info_session, 2023, 1, "R").run(tmp_path)
        assert path.name == "session_info.parquet"

    def test_parquet_readable_and_matches_clean_output(self, session_info_session, tmp_path):
        cleaner = SessionInfoCleaner(session_info_session, 2023, 1, "R")
        path = cleaner.run(tmp_path)
        written = pd.read_parquet(path)
        expected = cleaner.clean()
        assert list(written.columns) == list(expected.columns)
        assert len(written) == len(expected)

    def test_returns_path_to_written_file(self, session_info_session, tmp_path):
        path = SessionInfoCleaner(session_info_session, 2023, 1, "R").run(tmp_path)
        assert isinstance(path, Path)

    def test_accepts_string_output_dir(self, session_info_session, tmp_path):
        path = SessionInfoCleaner(session_info_session, 2023, 1, "R").run(str(tmp_path))
        assert path.exists()

    def test_parquet_filename_is_session_results(self, session_results_session, tmp_path):
        path = SessionResultsCleaner(session_results_session, 2023, 1, "R").run(tmp_path)
        assert path.name == "session_results.parquet"

    def test_parquet_filename_is_laps(self, laps_session, tmp_path):
        path = LapsCleaner(laps_session, 2023, 1, "R").run(tmp_path)
        assert path.name == "laps.parquet"

    def test_parquet_filename_is_weather(self, weather_session, tmp_path):
        path = WeatherCleaner(weather_session, 2023, 1, "R").run(tmp_path)
        assert path.name == "weather.parquet"


# --------------------------------------------------------------------------- #
# SessionResultsCleaner.clean()
# --------------------------------------------------------------------------- #
class TestSessionResultsClean:
    def test_returns_dataframe(self, session_results_session):
        df = SessionResultsCleaner(session_results_session, 2023, 1, "R").clean()
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_driver_count(self, session_results_session):
        df = SessionResultsCleaner(session_results_session, 2023, 1, "R").clean()
        assert len(df) == len(session_results_session.results)

    def test_contains_expected_columns(self, session_results_session):
        df = SessionResultsCleaner(session_results_session, 2023, 1, "R").clean()
        for col in ("DriverNumber", "Abbreviation", "Position", "ClassifiedPosition",
                    "GridPosition", "Status", "Points", "Time"):
            assert col in df.columns, f"missing column: {col}"

    def test_partition_keys_injected(self, session_results_session):
        df = SessionResultsCleaner(session_results_session, 2023, 5, "R").clean()
        assert all(df["year"] == 2023)
        assert all(df["round_number"] == 5)
        assert all(df["session_type"] == "R")

    def test_timing_columns_converted_to_float_seconds(self, session_results_session):
        df = SessionResultsCleaner(session_results_session, 2023, 1, "R").clean()
        assert pd.api.types.is_float_dtype(df["Time"])
        # winner's race time: 1h30m = 5400.0s
        assert pytest.approx(df.loc[0, "Time"], abs=1e-3) == 5400.0
        # P2 gap: 5.123s
        assert pytest.approx(df.loc[1, "Time"], abs=1e-3) == 5.123

    def test_nat_timing_values_become_nan(self, session_results_session):
        df = SessionResultsCleaner(session_results_session, 2023, 1, "R").clean()
        # All Q1/Q2/Q3 are NaT in the race fixture
        assert pd.isna(df["Q1"]).all()
        assert pd.isna(df["Q2"]).all()
        assert pd.isna(df["Q3"]).all()
        # Retired driver has NaT race time
        assert pd.isna(df.loc[2, "Time"])

    def test_classified_position_is_string_dtype(self, session_results_session):
        df = SessionResultsCleaner(session_results_session, 2023, 1, "R").clean()
        assert df["ClassifiedPosition"].dtype == pd.StringDtype()
        # "R" (retired) must survive without coercion
        assert "R" in df["ClassifiedPosition"].values

    def test_driver_number_is_int8(self, session_results_session):
        df = SessionResultsCleaner(session_results_session, 2023, 1, "R").clean()
        assert df["DriverNumber"].dtype == pd.Int8Dtype()

    def test_points_and_position_are_float32(self, session_results_session):
        df = SessionResultsCleaner(session_results_session, 2023, 1, "R").clean()
        assert df["Points"].dtype == pd.Float32Dtype()
        assert df["Position"].dtype == pd.Float32Dtype()
        assert df["GridPosition"].dtype == pd.Float32Dtype()

    def test_string_columns_have_string_dtype(self, session_results_session):
        df = SessionResultsCleaner(session_results_session, 2023, 1, "R").clean()
        assert df["Abbreviation"].dtype == pd.StringDtype()
        assert df["TeamName"].dtype == pd.StringDtype()
        assert df["Status"].dtype == pd.StringDtype()


# --------------------------------------------------------------------------- #
# LapsCleaner.clean()
# --------------------------------------------------------------------------- #
class TestLapsClean:
    def test_returns_dataframe(self, laps_session):
        df = LapsCleaner(laps_session, 2023, 1, "R").clean()
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_lap_count(self, laps_session):
        df = LapsCleaner(laps_session, 2023, 1, "R").clean()
        assert len(df) == len(laps_session.laps)

    def test_contains_expected_columns(self, laps_session):
        df = LapsCleaner(laps_session, 2023, 1, "R").clean()
        for col in ("Driver", "DriverNumber", "LapNumber", "Stint",
                    "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
                    "Compound", "TyreLife", "IsAccurate"):
            assert col in df.columns, f"missing column: {col}"

    def test_partition_keys_injected(self, laps_session):
        df = LapsCleaner(laps_session, 2023, 3, "Q").clean()
        assert all(df["year"] == 2023)
        assert all(df["round_number"] == 3)
        assert all(df["session_type"] == "Q")

    def test_timedelta_columns_converted_to_float_seconds(self, laps_session):
        df = LapsCleaner(laps_session, 2023, 1, "R").clean()
        assert pd.api.types.is_float_dtype(df["LapTime"])
        # VER lap 1: 0:01:30.123 = 90.123s
        assert pytest.approx(df.loc[0, "LapTime"], abs=1e-3) == 90.123
        # Sector1Time VER lap 1: 30.1s
        assert pytest.approx(df.loc[0, "Sector1Time"], abs=1e-3) == 30.1

    def test_nat_lap_time_becomes_nan(self, laps_session):
        df = LapsCleaner(laps_session, 2023, 1, "R").clean()
        # HAM lap 1 (row index 2) has NaT LapTime
        assert pd.isna(df.loc[2, "LapTime"])
        assert pd.isna(df.loc[2, "Sector1Time"])
        assert pd.isna(df.loc[2, "Sector2Time"])
        assert pd.isna(df.loc[2, "Sector3Time"])

    def test_pit_out_time_nat_becomes_nan(self, laps_session):
        df = LapsCleaner(laps_session, 2023, 1, "R").clean()
        # VER lap 2 has NaT PitOutTime
        assert pd.isna(df.loc[1, "PitOutTime"])
        # VER lap 1 PitOutTime: 5.0s
        assert pytest.approx(df.loc[0, "PitOutTime"], abs=1e-3) == 5.0

    def test_boolean_columns_are_boolean_dtype(self, laps_session):
        df = LapsCleaner(laps_session, 2023, 1, "R").clean()
        assert df["FreshTyre"].dtype == pd.BooleanDtype()
        assert df["IsAccurate"].dtype == pd.BooleanDtype()
        assert df["Deleted"].dtype == pd.BooleanDtype()

    def test_lap_number_is_int16(self, laps_session):
        df = LapsCleaner(laps_session, 2023, 1, "R").clean()
        assert df["LapNumber"].dtype == pd.Int16Dtype()

    def test_stint_is_int8(self, laps_session):
        df = LapsCleaner(laps_session, 2023, 1, "R").clean()
        assert df["Stint"].dtype == pd.Int8Dtype()

    def test_string_columns_have_string_dtype(self, laps_session):
        df = LapsCleaner(laps_session, 2023, 1, "R").clean()
        assert df["Driver"].dtype == pd.StringDtype()
        assert df["Compound"].dtype == pd.StringDtype()
        assert df["Team"].dtype == pd.StringDtype()

    def test_lap_start_date_is_datetime(self, laps_session):
        df = LapsCleaner(laps_session, 2023, 1, "R").clean()
        assert pd.api.types.is_datetime64_any_dtype(df["LapStartDate"])


# --------------------------------------------------------------------------- #
# WeatherCleaner.clean()
# --------------------------------------------------------------------------- #
class TestWeatherClean:
    def test_returns_dataframe(self, weather_session):
        df = WeatherCleaner(weather_session, 2023, 1, "R").clean()
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_sample_count(self, weather_session):
        df = WeatherCleaner(weather_session, 2023, 1, "R").clean()
        assert len(df) == len(weather_session.weather_data)

    def test_contains_expected_columns(self, weather_session):
        df = WeatherCleaner(weather_session, 2023, 1, "R").clean()
        for col in ("Time", "AirTemp", "TrackTemp", "Humidity",
                    "Pressure", "Rainfall", "WindDirection", "WindSpeed"):
            assert col in df.columns, f"missing column: {col}"

    def test_partition_keys_injected(self, weather_session):
        df = WeatherCleaner(weather_session, 2024, 7, "FP1").clean()
        assert all(df["year"] == 2024)
        assert all(df["round_number"] == 7)
        assert all(df["session_type"] == "FP1")

    def test_time_converted_to_float_seconds(self, weather_session):
        df = WeatherCleaner(weather_session, 2023, 1, "R").clean()
        assert pd.api.types.is_float_dtype(df["Time"])
        # 0:00:18 = 18.0s, 0:01:18 = 78.0s
        assert pytest.approx(df.loc[0, "Time"], abs=1e-3) == 18.0
        assert pytest.approx(df.loc[1, "Time"], abs=1e-3) == 78.0

    def test_rainfall_is_boolean_dtype(self, weather_session):
        df = WeatherCleaner(weather_session, 2023, 1, "R").clean()
        assert df["Rainfall"].dtype == pd.BooleanDtype()

    def test_float_columns_are_float32(self, weather_session):
        df = WeatherCleaner(weather_session, 2023, 1, "R").clean()
        for col in ("AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed"):
            assert df[col].dtype == pd.Float32Dtype(), f"{col} should be Float32"

    def test_wind_direction_is_int16(self, weather_session):
        df = WeatherCleaner(weather_session, 2023, 1, "R").clean()
        assert df["WindDirection"].dtype == pd.Int16Dtype()
