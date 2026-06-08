"""Unit tests for BaseCleaner, SessionInfoCleaner, and DriverInfoCleaner."""
from pathlib import Path

import pandas as pd
import pytest

from pipeline.cleaner.base import BaseCleaner
from pipeline.cleaner.driver_info import DriverInfoCleaner
from pipeline.cleaner.session_info import SessionInfoCleaner


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
