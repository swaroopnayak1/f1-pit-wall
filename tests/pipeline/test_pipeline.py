"""Unit tests for pipeline.pipeline (orchestrator logic)."""
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

import pipeline.pipeline as pp
from pipeline.loader.loader import LoadedSession
from pipeline.pipeline import (
    ACTIVE_CLEANERS,
    _session_output_dir,
    run_pipeline,
    split_train_test,
)


# --------------------------------------------------------------------------- #
# _session_output_dir (pure function)
# --------------------------------------------------------------------------- #
class TestSessionOutputDir:
    def test_hive_style_path(self, tmp_path):
        result = _session_output_dir(tmp_path, 2023, 1, "R")
        assert result == tmp_path / "year=2023" / "round=01" / "session=R"

    def test_round_zero_padded_to_two_digits(self, tmp_path):
        result = _session_output_dir(tmp_path, 2023, 5, "Q")
        assert "round=05" in str(result)

    def test_two_digit_round_not_over_padded(self, tmp_path):
        result = _session_output_dir(tmp_path, 2023, 15, "R")
        assert "round=15" in str(result)

    def test_session_type_preserved(self, tmp_path):
        for stype in ("R", "Q", "FP1", "SQ"):
            assert f"session={stype}" in str(_session_output_dir(tmp_path, 2023, 1, stype))


# --------------------------------------------------------------------------- #
# run_pipeline (mocked loader + cleaners)
# --------------------------------------------------------------------------- #
class TestRunPipeline:
    def _make_loaded_session(self, year=2023, rnd=1, stype="R"):
        return LoadedSession(year, rnd, stype, MagicMock())

    def test_calls_registry_get_for_each_active_cleaner(self, tmp_path):
        mock_cleaner_instance = MagicMock()
        mock_cleaner_cls = MagicMock(return_value=mock_cleaner_instance)
        mock_loader = MagicMock()
        mock_loader.iter_sessions.return_value = [self._make_loaded_session()]

        with patch("pipeline.pipeline.build_loader", return_value=mock_loader), \
             patch.object(pp.registry, "get", return_value=mock_cleaner_cls) as mock_get:
            run_pipeline([2023], output_root=tmp_path)

        assert mock_get.call_count == len(ACTIVE_CLEANERS)

    def test_instantiates_each_cleaner_with_session_and_identifiers(self, tmp_path):
        loaded = self._make_loaded_session(year=2023, rnd=3, stype="Q")
        mock_cleaner_cls = MagicMock()
        mock_loader = MagicMock()
        mock_loader.iter_sessions.return_value = [loaded]

        with patch("pipeline.pipeline.build_loader", return_value=mock_loader), \
             patch.object(pp.registry, "get", return_value=mock_cleaner_cls):
            run_pipeline([2023], output_root=tmp_path)

        for c in mock_cleaner_cls.call_args_list:
            args = c.args
            assert args[0] is loaded.session
            assert args[1] == 2023
            assert args[2] == 3
            assert args[3] == "Q"

    def test_calls_run_on_each_cleaner_instance(self, tmp_path):
        mock_cleaner_instance = MagicMock()
        mock_cleaner_cls = MagicMock(return_value=mock_cleaner_instance)
        mock_loader = MagicMock()
        mock_loader.iter_sessions.return_value = [self._make_loaded_session()]

        with patch("pipeline.pipeline.build_loader", return_value=mock_loader), \
             patch.object(pp.registry, "get", return_value=mock_cleaner_cls):
            run_pipeline([2023], output_root=tmp_path)

        assert mock_cleaner_instance.run.call_count == len(ACTIVE_CLEANERS)

    def test_run_receives_hive_partitioned_path(self, tmp_path):
        mock_cleaner_instance = MagicMock()
        mock_cleaner_cls = MagicMock(return_value=mock_cleaner_instance)
        mock_loader = MagicMock()
        mock_loader.iter_sessions.return_value = [self._make_loaded_session(rnd=7, stype="R")]

        with patch("pipeline.pipeline.build_loader", return_value=mock_loader), \
             patch.object(pp.registry, "get", return_value=mock_cleaner_cls):
            run_pipeline([2023], output_root=tmp_path)

        run_path = mock_cleaner_instance.run.call_args.args[0]
        assert "year=2023" in str(run_path)
        assert "round=07" in str(run_path)
        assert "session=R" in str(run_path)

    def test_continues_when_cleaner_raises(self, tmp_path):
        """A cleaner error on one table must not abort the rest of the session."""
        call_counts = {"run": 0}

        def _run(path):
            call_counts["run"] += 1
            if call_counts["run"] == 1:
                raise RuntimeError("simulated cleaner failure")

        mock_cleaner_instance = MagicMock()
        mock_cleaner_instance.run.side_effect = _run
        mock_cleaner_cls = MagicMock(return_value=mock_cleaner_instance)
        mock_loader = MagicMock()
        mock_loader.iter_sessions.return_value = [self._make_loaded_session()]

        with patch("pipeline.pipeline.build_loader", return_value=mock_loader), \
             patch.object(pp.registry, "get", return_value=mock_cleaner_cls):
            run_pipeline([2023], output_root=tmp_path)

        assert call_counts["run"] == len(ACTIVE_CLEANERS)

    def test_iterates_each_requested_year(self, tmp_path):
        mock_cleaner_cls = MagicMock(return_value=MagicMock())
        mock_loader = MagicMock()
        mock_loader.iter_sessions.return_value = []

        with patch("pipeline.pipeline.build_loader", return_value=mock_loader), \
             patch.object(pp.registry, "get", return_value=mock_cleaner_cls):
            run_pipeline([2021, 2022, 2023], output_root=tmp_path)

        assert mock_loader.iter_sessions.call_count == 3
        years_called = [c.args[0] for c in mock_loader.iter_sessions.call_args_list]
        assert years_called == [2021, 2022, 2023]

    def test_passes_mode_to_build_loader(self, tmp_path):
        mock_loader = MagicMock()
        mock_loader.iter_sessions.return_value = []

        with patch("pipeline.pipeline.build_loader", return_value=mock_loader) as mock_build, \
             patch.object(pp.registry, "get", return_value=MagicMock(return_value=MagicMock())):
            run_pipeline([2023], mode="viz", output_root=tmp_path)

        mock_build.assert_called_once_with(mode="viz", offline=False)

    def test_active_override_limits_cleaners_run(self, tmp_path):
        mock_cleaner_instance = MagicMock()
        mock_cleaner_cls = MagicMock(return_value=mock_cleaner_instance)
        mock_loader = MagicMock()
        mock_loader.iter_sessions.return_value = [self._make_loaded_session()]

        with patch("pipeline.pipeline.build_loader", return_value=mock_loader), \
             patch.object(pp.registry, "get", return_value=mock_cleaner_cls) as mock_get:
            run_pipeline([2023], active=["session_info"], output_root=tmp_path)

        assert mock_get.call_count == 1
        mock_get.assert_called_with("session_info")


# --------------------------------------------------------------------------- #
# split_train_test
# --------------------------------------------------------------------------- #
class TestSplitTrainTest:
    @pytest.fixture
    def features_parquet(self, tmp_path) -> Path:
        """Minimal features.parquet spanning 3 seasons."""
        df = pd.DataFrame({
            "year":         [2023, 2023, 2024, 2024, 2025, 2025],
            "round_number": [1,    2,    1,    2,    1,    2],
            "DriverId":     ["VER", "HAM", "VER", "HAM", "VER", "HAM"],
            "RacePosition": [1,    2,    1,    2,    1,    2],
        })
        path = tmp_path / "features.parquet"
        df.to_parquet(path, index=False)
        return path

    def test_writes_train_and_test_files(self, features_parquet):
        split_train_test(features_parquet)
        assert (features_parquet.parent / "train.parquet").exists()
        assert (features_parquet.parent / "test.parquet").exists()

    def test_test_contains_only_test_year(self, features_parquet):
        _, test = split_train_test(features_parquet)
        assert (test["year"] == 2025).all()

    def test_train_excludes_test_year(self, features_parquet):
        train, _ = split_train_test(features_parquet)
        assert (train["year"] != 2025).all()

    def test_all_rows_accounted_for(self, features_parquet):
        train, test = split_train_test(features_parquet)
        total = pd.read_parquet(features_parquet).shape[0]
        assert len(train) + len(test) == total

    def test_columns_identical_in_both_splits(self, features_parquet):
        train, test = split_train_test(features_parquet)
        assert list(train.columns) == list(test.columns)

    def test_written_files_match_returned_dataframes(self, features_parquet):
        train, test = split_train_test(features_parquet)
        pd.testing.assert_frame_equal(train, pd.read_parquet(features_parquet.parent / "train.parquet"))
        pd.testing.assert_frame_equal(test,  pd.read_parquet(features_parquet.parent / "test.parquet"))

    def test_custom_test_year(self, features_parquet):
        train, test = split_train_test(features_parquet, test_year=2024)
        assert (test["year"] == 2024).all()
        assert set(train["year"].unique()) == {2023, 2025}

    def test_empty_test_when_year_absent(self, features_parquet):
        _, test = split_train_test(features_parquet, test_year=2030)
        assert len(test) == 0

    def test_empty_train_when_only_test_year_present(self, tmp_path):
        df = pd.DataFrame({"year": [2025, 2025], "round_number": [1, 2]})
        path = tmp_path / "features.parquet"
        df.to_parquet(path, index=False)
        train, _ = split_train_test(path)
        assert len(train) == 0
