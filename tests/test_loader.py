"""Unit tests for pipeline.loader (parse_years, strategies, F1SessionLoader, build_loader)."""
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from pipeline.loader import build_loader, parse_years
from pipeline.loader.loader import F1SessionLoader, LoadedSession, SESSION_TYPES
from pipeline.loader.strategies import (
    LiveF1Source,
    MLLoadStrategy,
    OfflineF1Source,
    VizLoadStrategy,
)


# --------------------------------------------------------------------------- #
# parse_years
# --------------------------------------------------------------------------- #
class TestParseYears:
    def test_single_year(self):
        assert parse_years(["2024"]) == [2024]

    def test_multiple_years_sorted(self):
        assert parse_years(["2022", "2021", "2024"]) == [2021, 2022, 2024]

    def test_range(self):
        assert parse_years(["2021-2024"]) == [2021, 2022, 2023, 2024]

    def test_range_single_element(self):
        assert parse_years(["2023-2023"]) == [2023]

    def test_mixed_range_and_singles(self):
        assert parse_years(["2020", "2022-2024"]) == [2020, 2022, 2023, 2024]

    def test_deduplication(self):
        assert parse_years(["2023", "2023"]) == [2023]

    def test_range_and_single_overlap_deduplicates(self):
        assert parse_years(["2021-2023", "2022"]) == [2021, 2022, 2023]

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="not a valid year"):
            parse_years(["abc"])

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="Invalid range"):
            parse_years(["2021-abc"])

    def test_range_start_greater_than_end_raises(self):
        with pytest.raises(ValueError, match="must be <= end"):
            parse_years(["2024-2021"])

    def test_empty_list_returns_empty(self):
        assert parse_years([]) == []


# --------------------------------------------------------------------------- #
# Load strategies
# --------------------------------------------------------------------------- #
class TestMLLoadStrategy:
    def test_name(self):
        assert MLLoadStrategy().name == "ml"

    def test_no_telemetry(self):
        assert MLLoadStrategy().data_flags()["telemetry"] is False

    def test_laps_and_weather_enabled(self):
        flags = MLLoadStrategy().data_flags()
        assert flags["laps"] is True
        assert flags["weather"] is True

    def test_messages_disabled(self):
        assert MLLoadStrategy().data_flags()["messages"] is False


class TestVizLoadStrategy:
    def test_name(self):
        assert VizLoadStrategy().name == "viz"

    def test_telemetry_enabled(self):
        assert VizLoadStrategy().data_flags()["telemetry"] is True

    def test_laps_and_weather_enabled(self):
        flags = VizLoadStrategy().data_flags()
        assert flags["laps"] is True
        assert flags["weather"] is True


# --------------------------------------------------------------------------- #
# build_loader
# --------------------------------------------------------------------------- #
class TestBuildLoader:
    def test_ml_mode_uses_ml_strategy(self):
        assert isinstance(build_loader("ml").load_strategy, MLLoadStrategy)

    def test_viz_mode_uses_viz_strategy(self):
        assert isinstance(build_loader("viz").load_strategy, VizLoadStrategy)

    def test_default_source_is_live(self):
        assert isinstance(build_loader("ml").source, LiveF1Source)

    def test_offline_flag_switches_to_offline_source(self):
        assert isinstance(build_loader("ml", offline=True).source, OfflineF1Source)

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid mode 'nope'"):
            build_loader("nope")

    def test_error_lists_valid_modes(self):
        with pytest.raises(ValueError, match="ml"):
            build_loader("bad")

    def test_custom_cache_dir(self, tmp_path):
        loader = build_loader("ml", cache_dir=tmp_path)
        assert Path(loader.source.cache_dir) == tmp_path


# --------------------------------------------------------------------------- #
# F1SessionLoader
# --------------------------------------------------------------------------- #
class TestF1SessionLoaderLoad:
    def test_calls_source_get_session_with_correct_args(self):
        mock_session = MagicMock()
        mock_source = MagicMock()
        mock_source.get_session.return_value = mock_session

        loader = F1SessionLoader(MLLoadStrategy(), mock_source)
        loader.load(2023, 1, "R")

        mock_source.get_session.assert_called_once_with(2023, 1, "R")

    def test_calls_session_load_with_strategy_flags(self):
        mock_session = MagicMock()
        mock_source = MagicMock()
        mock_source.get_session.return_value = mock_session

        loader = F1SessionLoader(MLLoadStrategy(), mock_source)
        loader.load(2023, 1, "R")

        mock_session.load.assert_called_once_with(**MLLoadStrategy().data_flags())

    def test_returns_the_session_object(self):
        mock_session = MagicMock()
        mock_source = MagicMock()
        mock_source.get_session.return_value = mock_session

        loader = F1SessionLoader(MLLoadStrategy(), mock_source)
        result = loader.load(2023, 1, "Q")

        assert result is mock_session

    def test_viz_strategy_passes_correct_flags(self):
        mock_session = MagicMock()
        mock_source = MagicMock()
        mock_source.get_session.return_value = mock_session

        loader = F1SessionLoader(VizLoadStrategy(), mock_source)
        loader.load(2023, 3, "R")

        mock_session.load.assert_called_once_with(**VizLoadStrategy().data_flags())


class TestF1SessionLoaderIterSessions:
    def _make_schedule(self, rounds):
        return pd.DataFrame({"RoundNumber": rounds})

    def test_yields_loaded_session_dataclass(self):
        schedule = self._make_schedule([1])
        mock_source = MagicMock()
        loader = F1SessionLoader(MLLoadStrategy(), mock_source)
        loader.load = MagicMock(return_value=MagicMock())

        with patch("fastf1.get_event_schedule", return_value=schedule):
            results = list(loader.iter_sessions(2023))

        assert all(isinstance(r, LoadedSession) for r in results)

    def test_yields_one_entry_per_session_type_per_round(self):
        schedule = self._make_schedule([1, 2])
        mock_source = MagicMock()
        loader = F1SessionLoader(MLLoadStrategy(), mock_source)
        loader.load = MagicMock(return_value=MagicMock())

        with patch("fastf1.get_event_schedule", return_value=schedule):
            results = list(loader.iter_sessions(2023))

        assert len(results) == 2 * len(SESSION_TYPES)

    def test_skips_session_when_load_raises(self):
        schedule = self._make_schedule([1])
        mock_source = MagicMock()
        loader = F1SessionLoader(MLLoadStrategy(), mock_source)

        def _load(year, rnd, session_type):
            if session_type != "R":
                raise Exception("unavailable")
            return MagicMock()

        loader.load = MagicMock(side_effect=_load)

        with patch("fastf1.get_event_schedule", return_value=schedule):
            results = list(loader.iter_sessions(2023))

        assert len(results) == 1
        assert results[0].session_type == "R"

    def test_skips_year_when_schedule_raises(self):
        mock_source = MagicMock()
        loader = F1SessionLoader(MLLoadStrategy(), mock_source)

        with patch("fastf1.get_event_schedule", side_effect=Exception("network error")):
            results = list(loader.iter_sessions(2023))

        assert results == []

    def test_loaded_session_fields_match_schedule_row(self):
        schedule = self._make_schedule([5])
        mock_session = MagicMock()
        mock_source = MagicMock()
        loader = F1SessionLoader(MLLoadStrategy(), mock_source)
        loader.load = MagicMock(return_value=mock_session)

        with patch("fastf1.get_event_schedule", return_value=schedule):
            results = list(loader.iter_sessions(2023))

        r_sessions = [r for r in results if r.session_type == "R"]
        assert len(r_sessions) == 1
        assert r_sessions[0].year == 2023
        assert r_sessions[0].round_number == 5
        assert r_sessions[0].session is mock_session
