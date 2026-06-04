"""Unit tests for pipeline.cleaner.registry."""
import pytest

import pipeline.cleaner.registry as reg
from pipeline.cleaner.base import BaseCleaner
from pipeline.cleaner.registry import all_cleaners, get_cleaner, register


# --------------------------------------------------------------------------- #
# register decorator
# --------------------------------------------------------------------------- #
class TestRegisterDecorator:
    def test_adds_class_to_registry(self, empty_registry):
        @register("my_table")
        class MyCleaner(BaseCleaner):
            @property
            def table_name(self):
                return "my_table"

            def clean(self):
                pass

        assert empty_registry["my_table"] is MyCleaner

    def test_returns_the_original_class_unchanged(self, empty_registry):
        @register("identity_check")
        class MyCleaner(BaseCleaner):
            @property
            def table_name(self):
                return "identity_check"

            def clean(self):
                pass

        assert MyCleaner.__name__ == "MyCleaner"

    def test_duplicate_name_raises_value_error(self, empty_registry):
        @register("dup")
        class First(BaseCleaner):
            @property
            def table_name(self):
                return "dup"

            def clean(self):
                pass

        with pytest.raises(ValueError, match="already registered"):

            @register("dup")
            class Second(BaseCleaner):
                @property
                def table_name(self):
                    return "dup"

                def clean(self):
                    pass

    def test_non_cleaner_subclass_raises_type_error(self, empty_registry):
        with pytest.raises(TypeError, match="must subclass BaseCleaner"):

            @register("bad")
            class NotACleaner:
                pass


# --------------------------------------------------------------------------- #
# get_cleaner
# --------------------------------------------------------------------------- #
class TestGetCleaner:
    def test_returns_registered_class(self):
        from pipeline.cleaner.session_info import SessionInfoCleaner

        assert get_cleaner("session_info") is SessionInfoCleaner

    def test_unknown_name_raises_key_error(self):
        with pytest.raises(KeyError, match="No cleaner registered as 'missing'"):
            get_cleaner("missing")

    def test_error_message_lists_known_keys(self):
        with pytest.raises(KeyError) as exc_info:
            get_cleaner("no_such_table")
        assert "session_info" in str(exc_info.value)


# --------------------------------------------------------------------------- #
# all_cleaners
# --------------------------------------------------------------------------- #
class TestAllCleaners:
    def test_returns_dict_containing_concrete_cleaners(self):
        result = all_cleaners()
        assert "session_info" in result
        assert "driver_info" in result

    def test_returns_a_copy(self):
        result = all_cleaners()
        result["injected"] = object()
        assert "injected" not in reg._REGISTRY
