"""Unit tests for pipeline.cleaner.registry (CleanerRegistry)."""
import pytest

from pipeline.cleaner.base import BaseCleaner
from pipeline.cleaner.registry import CleanerRegistry, registry


# --------------------------------------------------------------------------- #
# CleanerRegistry.register()
# --------------------------------------------------------------------------- #
class TestRegister:
    def test_adds_class_to_registry(self, empty_registry):
        class MyCleaner(BaseCleaner):
            @property
            def table_name(self): return "my_table"
            def clean(self): pass

        registry.register("my_table", MyCleaner)
        assert registry.get("my_table") is MyCleaner

    def test_duplicate_name_raises_value_error(self, empty_registry):
        class First(BaseCleaner):
            @property
            def table_name(self): return "dup"
            def clean(self): pass

        class Second(BaseCleaner):
            @property
            def table_name(self): return "dup"
            def clean(self): pass

        registry.register("dup", First)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("dup", Second)

    def test_non_cleaner_subclass_raises_type_error(self, empty_registry):
        class NotACleaner:
            pass

        with pytest.raises(TypeError, match="must subclass BaseCleaner"):
            registry.register("bad", NotACleaner)


# --------------------------------------------------------------------------- #
# CleanerRegistry.get()
# --------------------------------------------------------------------------- #
class TestGet:
    def test_returns_registered_class(self):
        from pipeline.cleaner.session_info import SessionInfoCleaner
        assert registry.get("session_info") is SessionInfoCleaner

    def test_unknown_name_raises_key_error(self):
        with pytest.raises(KeyError, match="No cleaner registered as 'missing'"):
            registry.get("missing")

    def test_error_message_lists_known_keys(self):
        with pytest.raises(KeyError) as exc_info:
            registry.get("no_such_table")
        assert "session_info" in str(exc_info.value)


# --------------------------------------------------------------------------- #
# CleanerRegistry.all()
# --------------------------------------------------------------------------- #
class TestAll:
    def test_returns_dict_containing_concrete_cleaners(self):
        result = registry.all()
        assert "session_info" in result
        assert "driver_info" in result

    def test_returns_a_copy(self):
        result = registry.all()
        result["injected"] = object()
        assert "injected" not in registry.all()
