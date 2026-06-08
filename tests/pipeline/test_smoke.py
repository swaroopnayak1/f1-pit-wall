"""
Smoke tests: verify the package wires up correctly at import time.
No network calls, no real FastF1 data, no mocking required.
"""
import importlib


def test_pipeline_package_importable():
    importlib.import_module("pipeline")


def test_cleaner_package_importable():
    importlib.import_module("pipeline.cleaner")


def test_loader_package_importable():
    importlib.import_module("pipeline.loader")


def test_pipeline_module_importable():
    importlib.import_module("pipeline.pipeline")


def test_registry_populated_after_import():
    """Importing pipeline.cleaner triggers explicit registrations for both concrete cleaners."""
    from pipeline.cleaner.registry import registry
    assert {"session_info", "driver_info"} <= set(registry.all())


def test_active_cleaners_all_resolvable():
    """Every name in ACTIVE_CLEANERS maps to a registered cleaner class."""
    from pipeline.cleaner.registry import registry
    from pipeline.pipeline import ACTIVE_CLEANERS
    for name in ACTIVE_CLEANERS:
        cls = registry.get(name)
        assert cls is not None, f"'{name}' is in ACTIVE_CLEANERS but not registered"


def test_build_loader_default_constructs():
    """build_loader() with default args constructs without touching the network."""
    from pipeline.loader import build_loader
    loader = build_loader("ml")
    assert loader is not None


def test_parse_years_basic_call():
    """parse_years is importable and handles a trivial input."""
    from pipeline.loader import parse_years
    assert parse_years(["2024"]) == [2024]
