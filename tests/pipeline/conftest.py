"""Shared pytest fixtures."""
from types import SimpleNamespace

import pandas as pd
import pytest

from pipeline.cleaner.registry import registry as _registry_singleton


# --------------------------------------------------------------------------- #
# Registry helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def empty_registry(monkeypatch):
    """Swap the registry singleton's internal dict for an empty one for test isolation."""
    monkeypatch.setattr(_registry_singleton, "_registry", {})
    yield _registry_singleton._registry


# --------------------------------------------------------------------------- #
# Mock FastF1 sessions
# --------------------------------------------------------------------------- #
@pytest.fixture
def session_info_session():
    """Minimal session whose session_info dict mirrors the FastF1 real structure."""
    sess = SimpleNamespace()
    sess.session_info = {
        "Meeting": {
            "Name": "Bahrain Grand Prix",
            "Country": {"Name": "Bahrain"},
            "Circuit": {"ShortName": "Sakhir"},
        },
        "Type": "Race",
        "StartDate": "2023-03-05T15:00:00",
        "EndDate": "2023-03-05T17:00:00",
        "GmtOffset": "03:00:00",
    }
    return sess


@pytest.fixture
def driver_info_session():
    """Minimal session whose drivers list mirrors the FastF1 real structure."""
    _DRIVERS = {
        "1": {
            "DriverNumber": "1", "BroadcastName": "M VERSTAPPEN",
            "FullName": "Max Verstappen", "Abbreviation": "VER",
            "DriverId": "verstappen", "TeamName": "Red Bull Racing",
            "TeamColour": "3671C6", "FirstName": "Max",
            "LastName": "Verstappen", "CountryCode": "NLD",
        },
        "11": {
            "DriverNumber": "11", "BroadcastName": "S PEREZ",
            "FullName": "Sergio Perez", "Abbreviation": "PER",
            "DriverId": "perez", "TeamName": "Red Bull Racing",
            "TeamColour": "3671C6", "FirstName": "Sergio",
            "LastName": "Perez", "CountryCode": "MEX",
        },
    }
    return SimpleNamespace(
        drivers=list(_DRIVERS.keys()),
        get_driver=lambda n: _DRIVERS[n],
    )
