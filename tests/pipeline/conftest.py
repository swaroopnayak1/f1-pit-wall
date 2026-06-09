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


@pytest.fixture
def session_results_session():
    """Minimal session whose results DataFrame mirrors FastF1 Race session results."""
    results = pd.DataFrame({
        "DriverNumber":       ["1",               "44",             "16"],
        "Abbreviation":       ["VER",             "HAM",            "LEC"],
        "FullName":           ["Max Verstappen",  "Lewis Hamilton", "Charles Leclerc"],
        "TeamName":           ["Red Bull Racing", "Mercedes",       "Ferrari"],
        "Position":           [1.0,               2.0,              3.0],
        "ClassifiedPosition": ["1",               "2",              "R"],
        "GridPosition":       [1.0,               3.0,              2.0],
        "Q1":  pd.to_timedelta(["NaT", "NaT", "NaT"]),
        "Q2":  pd.to_timedelta(["NaT", "NaT", "NaT"]),
        "Q3":  pd.to_timedelta(["NaT", "NaT", "NaT"]),
        "Time": pd.to_timedelta(["1:30:00", "0:00:05.123", "NaT"]),
        "Status": ["Finished", "Finished", "Retired"],
        "Points": [25.0, 18.0, 0.0],
        "Laps":   [57.0, 57.0, 40.0],
    })
    return SimpleNamespace(results=results)


@pytest.fixture
def laps_session():
    """Minimal session whose laps DataFrame mirrors FastF1 lap data."""
    laps = pd.DataFrame({
        "Driver":       ["VER",             "VER",             "HAM",             "HAM"],
        "DriverNumber": ["1",               "1",               "44",              "44"],
        "LapNumber":    [1.0,               2.0,               1.0,               2.0],
        "Stint":        [1.0,               1.0,               1.0,               1.0],
        # NaT on HAM lap 1 simulates an out-lap with no valid lap time
        "LapTime":     pd.to_timedelta(["0:01:30.123", "0:01:28.456", "NaT",         "0:01:29.789"]),
        "Sector1Time": pd.to_timedelta(["0:00:30.100", "0:00:29.800", "NaT",         "0:00:30.200"]),
        "Sector2Time": pd.to_timedelta(["0:00:30.200", "0:00:29.900", "NaT",         "0:00:30.300"]),
        "Sector3Time": pd.to_timedelta(["0:00:29.800", "0:00:28.700", "NaT",         "0:00:29.300"]),
        "SpeedI1":     [290.0, 292.0, 288.0, 291.0],
        "SpeedI2":     [310.0, 312.0, 308.0, 311.0],
        "SpeedFL":     [285.0, 287.0, 283.0, 286.0],
        "SpeedST":     [320.0, 322.0, 318.0, 321.0],
        "PitOutTime":  pd.to_timedelta(["0:00:05.000", "NaT",         "0:00:05.500", "NaT"]),
        "PitInTime":   pd.to_timedelta(["NaT",         "NaT",         "NaT",         "0:50:00"]),
        "Compound":    ["SOFT",            "SOFT",            "MEDIUM",          "MEDIUM"],
        "TyreLife":    [1.0,               2.0,               1.0,               2.0],
        "FreshTyre":   [True,              False,             True,              False],
        "Team":        ["Red Bull Racing", "Red Bull Racing", "Mercedes",        "Mercedes"],
        "LapStartDate": pd.to_datetime([
            "2023-03-05 15:00:00", "2023-03-05 15:01:30",
            "2023-03-05 15:00:00", "2023-03-05 15:01:30",
        ]),
        "TrackStatus": ["1", "1", "1", "1"],
        "Position":    [1.0, 1.0, 2.0, 2.0],
        "IsAccurate":  [False, True, False, True],
        "Deleted":     [False, False, False, False],
    })
    return SimpleNamespace(laps=laps)


@pytest.fixture
def weather_session():
    """Minimal session whose weather_data DataFrame mirrors FastF1 weather data."""
    weather = pd.DataFrame({
        "Time":          pd.to_timedelta(["0:00:18", "0:01:18", "0:02:18"]),
        "AirTemp":       [21.5,   21.6,   21.4],
        "Humidity":      [55.0,   54.0,   55.0],
        "Pressure":      [1017.3, 1017.3, 1017.2],
        "Rainfall":      [False,  False,  False],
        "TrackTemp":     [42.6,   43.6,   43.5],
        "WindDirection": [298,    176,    180],
        "WindSpeed":     [1.0,    0.7,    2.2],
    })
    return SimpleNamespace(weather_data=weather)
