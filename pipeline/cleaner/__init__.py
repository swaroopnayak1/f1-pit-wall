"""
Cleaner package.

All cleaner strategies are explicitly registered here. To add a new cleaner:
  1. Create a module with a class that subclasses BaseCleaner.
  2. Import it and call registry.register("<table_name>", YourCleaner) below.
"""
from .base import BaseCleaner
from .registry import CleanerRegistry, registry
from .driver_info import DriverInfoCleaner
from .session_info import SessionInfoCleaner
from .session_results import SessionResultsCleaner
from .laps import LapsCleaner
from .weather import WeatherCleaner

registry.register("driver_info", DriverInfoCleaner)
registry.register("session_info", SessionInfoCleaner)
registry.register("session_results", SessionResultsCleaner)
registry.register("laps", LapsCleaner)
registry.register("weather", WeatherCleaner)

__all__ = ["BaseCleaner", "CleanerRegistry", "registry"]