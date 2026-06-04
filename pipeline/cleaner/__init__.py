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

registry.register("driver_info", DriverInfoCleaner)
registry.register("session_info", SessionInfoCleaner)

__all__ = ["BaseCleaner", "CleanerRegistry", "registry"]