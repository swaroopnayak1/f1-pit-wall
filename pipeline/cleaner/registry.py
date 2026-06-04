"""
Cleaner registry.

``CleanerRegistry`` maps table names to their cleaner strategy classes.
Registrations are explicit: call ``registry.register(name, cls)`` in one place
(__init__.py) rather than scattering decorators across every cleaner module.
Cleaners stay pure — no registry knowledge required.
"""
from __future__ import annotations

from .base import BaseCleaner


class CleanerRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, type[BaseCleaner]] = {}

    def register(self, name: str, cls: type[BaseCleaner]) -> None:
        """Register a cleaner strategy class under ``name``."""
        if not issubclass(cls, BaseCleaner):
            raise TypeError(f"{cls.__name__} must subclass BaseCleaner to be registered.")
        if name in self._registry:
            raise ValueError(
                f"Cleaner name '{name}' is already registered to "
                f"{self._registry[name].__name__}."
            )
        self._registry[name] = cls

    def get(self, name: str) -> type[BaseCleaner]:
        """Return the cleaner class registered under ``name``."""
        try:
            return self._registry[name]
        except KeyError:
            known = ", ".join(sorted(self._registry)) or "<none>"
            raise KeyError(f"No cleaner registered as '{name}'. Registered: {known}.") from None

    def all(self) -> dict[str, type[BaseCleaner]]:
        """Return a copy of the full ``name -> cleaner class`` mapping."""
        return dict(self._registry)


registry = CleanerRegistry()