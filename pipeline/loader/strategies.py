"""
Loading strategies.

Two independent (layered) axes of variation, each its own Strategy hierarchy:

1. :class:`LoadStrategy` — *what* data to pull into a session. The cost of a
   FastF1 load is dominated by telemetry, so the mode (ml vs viz) decides which
   data flags are enabled.

2. :class:`SessionSource` — *where* a session comes from and how the cache
   behaves. Live sources hit the network (and populate the cache); offline
   sources serve only from the local cache.

:class:`~pipeline.loader.loader.F1SessionLoader` composes one of each, so the
two concerns can vary independently (e.g. an ml load served entirely offline).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import fastf1


# --------------------------------------------------------------------------- #
# Axis 1: what data to load
# --------------------------------------------------------------------------- #
class LoadStrategy(ABC):
    """Decides which FastF1 data streams are loaded into a session."""

    #: Human-readable mode key, e.g. "ml" or "viz".
    name: str

    @abstractmethod
    def data_flags(self) -> dict[str, bool]:
        """Return keyword flags passed to ``session.load(**flags)``."""
        raise NotImplementedError


class MLLoadStrategy(LoadStrategy):
    """Lightweight: laps + weather, no telemetry. For feature engineering."""

    name = "ml"

    def data_flags(self) -> dict[str, bool]:
        return {
            "laps": True,
            "telemetry": False,  # too granular for ML; aggregate from laps instead
            "weather": True,
            "messages": False,
        }


class VizLoadStrategy(LoadStrategy):
    """Heavier: laps + telemetry + weather. For speed/throttle/brake charts."""

    name = "viz"

    def data_flags(self) -> dict[str, bool]:
        return {
            "laps": True,
            "telemetry": True,  # needed for speed traces, throttle, brake charts
            "weather": True,
            "messages": False,
        }


# --------------------------------------------------------------------------- #
# Axis 2: where the session comes from
# --------------------------------------------------------------------------- #
class SessionSource(ABC):
    """
    Provides raw (unloaded) FastF1 sessions and configures cache behaviour.

    Implementations configure the FastF1 cache on construction; because FastF1's
    cache is process-global, the most recently constructed source wins.
    """

    def __init__(self, cache_dir: str | os.PathLike[str]):
        self.cache_dir = os.fspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._configure_cache()

    @abstractmethod
    def _configure_cache(self) -> None:
        """Enable/configure the FastF1 cache for this source's behaviour."""
        raise NotImplementedError

    def get_session(self, year: int, rnd: int, session_type: str) -> "fastf1.core.Session":
        """Return the (not-yet-loaded) session for the given identifiers."""
        return fastf1.get_session(year, rnd, session_type)


class LiveF1Source(SessionSource):
    """Hits the network as needed and writes responses to the cache."""

    def _configure_cache(self) -> None:
        fastf1.Cache.enable_cache(self.cache_dir)


class OfflineF1Source(SessionSource):
    """
    Serves only from the local cache; never touches the network.

    Useful for deterministic re-runs of the pipeline. Sessions absent from the
    cache will raise when loaded.
    """

    def _configure_cache(self) -> None:
        fastf1.Cache.enable_cache(self.cache_dir)
        fastf1.Cache.offline_mode(enabled=True)
