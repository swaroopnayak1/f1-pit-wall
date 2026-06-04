"""
Session loader.

:class:`F1SessionLoader` composes a :class:`LoadStrategy` (what to load) and a
:class:`SessionSource` (where to load it from) and exposes a small surface to
the orchestrator: load a single session, or iterate every session of a season.

This replaces the earlier procedural ``load_session`` / ``get_events`` helpers
with an object that owns its strategies, so the pipeline can be reconfigured
(ml vs viz, live vs offline) without touching call sites.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fastf1

from .strategies import (
    LiveF1Source,
    LoadStrategy,
    MLLoadStrategy,
    OfflineF1Source,
    SessionSource,
    VizLoadStrategy,
)

# All session types FastF1 may expose for an event.
SESSION_TYPES = ["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"]

# Available load modes, mapped to their strategy classes.
LOAD_STRATEGIES: dict[str, type[LoadStrategy]] = {
    "ml": MLLoadStrategy,
    "viz": VizLoadStrategy,
}

# Default cache lives at the repo root .cache (two levels up from this loader package).
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache"


@dataclass(frozen=True)
class LoadedSession:
    """A fully loaded session paired with the identifiers that locate it."""

    year: int
    round_number: int
    session_type: str
    session: "fastf1.core.Session"


class F1SessionLoader:
    """Loads FastF1 sessions using a load strategy and a session source."""

    def __init__(self, load_strategy: LoadStrategy, source: SessionSource):
        self.load_strategy = load_strategy
        self.source = source

    def load(self, year: int, rnd: int, session_type: str) -> "fastf1.core.Session":
        """Fetch and load a single session according to the load strategy."""
        session = self.source.get_session(year, rnd, session_type)
        session.load(**self.load_strategy.data_flags())
        return session

    def iter_sessions(self, year: int) -> Iterator[LoadedSession]:
        """
        Yield every loadable session for a season.

        Sessions that don't exist for a round, or whose data is unavailable, are
        skipped with a log line rather than aborting the whole season.
        """
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc: 
            print(f"[ERROR] Could not fetch schedule for {year}: {exc}")
            return

        if schedule.empty:
            print(f"[WARN] No events found for {year}.")
            return

        for _, event in schedule.iterrows():
            rnd = int(event["RoundNumber"])
            for session_type in SESSION_TYPES:
                try:
                    session = self.load(year, rnd, session_type)
                except Exception as exc:
                    print(f"[SKIP] {year} R{rnd:02d} {session_type}: {exc}")
                    continue
                yield LoadedSession(year, rnd, session_type, session)


def build_loader(
    mode: str = "ml",
    *,
    offline: bool = False,
    cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR,
) -> F1SessionLoader:
    """
    Construct an :class:`F1SessionLoader` for the given mode and source.

    ``mode``    selects the load strategy ("ml" or "viz").
    ``offline`` chooses an offline (cache-only) source over a live one.
    """
    try:
        strategy_cls = LOAD_STRATEGIES[mode]
    except KeyError:
        valid = ", ".join(sorted(LOAD_STRATEGIES))
        raise ValueError(f"Invalid mode '{mode}'. Choose from: {valid}.") from None

    source: SessionSource = (
        OfflineF1Source(cache_dir) if offline else LiveF1Source(cache_dir)
    )
    return F1SessionLoader(strategy_cls(), source)


def parse_years(args: list[str]) -> list[int]:
    """
    Parse year arguments into a sorted list of unique years.

    Accepts single years ("2024"), multiple years ("2021 2022 2024"), and
    ranges ("2021-2024"); modes can be mixed. Raises ``ValueError`` on bad input.
    """
    years: set[int] = set()
    for arg in args:
        # Range mode: "2021-2024"
        if arg.count("-") == 1 and not arg.startswith("-"):
            start_str, end_str = arg.split("-")
            try:
                start, end = int(start_str), int(end_str)
            except ValueError:
                raise ValueError(
                    f"Invalid range '{arg}'. Use START-END (e.g. 2021-2024)."
                ) from None
            if start > end:
                raise ValueError(f"Range start ({start}) must be <= end ({end}).")
            years.update(range(start, end + 1))
        else:
            try:
                years.add(int(arg))
            except ValueError:
                raise ValueError(f"'{arg}' is not a valid year or range.") from None
    return sorted(years)
