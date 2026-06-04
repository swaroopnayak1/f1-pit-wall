"""
Ingestion pipeline orchestrator.

Ties the three stages together for every requested season:

    load (F1SessionLoader)  ->  clean/flatten (registered cleaners)  ->  write Parquet

Cleaners are looked up by name from the registry, so this module never imports
concrete cleaner classes directly. To control which tables are produced, edit
``ACTIVE_CLEANERS`` (or pass ``active`` to :func:`run_pipeline`); to add a new
table, create a BaseCleaner subclass and register it in
``pipeline/cleaner/__init__.py``.

Output is partitioned on disk by season / round / session, e.g.::
    data/year=2024/round=01/session=R/session_info.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .cleaner import registry
from .loader import build_loader, parse_years
from .loader.strategies import LoadStrategy

# Cleaners (by registry name) the pipeline runs for each session, in order.
ACTIVE_CLEANERS: list[str] = ["session_info", "driver_info"]

# Default output root for the Parquet dataset (pipeline/../data).
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def _session_output_dir(output_root: Path, year: int, rnd: int, session_type: str) -> Path:
    """Hive-style partition path for one session's tables."""
    return output_root / f"year={year}" / f"round={rnd:02d}" / f"session={session_type}"


def run_pipeline(
    years: list[int],
    *,
    mode: str = "ml",
    offline: bool = False,
    output_root: str | Path = DEFAULT_OUTPUT_DIR,
    active: list[str] | None = None,
) -> None:
    """
    Run the full pipeline for the given years.

    ``mode``    load strategy ("ml" or "viz").
    ``offline`` use the cache-only source (no network).
    ``active``  cleaner names to run; defaults to :data:`ACTIVE_CLEANERS`.
    """
    active = active if active is not None else ACTIVE_CLEANERS
    output_root = Path(output_root)
    loader = build_loader(mode=mode, offline=offline)

    # Resolve cleaner classes up front so a bad name fails fast, before any load.
    cleaner_classes = {name: registry.get(name) for name in active}

    print(f"Mode: {mode}{' (offline)' if offline else ''}")
    print(f"Cleaners: {', '.join(active)}")
    print(f"Output: {output_root}")
    print(f"Years: {', '.join(map(str, years))}")

    for year in years:
        for loaded in loader.iter_sessions(year):
            session_dir = _session_output_dir(
                output_root, loaded.year, loaded.round_number, loaded.session_type
            )
            tag = f"{loaded.year} R{loaded.round_number:02d} {loaded.session_type}"
            print(f"[LOADED] {tag}")
            for name, cleaner_cls in cleaner_classes.items():
                cleaner = cleaner_cls(
                    loaded.session,
                    loaded.year,
                    loaded.round_number,
                    loaded.session_type,
                )
                try:
                    cleaner.run(session_dir)
                except Exception as exc:
                    print(f"[ERROR] {tag} {name}: {exc}")

    print("Pipeline complete.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Load FastF1 sessions, clean them, and write Parquet tables.",
        epilog=(
            "Examples:\n"
            "  python -m pipeline.pipeline 2024\n"
            "  python -m pipeline.pipeline 2021 2022 2024 --mode viz\n"
            "  python -m pipeline.pipeline 2021-2024 --offline\n"
            "Note: extensive data is only available from 2018 onwards."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("years", nargs="+", help="Year(s) or range, e.g. 2024 / 2021 2022 / 2021-2024")
    parser.add_argument("--mode", choices=["ml", "viz"], default="ml", help="Load mode (default: ml)")
    parser.add_argument("--offline", action="store_true", help="Use cache only; do not hit the network")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR), help="Output root directory")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    try:
        years = parse_years(args.years)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    run_pipeline(years, mode=args.mode, offline=args.offline, output_root=args.out)


if __name__ == "__main__":
    main()
