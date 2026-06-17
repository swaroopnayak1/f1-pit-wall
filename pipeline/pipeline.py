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

import pandas as pd

from .cleaner import registry
from .feature_engineering import run_feature_engineering
from .feature_engineering.feature_engineering import DEFAULT_OUTPUT as _DEFAULT_FEATURES
from .loader import build_loader, parse_years
from .loader.strategies import LoadStrategy

# Cleaners (by registry name) the pipeline runs for each session, in order.
ACTIVE_CLEANERS: list[str] = ["session_info", "driver_info", "session_results", "laps", "weather"]

# Default output root for the Parquet dataset (pipeline/../data).
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def split_train_test(
    features_path: str | Path = _DEFAULT_FEATURES,
    *,
    test_year: int = 2025,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the feature matrix into train and test sets by year and write them
    as Parquet files next to ``features_path``.

    The most recent season (``test_year``) becomes the held-out test set;
    all earlier seasons form the training set.

    Parameters
    ----------
    features_path: Path to the features Parquet file produced by
                   :func:`run_feature_engineering`.
    test_year:     The season to hold out as test data (default: 2025).

    Returns
    -------
    (train_df, test_df) — DataFrames with identical columns.

    Side effects
    ------------
    Writes ``train.parquet`` and ``test.parquet`` in the same directory as
    ``features_path``.
    """
    features_path = Path(features_path)
    df = pd.read_parquet(features_path)
    train = df[df["year"] != test_year].reset_index(drop=True)
    test  = df[df["year"] == test_year].reset_index(drop=True)

    out_dir = features_path.parent
    train.to_parquet(out_dir / "train.parquet", index=False, compression="snappy")
    test.to_parquet(out_dir  / "test.parquet",  index=False, compression="snappy")

    print(
        f"[split] train={len(train)} rows ({sorted(train['year'].unique())}) -> {out_dir / 'train.parquet'}\n"
        f"[split] test={len(test)} rows (year={test_year}) -> {out_dir / 'test.parquet'}"
    )
    return train, test


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
    Run the data ingestion stage for the given years.

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

    print("Ingestion complete.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Load FastF1 sessions, clean them, and write Parquet tables.",
        epilog=(
            "Examples:\n"
            "  python -m pipeline.pipeline 2024                       # full pipeline\n"
            "  python -m pipeline.pipeline 2024 --module data         # ingestion only\n"
            "  python -m pipeline.pipeline 2024 --module fe           # feature engineering only\n"
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
    parser.add_argument("--module", choices=["data", "fe", "all"], default="all",
                        help="Stage to run: data (ingestion only), fe (feature engineering only), all (default)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    try:
        years = parse_years(args.years)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    output_root = Path(args.out)

    if args.module in ("data", "all"):
        run_pipeline(years, mode=args.mode, offline=args.offline, output_root=output_root)
    if args.module in ("fe", "all"):
        features_path = run_feature_engineering(output_root, years=years)
        split_train_test(features_path)

    print("Done.")


if __name__ == "__main__":
    main()
