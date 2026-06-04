import fastf1
import sys
import os

# Global variables
SESSION_TYPES = ["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"]
VALID_MODES   = ["ml", "viz"]

# Enable cache to speed up repeated requests
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def parse_years(args: list[str]) -> list[int]:
    """
    Accepts three input modes:
      1. Single year       : 2024
      2. Multiple years    : 2021 2022 2024
      3. Year range        : 2021-2024
    Returns a sorted list of unique valid years.
    """
    years = set()
    for arg in args:
        # Range mode: "2021-2024"
        if arg.count("-") == 1 and not arg.startswith("-"):
            parts = arg.split("-")
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError:
                print(f"Error: Invalid range '{arg}'. Use format START-END (e.g. 2021-2024).")
                sys.exit(1)
            if start > end:
                print(f"Error: Range start ({start}) must be <= end ({end}).")
                sys.exit(1)
            for y in range(start, end + 1):
                years.add(y)
        # Single year or one of many years
        else:
            try:
                years.add(int(arg))
            except ValueError:
                print(f"Error: '{arg}' is not a valid year or range.")
                sys.exit(1)
    return sorted(years)


def load_session(year: int, rnd: int, session_type: str, mode: str = "ml") -> fastf1.core.Session:
    """
    Load a FastF1 session with data flags based on the mode.

    Modes:
      ml  — laps + weather only (lightweight, for feature engineering)
      viz — laps + telemetry + weather (heavier, for visualizations)
    """
    session = fastf1.get_session(year, rnd, session_type)

    if mode == "viz":
        session.load(
            laps=True,
            telemetry=True,   # needed for speed traces, throttle, brake charts
            weather=True,
            messages=False,
        )
    else:  # default: ml
        session.load(
            laps=True,
            telemetry=False,  # too granular for ML; aggregate from laps instead
            weather=True,
            messages=False,
        )

    return session


def get_events(year: int, mode: str = "ml"):  # type: ignore
    """Fetch and yield all F1 sessions for the given year."""
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        if schedule.empty:
            print(f"No events found for {year}.")
            return
        for _, event in schedule.iterrows():
            rnd = int(event["RoundNumber"])
            for session_type in SESSION_TYPES:
                try:
                    session = load_session(year, rnd, session_type, mode=mode)
                    yield year, rnd, session_type, session
                except Exception as exc:
                    # Session doesn't exist for this round, or data unavailable
                    print(f"[SKIP] {year} R{rnd:02d} {session_type}: {exc}")
                    continue
    except Exception as e:
        print(f"Error fetching schedule for {year}: {e}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single year   : python f1_events.py 2024")
        print("  Multiple years: python f1_events.py 2021 2022 2024")
        print("  Year range    : python f1_events.py 2021-2024")
        print("  With mode     : python f1_events.py 2024 --mode viz")
        print("  Modes         : ml (default), viz")
        print("Note: Extensive data is only available from 2018 onwards.")
        sys.exit(1)

    # Parse optional --mode flag
    args = sys.argv[1:]
    mode = "ml"  # default
    if "--mode" in args:
        idx = args.index("--mode")
        if idx + 1 >= len(args):
            print("Error: --mode flag requires a value (ml or viz).")
            sys.exit(1)
        mode = args[idx + 1]
        if mode not in VALID_MODES:
            print(f"Error: Invalid mode '{mode}'. Choose from: {', '.join(VALID_MODES)}.")
            sys.exit(1)
        args = [a for i, a in enumerate(args) if i != idx and i != idx + 1]

    years = parse_years(args)

    print(f"Mode: {mode}")
    print(f"Fetching schedules for: {', '.join(str(y) for y in years)}")

    for year in years:
        for year_, rnd, session_type, session in get_events(year, mode=mode):
            print(f"[LOADED] {year_} R{rnd:02d} {session_type}")

    print("Data fetching complete.")


if __name__ == "__main__":
    main()