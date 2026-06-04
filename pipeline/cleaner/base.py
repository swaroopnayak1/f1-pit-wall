# cleaners/base.py
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import fastf1


class BaseCleaner(ABC):
    """
    Base class for all FastF1 session cleaners.
    Each subclass handles one table (session_info, driver_info, etc.)
    """

    def __init__(self, session: fastf1.core.Session, year: int, round_number: int, session_type: str):
        self.session      = session
        self.year         = year
        self.round_number = round_number
        self.session_type = session_type

    # Abstract methods that each cleaner sub class must implement:
    @property
    @abstractmethod
    def table_name(self) -> str:
        """The output Parquet filename stem, e.g. 'session_info'."""
        pass

    @abstractmethod
    def clean(self) -> pd.DataFrame:
        """Flatten and clean the raw data. Returns a cleaned DataFrame."""
        pass
    
    # Concrete method to write the cleaned DataFrame to Parquet:
    def run(self, output_dir: str | Path) -> Path:
        """
        Clean and write to Parquet. Returns the path written.
        Call this instead of clean() directly.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df   = self.clean()
        path = output_dir / f"{self.table_name}.parquet"

        df.to_parquet(path, index=False, compression="snappy")
        print(f"[{self.table_name}] wrote {len(df)} row(s) -> {path}")

        return path

    def _inject_partition_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach year, round_number, session_type to any DataFrame."""
        df["year"]         = pd.array([self.year]         * len(df), dtype="int16")
        df["round_number"] = pd.array([self.round_number] * len(df), dtype="int8")
        df["session_type"] = self.session_type
        return df