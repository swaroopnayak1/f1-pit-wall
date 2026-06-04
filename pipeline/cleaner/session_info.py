from .base import BaseCleaner
import pandas as pd


class SessionInfoCleaner(BaseCleaner):

    @property
    def table_name(self) -> str:
        return "session_info"

    def clean(self) -> pd.DataFrame:
        # Flatten the nested dict into a flat DataFrame with dot-separated columns
        df = pd.json_normalize(self.session.session_info, sep=".")
        # Produces dot-separated columns like:
        # "Meeting.Name", "Meeting.Country.Name", "Meeting.Circuit.ShortName", etc.

        # Keeping only relevant columns (if they exist in the raw dict)
        keep = [
            "Meeting.Name", "Meeting.Country.Name",
            "Meeting.Circuit.ShortName", "Type",
            "StartDate", "EndDate", "GmtOffset",
        ]
        df = df[[col for col in keep if col in df.columns]]

        # Parse dates and time deltas, ensuring timezone-naive UTC datetimes with microsecond precision
        df["StartDate"] = pd.to_datetime(df["StartDate"], utc=False).astype("datetime64[us]")
        df["EndDate"]   = pd.to_datetime(df["EndDate"],   utc=False).astype("datetime64[us]")

        # Convert string columns to pandas StringDtype for better memory efficiency and consistency
        str_cols = ["Meeting.Name", "Meeting.Country.Name",
                    "Meeting.Circuit.ShortName", "Type", "GmtOffset"]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype("string")

        return self._inject_partition_keys(df)