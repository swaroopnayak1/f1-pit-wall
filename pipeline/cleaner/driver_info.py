from .base import BaseCleaner
import pandas as pd


class DriverInfoCleaner(BaseCleaner):

    @property
    def table_name(self) -> str:
        return "driver_info"

    def clean(self) -> pd.DataFrame:
        # Flatten the list of drivers into a DataFrame, keeping only relevant columns and casting types
        # session.drivers is a list of driver number strings: ["1", "16", "44", ...]
        rows = [self.session.get_driver(drv) for drv in self.session.drivers]
        df   = pd.DataFrame(rows)

        # Keeping only relevant columns (if they exist in the raw dict)
        keep = [
            "DriverNumber", "BroadcastName", "FullName", "Abbreviation", "DriverId",
            "TeamName", "TeamColour", "FirstName", "LastName", "CountryCode",
        ]
        df = df[[col for col in keep if col in df.columns]]

        # Cast DriverNumber to numeric, coercing errors to NaN, then to Int8 (allows for nulls if parsing fails)
        # DriverNumber comes in as str ("1", "16") — cast to int8
        df["DriverNumber"] = pd.to_numeric(df["DriverNumber"], errors="coerce").astype("Int8")

        # All remaining columns stay as str
        str_cols = [
            "BroadcastName", "FullName", "Abbreviation", "DriverId",
            "TeamName", "TeamColour", "FirstName", "LastName", "CountryCode",
        ]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype("string")

        return self._inject_partition_keys(df)