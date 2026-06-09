from .base import BaseCleaner
import pandas as pd


class LapsCleaner(BaseCleaner):

    @property
    def table_name(self) -> str:
        return "laps"

    def clean(self) -> pd.DataFrame:
        df = self.session.laps.copy().reset_index(drop=True)

        keep = [
            "Driver", "DriverNumber", "LapNumber", "Stint",
            "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
            "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
            "PitOutTime", "PitInTime",
            "Compound", "TyreLife", "FreshTyre", "Team",
            "LapStartDate", "TrackStatus", "Position",
            "IsAccurate", "Deleted",
        ]
        df = df[[col for col in keep if col in df.columns]]

        # Timedelta columns -> total seconds (float64, NaT becomes NaN)
        td_cols = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "PitOutTime", "PitInTime"]
        for col in td_cols:
            if col in df.columns:
                df[col] = df[col].dt.total_seconds()

        df["DriverNumber"] = pd.to_numeric(df["DriverNumber"], errors="coerce").astype("Int8")
        df["LapNumber"]    = pd.to_numeric(df["LapNumber"],    errors="coerce").astype("Int16")
        df["Stint"]        = pd.to_numeric(df["Stint"],        errors="coerce").astype("Int8")
        df["TyreLife"]     = pd.to_numeric(df["TyreLife"],     errors="coerce").astype("Int16")
        df["Position"]     = pd.to_numeric(df["Position"],     errors="coerce").astype("Float32")

        for col in ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float32")

        for col in ["Driver", "Compound", "Team", "TrackStatus"]:
            if col in df.columns:
                df[col] = df[col].astype("string")

        if "LapStartDate" in df.columns:
            df["LapStartDate"] = pd.to_datetime(df["LapStartDate"], utc=False).astype("datetime64[us]")

        for col in ["FreshTyre", "IsAccurate", "Deleted"]:
            if col in df.columns:
                df[col] = df[col].astype("boolean")

        return self._inject_partition_keys(df)
