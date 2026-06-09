from .base import BaseCleaner
import pandas as pd


class SessionResultsCleaner(BaseCleaner):

    @property
    def table_name(self) -> str:
        return "session_results"

    def clean(self) -> pd.DataFrame:
        df = self.session.results.copy().reset_index(drop=True)

        keep = [
            "DriverNumber", "Abbreviation", "FullName", "TeamName",
            "GridPosition", "Position", "ClassifiedPosition",
            "Q1", "Q2", "Q3",
            "Time", "Status", "Points", "Laps",
        ]
        df = df[[col for col in keep if col in df.columns]]

        # Timedelta columns -> total seconds (float64, NaT becomes NaN)
        for col in ["Q1", "Q2", "Q3", "Time"]:
            if col in df.columns:
                df[col] = df[col].dt.total_seconds()

        df["DriverNumber"] = pd.to_numeric(df["DriverNumber"], errors="coerce").astype("Int8")
        df["GridPosition"] = pd.to_numeric(df["GridPosition"], errors="coerce").astype("Float32")
        df["Position"]     = pd.to_numeric(df["Position"],     errors="coerce").astype("Float32")
        df["Points"]       = pd.to_numeric(df["Points"],       errors="coerce").astype("Float32")
        df["Laps"]         = pd.to_numeric(df["Laps"],         errors="coerce").astype("Float32")

        for col in ["Abbreviation", "FullName", "TeamName", "ClassifiedPosition", "Status"]:
            if col in df.columns:
                df[col] = df[col].astype("string")

        return self._inject_partition_keys(df)
