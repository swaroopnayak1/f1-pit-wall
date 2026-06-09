from .base import BaseCleaner
import pandas as pd


class WeatherCleaner(BaseCleaner):

    @property
    def table_name(self) -> str:
        return "weather"

    def clean(self) -> pd.DataFrame:
        df = self.session.weather_data.copy().reset_index(drop=True)

        keep = [
            "Time", "AirTemp", "Humidity", "Pressure",
            "Rainfall", "TrackTemp", "WindDirection", "WindSpeed",
        ]
        df = df[[col for col in keep if col in df.columns]]

        # Timedelta -> total seconds (float64, NaT becomes NaN)
        if "Time" in df.columns:
            df["Time"] = df["Time"].dt.total_seconds()

        for col in ["AirTemp", "Humidity", "Pressure", "TrackTemp", "WindSpeed"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float32")

        if "WindDirection" in df.columns:
            df["WindDirection"] = pd.to_numeric(df["WindDirection"], errors="coerce").astype("Int16")

        if "Rainfall" in df.columns:
            df["Rainfall"] = df["Rainfall"].astype("boolean")

        return self._inject_partition_keys(df)
