import pandas as pd

class SchemaDetector:

    @staticmethod
    def detect(df: pd.DataFrame):
        schema = {
            "columns": {},
            "primary_key": None,
            "time_columns": []
        }

        for col in df.columns:
            dtype = str(df[col].dtype)
            schema["columns"][col] = dtype

            # Detect timestamp-like columns
            if "date" in col.lower() or "time" in col.lower():
                schema["time_columns"].append(col)

        # Primary key detection (simple heuristic)
        for col in df.columns:
            if df[col].is_unique:
                schema["primary_key"] = col
                break

        return schema
