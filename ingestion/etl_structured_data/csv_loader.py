import pandas as pd

class CSVLoader:
    @staticmethod
    def load(path: str) -> pd.DataFrame:
        return pd.read_csv(path)
