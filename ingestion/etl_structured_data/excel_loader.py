import pandas as pd

class ExcelLoader:
    @staticmethod
    def load(path: str, sheet_name=0) -> pd.DataFrame:
        return pd.read_excel(path, sheet_name=sheet_name)
