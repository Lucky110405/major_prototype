import sqlalchemy
import pandas as pd
from sqlalchemy.engine import create_engine
from typing import Optional

class SQLConnector:
    def __init__(self, db_url: str):
        """
        db_url examples:
        - sqlite:///local.db
        - postgresql://user:pass@host:5432/db
        - mysql+pymysql://user:pass@host:3306/db
        """
        self.engine = create_engine(db_url)

    def list_tables(self):
        """Returns all tables in the DB."""
        return sqlalchemy.inspect(self.engine).get_table_names()

    def fetch_table(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Load table into pandas dataframe."""
        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"
        return pd.read_sql(query, self.engine)

    def run_query(self, query: str) -> pd.DataFrame:
        """Execute any SQL query."""
        return pd.read_sql(query, self.engine)

    def get_schema(self, table_name: str):
        """Get column names and types."""
        inspector = sqlalchemy.inspect(self.engine)
        return inspector.get_columns(table_name)
