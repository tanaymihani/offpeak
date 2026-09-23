"""Parquet I/O through DuckDB (avoids a pyarrow dependency)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    con = duckdb.connect()
    con.register("df_view", df)
    con.execute(f"COPY df_view TO '{Path(path)}' (FORMAT parquet)")
    con.close()


def read_parquet(path: Path, where: str | None = None) -> pd.DataFrame:
    query = f"SELECT * FROM read_parquet('{Path(path)}')"
    if where:
        query += f" WHERE {where}"
    con = duckdb.connect()
    df = con.execute(query).df()
    con.close()
    return df
