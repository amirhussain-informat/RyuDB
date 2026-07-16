"""Storage: GPU-side columnar scans over Parquet via cuDF.

Phase 1 is read-only. `scan` reads only the requested columns directly into GPU
memory with `cudf.read_parquet`, which lets the optimizer's projection pruning
pay off in both I/O and GPU memory.
"""

from __future__ import annotations

import cudf

from .catalog import TableInfo


def scan(table: TableInfo, columns: set[str] | None) -> cudf.DataFrame:
    cols = sorted(columns) if columns else None
    paths = table.paths
    if len(paths) == 1:
        df = cudf.read_parquet(paths[0], columns=cols)
    else:
        df = cudf.read_parquet(paths, columns=cols)
    _coerce_decimals(df)
    return df


def _coerce_decimals(df: cudf.DataFrame) -> None:
    """Cast DECIMAL columns to float64 in place.

    TPC-H numeric columns are DECIMAL in Parquet. cuDF's decimal reductions
    (sum/min/max/avg) are not fully supported, and mixing Decimal/float types
    across the GPU/CPU boundary breaks value comparison. float64 is exact enough
    for the Phase-1 analytical workload and the benchmark's 6-decimal rounding.
    """
    for col in list(df.columns):
        dtype = df[col].dtype
        if "Decimal" in type(dtype).__name__:
            df[col] = df[col].astype("float64")