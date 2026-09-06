"""Read Freddie Mac SFLD pipe-delimited files into DataFrames.

The files carry no header row, so :mod:`cde.data.schema` supplies the column
names and ordering. Everything is read as string first: several columns encode
"not available" as blanks or sentinels (999, 9999), and one of them
(``current_loan_delinquency_status``) is genuinely alphanumeric.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cde.data.schema import ORIGINATION_NAMES, PERFORMANCE_NAMES


def _read(path: Path, names: list[str]) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=names,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )
    if df.shape[1] != len(names):
        raise ValueError(f"{path}: expected {len(names)} columns, found {df.shape[1]}")
    return df


def read_origination(path: str | Path) -> pd.DataFrame:
    """Read an origination file (``orig_YYYYQn.txt`` or ``sample_orig_YYYY.txt``)."""
    return _read(Path(path), ORIGINATION_NAMES)


def read_performance(path: str | Path) -> pd.DataFrame:
    """Read a performance file (``perf_YYYYQn.txt`` or ``sample_perf_YYYY.txt``)."""
    return _read(Path(path), PERFORMANCE_NAMES)
