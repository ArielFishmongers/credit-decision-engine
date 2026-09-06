"""The schema is load-bearing: the data files carry no header row, so a wrong
column list silently mislabels every field rather than failing.

These tests check it against Freddie Mac's own Release 47 artefacts. Those live under
``docs/freddie-mac/`` which is gitignored (their documentation is not ours to
redistribute), so the tests skip when it is absent — run ``scripts/fetch_reference_docs.sh``
to restore it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cde.data.schema import (
    ORIGINATION_COLUMNS,
    ORIGINATION_NAMES,
    PERFORMANCE_COLUMNS,
    PERFORMANCE_NAMES,
)

REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "freddie-mac"
SAMPLES = REFERENCE / "Sample Files"

requires_reference = pytest.mark.skipif(
    not REFERENCE.is_dir(), reason="reference docs absent; run scripts/fetch_reference_docs.sh"
)


def test_release_47_column_counts() -> None:
    assert len(ORIGINATION_COLUMNS) == 31
    assert len(PERFORMANCE_COLUMNS) == 35


def test_names_are_unique() -> None:
    assert len(set(ORIGINATION_NAMES)) == len(ORIGINATION_NAMES)
    assert len(set(PERFORMANCE_NAMES)) == len(PERFORMANCE_NAMES)


def test_renamed_fields_use_release_47_names() -> None:
    """Anything published before August 2026 has the superseded names."""
    officials = {official for _, official in ORIGINATION_COLUMNS + PERFORMANCE_COLUMNS}
    current_names = (
        "CLASSIC FICO",
        "LOAN IDENTIFIER",
        "PERIOD",
        "NET SALES PROCEEDS",
        "ACTUAL LOSS",
    )
    for current in current_names:
        assert current in officials
    for superseded in ("CREDIT SCORE", "LOAN SEQUENCE NUMBER", "MONTHLY REPORTING PERIOD"):
        assert superseded not in officials


def test_join_key_present_in_both_files() -> None:
    assert "loan_identifier" in ORIGINATION_NAMES
    assert "loan_identifier" in PERFORMANCE_NAMES


@requires_reference
@pytest.mark.parametrize(
    "header_file, expected",
    [
        ("origination_data_file_header.txt", ORIGINATION_COLUMNS),
        ("performance_data_file_header.txt", PERFORMANCE_COLUMNS),
    ],
)
def test_matches_freddie_mac_headers(header_file: str, expected: list[tuple[str, str]]) -> None:
    official = (REFERENCE / header_file).read_text().strip().split("|")
    assert official == [name for _, name in expected]


@requires_reference
@pytest.mark.parametrize(
    "sample_file, names",
    [
        ("origination_sample_file.txt", ORIGINATION_NAMES),
        ("performance_sample_file.txt", PERFORMANCE_NAMES),
    ],
)
def test_sample_rows_have_the_expected_field_count(sample_file: str, names: list[str]) -> None:
    lines = (SAMPLES / sample_file).read_text().splitlines()
    assert lines, "sample file is empty"
    for line in lines:
        assert len(line.split("|")) == len(names)


@requires_reference
def test_loader_reads_the_format_example() -> None:
    from cde.data.loader import read_origination, read_performance

    orig = read_origination(SAMPLES / "origination_sample_file.txt")
    perf = read_performance(SAMPLES / "performance_sample_file.txt")

    assert list(orig.columns) == ORIGINATION_NAMES
    assert list(perf.columns) == PERFORMANCE_NAMES

    # Delinquency status is alphanumeric ("RA" = REO acquisition), so it must never be
    # parsed as a number: that would both crash on "RA" and strip the leading zeros.
    status = perf["current_loan_delinquency_status"].dropna()
    assert not pd.api.types.is_numeric_dtype(status)
    assert status.map(type).eq(str).all()
    assert status.str.fullmatch(r"[0-9A-Z]+").all()
