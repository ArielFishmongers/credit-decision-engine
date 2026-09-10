"""The schema is load-bearing: the data files carry no header row, so a wrong
column list silently mislabels every field rather than failing.

These tests check it against Freddie Mac's own Release 47 artefacts. Those live under
``docs/freddie-mac/`` which is gitignored (their documentation is not ours to
redistribute), so the tests skip when it is absent — run ``scripts/fetch_reference_docs.sh``
to restore it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cde.data.schema import (
    ORIGINATION_COLUMNS,
    ORIGINATION_NAMES,
    PERFORMANCE_COLUMNS,
    PERFORMANCE_NAMES,
)
from tests.conftest import REFERENCE, SAMPLES, requires_reference


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


# --- Release 47 enumerations -------------------------------------------------------
# These come from general_user_guide_july_2026.pdf rather than the header files, which
# carry column names only. A wrong code list here does not fail loudly: it silently
# reclassifies defaults as censored observations.


def test_zero_balance_groups_partition_the_enumeration() -> None:
    """Every Release 47 zero-balance code must be classified exactly once.

    An unclassified code would be silently swept into "still alive", overstating
    survival; a double-classified one would be counted as both default and censored.
    """
    from cde.data.schema import (
        ZERO_BALANCE_CODES,
        ZERO_BALANCE_DEFAULT,
        ZERO_BALANCE_INFORMATIVE_EXIT,
        ZERO_BALANCE_PREPAID,
    )

    groups = [{ZERO_BALANCE_PREPAID}, set(ZERO_BALANCE_DEFAULT), set(ZERO_BALANCE_INFORMATIVE_EXIT)]
    assert set().union(*groups) == set(ZERO_BALANCE_CODES)
    assert sum(len(g) for g in groups) == len(ZERO_BALANCE_CODES), "codes overlap between groups"
    assert set(ZERO_BALANCE_CODES) == {"01", "02", "03", "09", "15", "16", "96"}


def test_third_party_sale_counts_as_a_credit_event() -> None:
    """02 is a foreclosure-auction disposition, and Freddie Mac computes ACTUAL LOSS for
    02, 03, 09 and 15. Pre-Release-47 write-ups commonly list only 03 and 09."""
    from cde.data.schema import ZERO_BALANCE_DEFAULT

    assert "02" in ZERO_BALANCE_DEFAULT


def test_delinquency_threshold_cannot_be_a_bare_string_comparison() -> None:
    """The reason DELINQUENCY_NON_NUMERIC exists.

    "RA" is a default state and "XX" is missing data, but both sort above "03", so
    ``status >= threshold`` alone would silently treat unknown status as 90+ DPD.
    """
    from cde.data.schema import DELINQUENCY_NON_NUMERIC, DELINQUENCY_NOT_AVAILABLE

    assert all(value >= "03" for value in DELINQUENCY_NON_NUMERIC)
    assert DELINQUENCY_NOT_AVAILABLE in DELINQUENCY_NON_NUMERIC


def test_sentinel_keys_are_real_columns() -> None:
    """A typo'd key would never fire, leaving 999s in a ratio and 9999s in a FICO."""
    from cde.data.schema import NOT_AVAILABLE_SENTINELS

    known = set(ORIGINATION_NAMES) | set(PERFORMANCE_NAMES)
    assert set(NOT_AVAILABLE_SENTINELS) <= known


@requires_reference
def test_sentinels_actually_occur_in_the_format_example() -> None:
    """Guards against a sentinel that is right in principle but wrong in width --
    "999" versus "999.00", say. Only checks columns where the 1,000-row example
    happens to contain a missing value, so it is a spot-check, not a proof."""
    from cde.data.loader import read_origination
    from cde.data.schema import NOT_AVAILABLE_SENTINELS

    orig = read_origination(SAMPLES / "origination_sample_file.txt")
    hits = {
        column: int((orig[column] == sentinel).sum())
        for column, sentinel in NOT_AVAILABLE_SENTINELS.items()
        if column in orig.columns
    }
    assert hits["vantagescore_4_0"] > 0, "VantageScore 4.0 is new in Release 47 and mostly 9999"
    assert any(count > 0 for count in hits.values())
