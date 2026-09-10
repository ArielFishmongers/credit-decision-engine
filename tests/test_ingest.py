"""The typed layer is where silent corruption would happen: nothing in this pipeline
raises if a postal code loses its leading zero or a 999 sentinel is read as a ratio.
These tests pin the casts that matter.

Built on the 1,000-row format example -- see tests/conftest.py on what that can and
cannot prove.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from cde.config import load_config
from cde.data.ingest import cast_losses, connect, discover, ingest, unpack
from tests.conftest import requires_reference, write_config

pytestmark = requires_reference


@pytest.fixture
def con(config_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    config = load_config(config_path)
    connection = connect(config.database)
    ingest(config, connection)
    yield connection
    connection.close()


def test_report_counts_what_was_loaded(config_path: Path) -> None:
    config = load_config(config_path)
    with connect(config.database) as connection:
        report = ingest(config, connection)
    assert len(report.files) == 4  # orig + perf, two vintages
    assert report.vintages == (2000, 2001)
    assert report.origination_rows == 2_000
    assert "origination" in report.summary()


def test_nothing_is_lost_in_the_cast(con: duckdb.DuckDBPyConnection) -> None:
    """A non-empty result means the schema types a column wrongly, or the data holds a
    code the user guide does not document. Either way it must not pass silently."""
    assert cast_losses(con) == []


def test_vintage_is_stamped_from_the_file(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        "SELECT origination_vintage, count(*) FROM origination GROUP BY 1 ORDER BY 1"
    ).fetchall()
    assert rows == [(2000, 1_000), (2001, 1_000)]


def test_postal_code_keeps_its_leading_zeros(con: duckdb.DuckDBPyConnection) -> None:
    """Three-digit ZIP prefix. Typed as a number it would become 56 and stop joining."""
    assert con.execute("SELECT typeof(postal_code) FROM origination LIMIT 1").fetchone() == (
        "VARCHAR",
    )
    zero_prefixed = con.execute(
        "SELECT count(*) FROM origination WHERE postal_code LIKE '0%'"
    ).fetchone()
    assert zero_prefixed is not None and zero_prefixed[0] > 0


def test_unknown_postal_code_is_null_not_zero(con: duckdb.DuckDBPyConnection) -> None:
    """"000 = Unknown" per the user guide, and 000 is not a real three-digit prefix."""
    assert con.execute("SELECT count(*) FROM origination WHERE postal_code = '000'").fetchone() == (
        0,
    )
    raw = con.execute("SELECT count(*) FROM origination_raw WHERE postal_code = '000'").fetchone()
    assert raw is not None and raw[0] > 0, "fixture no longer exercises the 000 sentinel"


@pytest.mark.parametrize(
    "column, sentinel",
    [
        ("original_debt_to_income_ratio", "999"),
        ("classic_fico", "9999"),
        ("vantagescore_4_0", "9999"),
    ],
)
def test_sentinels_become_null(
    con: duckdb.DuckDBPyConnection, column: str, sentinel: str
) -> None:
    """A DTI of 999 read as a number is a borrower spending 999% of income on debt, and
    it drags every mean and coefficient that touches the column."""
    raw = con.execute(
        f"SELECT count(*) FROM origination_raw WHERE {column} = '{sentinel}'"
    ).fetchone()
    typed = con.execute(f"SELECT count(*) FROM origination WHERE {column} IS NULL").fetchone()
    assert raw is not None and typed is not None
    assert raw[0] > 0, f"fixture no longer exercises the {column} sentinel"
    assert typed[0] == raw[0]


def test_credit_scores_are_in_range_once_sentinels_are_cleared(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """Values outside 300-850 are disclosed as not available, so anything outside that
    range surviving the cast means a sentinel was missed."""
    row = con.execute(
        "SELECT min(classic_fico), max(classic_fico) FROM origination "
        "WHERE classic_fico IS NOT NULL"
    ).fetchone()
    assert row is not None
    assert 300 <= row[0] and row[1] <= 850


def test_delinquency_status_stays_text(con: duckdb.DuckDBPyConnection) -> None:
    """It is alphanumeric: "RA" is REO acquisition. An integer cast would drop those rows
    -- the most severe ones in the file."""
    kinds = con.execute(
        "SELECT DISTINCT typeof(current_loan_delinquency_status) FROM performance"
    ).fetchall()
    assert kinds == [("VARCHAR",)]


def test_month_columns_get_a_date_sibling(con: duckdb.DuckDBPyConnection) -> None:
    """YYYYMM text sorts correctly but cannot be subtracted, and stage 1 needs month
    arithmetic in several places."""
    row = con.execute(
        "SELECT period, period_date, typeof(period_date) FROM performance "
        "ORDER BY period LIMIT 1"
    ).fetchone()
    assert row is not None
    period, as_date, kind = row
    assert kind == "DATE"
    assert as_date.strftime("%Y%m") == period


def test_raw_layer_is_untouched_text(con: duckdb.DuckDBPyConnection) -> None:
    """The point of keeping it: every cast can be checked back against the file."""
    kinds = con.execute(
        "SELECT DISTINCT typeof(original_debt_to_income_ratio) FROM origination_raw"
    ).fetchall()
    assert kinds == [("VARCHAR",)]


def test_ingest_is_idempotent(config_path: Path) -> None:
    config = load_config(config_path)
    with connect(config.database) as connection:
        first = ingest(config, connection)
        second = ingest(config, connection)
        assert first.origination_rows == second.origination_rows


def test_unpack_extracts_zips_in_place(project: Path) -> None:
    """The download arrives as sample_YYYY.zip, so ingest has to cope with archives."""
    raw = project / "raw"
    for vintage in (2000, 2001):
        with zipfile.ZipFile(raw / f"sample_{vintage}.zip", "w") as zf:
            for kind in ("orig", "perf"):
                zf.write(raw / f"sample_{kind}_{vintage}.txt", f"sample_{kind}_{vintage}.txt")
        (raw / f"sample_orig_{vintage}.txt").unlink()
        (raw / f"sample_perf_{vintage}.txt").unlink()

    config = load_config(write_config(project, raw, project / "interim"))
    assert len(unpack(config)) == 4
    assert len(discover(config)) == 4


def test_missing_files_name_the_vintage_and_the_portal(project: Path) -> None:
    for path in (project / "raw").glob("*2001*"):
        path.unlink()
    config = load_config(write_config(project, project / "raw", project / "interim"))
    with pytest.raises(FileNotFoundError) as caught:
        discover(config)
    message = str(caught.value)
    assert "sample_orig_2001.txt" in message
    assert "claritydownload" in message
