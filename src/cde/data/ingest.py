"""Load the Freddie Mac files into DuckDB.

Two layers, deliberately.

``origination_raw`` / ``performance_raw`` hold every column as ``VARCHAR``, exactly as it
appeared in the file. ``origination`` / ``performance`` are views over them that clear the
"not available" sentinels and cast to real types.

The split exists because this dataset punishes type inference. A sniffer turns postal code
``00501`` into the integer ``501``, chokes on ``RA`` in the delinquency column, and reads a
debt-to-income sentinel of ``999`` as a borrower spending 999% of income on debt. Keeping
an untouched text layer means every cast is visible, reversible and auditable against the
file, and :func:`cast_losses` reports anything a cast dropped rather than letting it vanish.

Why DuckDB rather than pandas: the person-period construction and the vintage curves in
stage 1 are window-function problems, the full panel is far larger than memory, and the SQL
is part of the deliverable.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import duckdb

from cde.config import Config
from cde.data.schema import (
    COLUMN_TYPES,
    MONTH_COLUMNS,
    NOT_AVAILABLE_SENTINELS,
    ORIGINATION_NAMES,
    PERFORMANCE_NAMES,
)

#: Origination vintage, added at load time from the file the row came from. The file is
#: authoritative for the cohort; first payment date is one to two months later.
VINTAGE_COLUMN = "origination_vintage"


@dataclass(frozen=True)
class KeyViolation(ValueError):
    """Raised when a primary key is not unique after loading."""


@dataclass(frozen=True)
class SourceFile:
    """One pipe-delimited file to load, and the cohort it belongs to."""

    path: Path
    vintage: int
    kind: str  # "origination" | "performance"


@dataclass(frozen=True)
class IngestReport:
    """What was loaded. Printed by the CLI so a run is self-documenting."""

    files: tuple[SourceFile, ...]
    origination_rows: int
    performance_rows: int
    loans: int
    vintages: tuple[int, ...]

    def summary(self) -> str:
        lines = [
            f"loaded {len(self.files)} file(s) for vintages "
            f"{', '.join(str(v) for v in self.vintages)}",
            f"  origination : {self.origination_rows:>12,} rows  ({self.loans:,} loans)",
            f"  performance : {self.performance_rows:>12,} rows",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Locating and unpacking files
# --------------------------------------------------------------------------------------


def unpack(config: Config) -> list[Path]:
    """Extract any ``sample_YYYY.zip`` in the raw directory, in place.

    Kept as its own step rather than folded into the load: extraction is slow, it only
    needs doing once, and having the ``.txt`` files sitting there makes it obvious what
    the pipeline is actually reading.
    """
    raw_dir = config.data.raw_dir
    extracted: list[Path] = []
    for archive in sorted(raw_dir.glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if not member.endswith(".txt"):
                    continue
                target = raw_dir / Path(member).name
                if not target.exists():
                    target.write_bytes(zf.read(member))
                extracted.append(target)
    return extracted


def discover(config: Config) -> list[SourceFile]:
    """Find the files for the configured vintages, or say precisely what is missing."""
    raw_dir = config.data.raw_dir
    found: list[SourceFile] = []
    missing: list[str] = []

    for vintage in config.data.vintages:
        if config.data.source == "sample":
            wanted = {
                "origination": [f"sample_orig_{vintage}.txt"],
                "performance": [f"sample_perf_{vintage}.txt"],
            }
        else:
            wanted = {
                "origination": [f"orig_{vintage}Q{q}.txt" for q in (1, 2, 3, 4)],
                "performance": [f"perf_{vintage}Q{q}.txt" for q in (1, 2, 3, 4)],
            }
        for kind, names in wanted.items():
            present = [raw_dir / name for name in names if (raw_dir / name).is_file()]
            if not present:
                missing.append(f"{vintage} {kind}: expected one of {names}")
            found.extend(SourceFile(path=path, vintage=vintage, kind=kind) for path in present)

    if missing:
        detail = "\n  ".join(missing)
        raise FileNotFoundError(
            f"missing input files under {raw_dir}/:\n  {detail}\n\n"
            f"The dataset is not downloadable without an account. Register at Clarity Data "
            f"Intelligence (CRT portal, not MBS) and take sample_YYYY.zip for each vintage:\n"
            f"  https://claritydownload.fmapps.freddiemac.com/CRT/#/sflld\n"
            f"Then run `cde ingest`, which unpacks any .zip it finds in {raw_dir}/."
        )
    return found


# --------------------------------------------------------------------------------------
# SQL generation for the raw and typed layers
# --------------------------------------------------------------------------------------


def _quoted(name: str) -> str:
    return f'"{name}"'


def _read_csv(path: Path, names: list[str]) -> str:
    """A ``read_csv`` call pinned to our schema.

    Every option is set explicitly. ``header=false`` because the files have none;
    ``quote=''`` because they use no quoting at all, so a stray ``"`` in a seller name
    would otherwise swallow the rest of the line; ``nullstr=''`` because absence is
    encoded as an empty field. Column count mismatches raise rather than being padded —
    a Release 46 file loaded under a Release 47 schema must fail, not shift every field
    by one.
    """
    columns = ", ".join(f"{_quoted(name)}: 'VARCHAR'" for name in names)
    return (
        f"read_csv('{path.as_posix()}', "
        f"delim = '|', header = false, quote = '', escape = '', nullstr = '', "
        f"columns = {{{columns}}})"
    )


def _cast_expressions(names: list[str]) -> list[str]:
    """Build the SELECT list for a typed view: clear sentinels, then cast.

    ``try_cast`` rather than ``cast`` so one malformed field in 20 million rows does not
    abort a load — but every value it turns into NULL is counted by :func:`cast_losses`,
    so nothing is dropped quietly.
    """
    expressions: list[str] = []
    for name in names:
        sentinel = NOT_AVAILABLE_SENTINELS.get(name)
        source = _quoted(name) if sentinel is None else f"nullif({_quoted(name)}, '{sentinel}')"
        sql_type = COLUMN_TYPES[name]
        if sql_type == "VARCHAR":
            expressions.append(f"{source} AS {_quoted(name)}")
        else:
            expressions.append(f"try_cast({source} AS {sql_type}) AS {_quoted(name)}")
        if name in MONTH_COLUMNS:
            # YYYYMM sorts correctly as text but cannot be subtracted, and stage 1 needs
            # month arithmetic everywhere. Exposed alongside rather than instead of.
            expressions.append(f"try_strptime({source}, '%Y%m')::DATE AS {_quoted(name + '_date')}")
    return expressions


def _create_typed_view(con: duckdb.DuckDBPyConnection, view: str, names: list[str]) -> None:
    select = ",\n    ".join([_quoted(VINTAGE_COLUMN), *_cast_expressions(names)])
    con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT\n    {select}\nFROM {view}_raw")


def _load_raw(
    con: duckdb.DuckDBPyConnection, table: str, names: list[str], files: list[SourceFile]
) -> None:
    declared = ", ".join(f"{_quoted(name)} VARCHAR" for name in names)
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(f"CREATE TABLE {table} ({_quoted(VINTAGE_COLUMN)} SMALLINT, {declared})")
    for source in files:
        con.execute(
            f"INSERT INTO {table} SELECT {source.vintage}, * FROM {_read_csv(source.path, names)}"
        )


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------


def connect(database: Path, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the DuckDB database, creating its directory if needed."""
    database.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(database), read_only=read_only)
    # The progress bar writes ANSI redraws to stdout, which corrupts piped output
    # and log files. Every long step here prints its own summary instead.
    con.execute("SET enable_progress_bar = false")
    return con


def ingest(config: Config, con: duckdb.DuckDBPyConnection | None = None) -> IngestReport:
    """Unpack, load and type the configured vintages. Idempotent — safe to re-run."""
    owned = con is None
    connection = con if con is not None else connect(config.database)
    try:
        unpack(config)
        files = discover(config)
        for kind, table, names in (
            ("origination", "origination", ORIGINATION_NAMES),
            ("performance", "performance", PERFORMANCE_NAMES),
        ):
            selected = [f for f in files if f.kind == kind]
            _load_raw(connection, f"{table}_raw", names, selected)
            _create_typed_view(connection, table, names)
        validate_keys(connection)

        counts = connection.execute(
            "SELECT (SELECT count(*) FROM origination_raw), "
            "(SELECT count(*) FROM performance_raw), "
            "(SELECT count(DISTINCT loan_identifier) FROM origination_raw)"
        ).fetchone()
        assert counts is not None
        return IngestReport(
            files=tuple(files),
            origination_rows=int(counts[0]),
            performance_rows=int(counts[1]),
            loans=int(counts[2]),
            vintages=config.data.vintages,
        )
    finally:
        if owned:
            connection.close()


def validate_keys(con: duckdb.DuckDBPyConnection) -> None:
    """Check the two primary keys the rest of stage 1 silently depends on.

    ``origination`` must be unique on loan, and ``performance`` unique on
    (loan, period). This is not defensive boilerplate: every window function
    downstream partitions by loan and orders by period, so a duplicated key does
    not raise anywhere — it makes ``lag()`` compare a loan against a copy of
    itself, which reads as the loan-age clock running backwards and silently
    flags healthy loans as modified.

    The realistic cause is a loading mistake rather than bad data: the same file
    ingested under two vintages, or overlapping quarters.
    """
    duplicate_loans = con.execute(
        "SELECT count(*) FROM (SELECT loan_identifier FROM origination_raw "
        "GROUP BY 1 HAVING count(*) > 1)"
    ).fetchone()
    assert duplicate_loans is not None
    if duplicate_loans[0]:
        raise KeyViolation(
            f"{duplicate_loans[0]:,} loan identifiers appear more than once in "
            f"origination. Each loan belongs to exactly one vintage file — check for the "
            f"same file loaded twice, or overlapping quarters."
        )

    duplicate_months = con.execute(
        "SELECT count(*) FROM (SELECT loan_identifier, period FROM performance_raw "
        "GROUP BY 1, 2 HAVING count(*) > 1)"
    ).fetchone()
    assert duplicate_months is not None
    if duplicate_months[0]:
        raise KeyViolation(
            f"{duplicate_months[0]:,} (loan, period) pairs appear more than once in "
            f"performance. The window functions in stage 1 partition by loan and order by "
            f"period, so duplicates would corrupt the loan-age reset detection rather than "
            f"raising anywhere."
        )


def cast_losses(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str, int]]:
    """Values that were present in the file but became NULL when cast.

    Returns ``(table, column, count)`` for every column that lost something. This should
    be empty. A non-empty result means either the schema assigns a column the wrong type,
    or the data holds a code the user guide does not document — both worth knowing before
    a number derived from that column reaches a write-up.
    """
    losses: list[tuple[str, str, int]] = []
    for table, names in (("origination", ORIGINATION_NAMES), ("performance", PERFORMANCE_NAMES)):
        for name in names:
            if COLUMN_TYPES[name] == "VARCHAR":
                continue
            sentinel = NOT_AVAILABLE_SENTINELS.get(name)
            raw = _quoted(name)
            cleared = raw if sentinel is None else f"nullif({raw}, '{sentinel}')"
            row = con.execute(
                f"SELECT count(*) FROM {table}_raw "
                f"WHERE {cleared} IS NOT NULL "
                f"AND try_cast({cleared} AS {COLUMN_TYPES[name]}) IS NULL"
            ).fetchone()
            assert row is not None
            if int(row[0]) > 0:
                losses.append((table, name, int(row[0])))
    return losses
