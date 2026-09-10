"""The person-period construction, tested against loans built to have a known answer.

Every case here exercises one way the event date can be got wrong. They are not
hypothetical: each was found in Freddie Mac's own format example or its user guide.
See tests/conftest.py for why the format example cannot be used for this.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pytest

from cde.config import Config, load_config
from cde.data.ingest import connect, ingest
from cde.features.person_period import EXIT_REASONS, build
from tests.conftest import SyntheticLoan


@dataclass(frozen=True)
class Built:
    """A connection with the person-period table built, plus what to expect in it."""

    con: duckdb.DuckDBPyConnection
    expected: list[SyntheticLoan]
    config: Config


@pytest.fixture
def built(synthetic: tuple[Path, list[SyntheticLoan]]) -> Iterator[Built]:
    config_path, expected = synthetic
    config = load_config(config_path)
    con = connect(config.database)
    ingest(config, con)
    build(config, con)
    yield Built(con, expected, config)
    con.close()


def test_every_loan_exits_for_the_expected_reason_at_the_expected_age(
    built: Built,
) -> None:
    con, expected = built.con, built.expected
    actual = {
        loan: (age, reason)
        for loan, age, reason in con.execute(
            "SELECT loan_identifier, exit_age, exit_reason FROM loan_exit"
        ).fetchall()
    }
    for case in expected:
        assert case.loan in actual, case.note
        assert actual[case.loan] == (case.expected_exit_age, case.expected_reason), case.note


def test_exit_reasons_are_exhaustive(
    built: Built,
) -> None:
    """An unclassified exit would be swept into "still alive" and overstate survival."""
    con = built.con
    seen = {row[0] for row in con.execute("SELECT DISTINCT exit_reason FROM loan_exit").fetchall()}
    assert seen <= set(EXIT_REASONS)
    assert seen == set(EXIT_REASONS), "the fixture should exercise every exit reason"


def test_first_passage_not_first_delinquency(
    built: Built,
) -> None:
    """The loan goes 30 days down at age 10 and 90 days down at age 12. The event is 12:
    30 and 60 days past due are not default under a 90+ DPD definition."""
    con, expected = built.con, built.expected
    loan = next(c.loan for c in expected if "first passage" in c.note)
    row = con.execute(
        "SELECT first_delinquent_age, default_age FROM loan_exit WHERE loan_identifier = ?",
        [loan],
    ).fetchone()
    assert row == (12, 12)


def test_reo_acquisition_counts_as_default(
    built: Built,
) -> None:
    """"RA" is not a day count, so try_cast makes it NULL. Caught by name instead --
    otherwise the most severe rows in the file would read as never-delinquent."""
    con, expected = built.con, built.expected
    loan = next(c.loan for c in expected if "RA" in c.note)
    row = con.execute(
        "SELECT exit_age, exit_reason FROM loan_exit WHERE loan_identifier = ?", [loan]
    ).fetchone()
    assert row == (30, "default")


def test_delinquency_beyond_the_horizon_is_censored_not_a_default(
    built: Built,
) -> None:
    """90+ DPD at age 65 with a 60-month censor is a non-event: the model must not be
    credited with foresight past its own horizon."""
    con, expected = built.con, built.expected
    loan = next(c.loan for c in expected if "beyond the horizon" in c.note)
    row = con.execute(
        "SELECT first_delinquent_age, first_delinquent_age_observed, exit_age, exit_reason "
        "FROM loan_exit WHERE loan_identifier = ?",
        [loan],
    ).fetchone()
    # Inside the modelling window there is no event; the full observed history
    # still records where it happened, for reporting only.
    assert row == (None, 65, 60, "administrative_censor")
    events = con.execute(
        "SELECT count(*) FROM person_period WHERE loan_identifier = ? AND default_event", [loan]
    ).fetchone()
    assert events == (0,)


def test_credit_event_without_prior_delinquency_is_still_a_default(
    built: Built,
) -> None:
    con, expected = built.con, built.expected
    loan = next(c.loan for c in expected if "without a prior" in c.note)
    row = con.execute(
        "SELECT exit_reason, credit_event_preceded_delinquency FROM loan_exit "
        "WHERE loan_identifier = ?",
        [loan],
    ).fetchone()
    assert row == ("default", True)


def test_left_truncation_is_recorded_not_assumed_away(
    built: Built,
) -> None:
    """A seasoned loan enters the panel at loan_age > 0, and contributes no rows for the
    months before Freddie Mac acquired it."""
    con, expected = built.con, built.expected
    loan = next(c.loan for c in expected if "left truncated" in c.note)
    row = con.execute(
        "SELECT first_observed_age FROM loan_exit WHERE loan_identifier = ?", [loan]
    ).fetchone()
    assert row == (6,)
    youngest = con.execute(
        "SELECT min(loan_age) FROM person_period WHERE loan_identifier = ?", [loan]
    ).fetchone()
    assert youngest == (6,)


def test_off_clock_loans_are_flagged_by_both_signals(
    built: Built,
) -> None:
    """A loan modified before acquisition enters at age 0 already delinquent. Naively it
    records as "defaulted in month 0", which would teach the hazard model that month 0 is
    near-certain default."""
    con, expected = built.con, built.expected
    loan = next(c.loan for c in expected if c.off_clock)
    row = con.execute(
        "SELECT off_clock, any_delinquency_exceeds_loan_age, entry_term_mismatch, "
        "entry_implied_term, original_loan_term FROM loan_exit WHERE loan_identifier = ?",
        [loan],
    ).fetchone()
    assert row == (True, True, True, 60, 360)


def test_off_clock_loans_are_excluded_from_person_period(
    built: Built,
) -> None:
    con, expected = built.con, built.expected
    loan = next(c.loan for c in expected if c.off_clock)
    rows = con.execute(
        "SELECT count(*) FROM person_period WHERE loan_identifier = ?", [loan]
    ).fetchone()
    assert rows == (0,), "an off-clock loan must not contribute risk-set rows"
    # But it is still visible and counted, not dropped at ingest.
    assert con.execute(
        "SELECT count(*) FROM loan_exit WHERE loan_identifier = ?", [loan]
    ).fetchone() == (1,)


def test_off_clock_exclusion_is_switchable(
    synthetic: tuple[Path, list[SyntheticLoan]], tmp_path: Path
) -> None:
    """Toggling the assumption must change the answer -- that is what makes it
    sensitivity-testable rather than a hidden decision."""
    config_path, expected = synthetic
    text = config_path.read_text().replace(
        "exclude_off_clock_loans: true", "exclude_off_clock_loans: false"
    )
    included = tmp_path / "included.yaml"
    included.write_text(text)

    config = load_config(included)
    loan = next(c.loan for c in expected if c.off_clock)
    with connect(config.database) as con:
        ingest(config, con)
        report = build(config, con)
        rows = con.execute(
            "SELECT count(*) FROM person_period WHERE loan_identifier = ?", [loan]
        ).fetchone()
    assert rows is not None and rows[0] > 0
    assert report.off_clock_loans > 0


def test_default_event_fires_once_and_only_on_the_exit_month(
    built: Built,
) -> None:
    """One event per loan at most. A duplicated event would double-count the numerator
    of every hazard estimate."""
    con = built.con
    worst = con.execute(
        "SELECT max(events) FROM (SELECT count(*) FILTER (WHERE default_event) AS events "
        "FROM person_period GROUP BY loan_identifier)"
    ).fetchone()
    assert worst == (1,) or worst == (0,)
    stray = con.execute(
        "SELECT count(*) FROM person_period WHERE default_event AND NOT is_exit_month"
    ).fetchone()
    assert stray == (0,)


def test_rows_run_from_entry_to_exit_with_no_gaps(
    built: Built,
) -> None:
    """The risk set must be contiguous: a missing month would understate exposure and
    inflate the hazard."""
    con = built.con
    bad = con.execute(
        "SELECT count(*) FROM ("
        "  SELECT loan_identifier, count(*) AS rows, "
        "         max(loan_age) - min(loan_age) + 1 AS span "
        "  FROM person_period GROUP BY loan_identifier"
        ") WHERE rows <> span"
    ).fetchone()
    assert bad == (0,)


def test_no_row_survives_past_the_horizon(
    built: Built,
) -> None:
    """Read the horizon from config rather than hard-coding it.

    This asserted `<= 60` literally, which passed only because the fixture pins 60 --
    so it would not have caught a production horizon regression at all.
    """
    con = built.con
    horizon = built.config.horizon.max_loan_age_months
    row = con.execute("SELECT max(loan_age) FROM person_period").fetchone()
    assert row is not None and row[0] <= horizon


def test_censored_loans_carry_no_event(
    built: Built,
) -> None:
    """Prepayment and censoring are how a loan leaves without defaulting. If any of them
    carried an event, the whole competing-risks discussion would be moot."""
    con = built.con
    leaked = con.execute(
        "SELECT count(*) FROM person_period WHERE default_event AND exit_reason <> 'default'"
    ).fetchone()
    assert leaked == (0,)


def test_report_counts_agree_with_the_table(built: Built) -> None:
    """The report is what gets read; it must not drift from the table it describes."""
    from cde.features.person_period import report as measure

    report = measure(built.config, built.con)
    row = built.con.execute(
        "SELECT count(*), count(DISTINCT loan_identifier) FROM person_period"
    ).fetchone()
    assert row is not None
    assert (report.rows, report.loans) == row
    # loan_exit covers every loan in the panel, including the off-clock one that
    # person_period excludes -- two vintages of the same synthetic book.
    assert sum(report.exit_mix.values()) == len(built.expected) * 2
    assert report.off_clock_loans == 2
    assert report.off_clock_excluded is True
    assert report.left_truncated_loans == 2
