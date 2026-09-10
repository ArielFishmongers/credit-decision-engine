"""Construct the person-period table: one row per loan per month at risk.

Stage 1. The performance file is already in this shape — LOAN AGE is the time index —
so this is a filter and a classification, not an inference. The SQL lives in
``sql/person_period.sql`` and carries the reasoning; this module substitutes the
configured assumptions into it and reports what came out.

Config is substituted with :class:`string.Template`, not f-strings or ``str.format``,
because the SQL contains no ``$`` of its own but does contain braces, and Template
leaves everything it does not recognise alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from string import Template

import duckdb

from cde.config import Config
from cde.data.schema import DELINQUENCY_REO, ZERO_BALANCE_PREPAID

#: Exit reasons, exhaustive and mutually exclusive. Order is presentation order:
#: the two that end a loan's life, then the two that merely end our sight of it,
#: then the one that is a censoring we know to be informative.
EXIT_REASONS = (
    "default",
    "prepaid",
    "administrative_censor",
    "data_censor",
    "informative_exit",
    "clock_reset_censor",
    "month_gap_censor",
)


@dataclass(frozen=True)
class PersonPeriodReport:
    """Shape and integrity of the constructed table."""

    cohort_loans: int
    loans_without_eligible_months: int
    rows: int
    loans: int
    events: int
    exit_mix: dict[str, int]
    mean_months_at_risk: float
    left_truncated_loans: int
    seasoned_loans: int
    off_clock_loans: int
    clock_reset_loans: int
    clock_reset_within_horizon: int
    month_gap_loans: int
    off_clock_excluded: bool
    entered_after_horizon: int
    credit_event_before_delinquency: int
    loans_missing_origination: int
    loans_missing_performance: int
    excluded_loans: int
    mode: str

    def summary(self) -> str:
        lines = [
            f"person_period: {self.rows:,} rows, {self.loans:,} loans, "
            f"{self.events:,} default events  (mode: {self.mode})",
            f"  mean months at risk : {self.mean_months_at_risk:.1f}",
            "  exit mix:",
        ]
        for reason in EXIT_REASONS:
            count = self.exit_mix.get(reason, 0)
            share = count / self.loans if self.loans else 0.0
            lines.append(f"    {reason:<24} {count:>10,}  {share:6.2%}")
        lines.append("  cohort accounting:")
        accounted = (
            self.cohort_loans - self.loans_without_eligible_months - self.excluded_loans
        )
        for label, value in (
            ("loans in the origination cohort", self.cohort_loans),
            ("less: no month inside the horizon", -self.loans_without_eligible_months),
            ("less: excluded as off-clock", -self.excluded_loans),
            ("= loans in person_period", self.loans),
        ):
            lines.append(f"    {label:<38} {value:>10,}")
        verdict = "yes" if accounted == self.loans else "NO -- INVESTIGATE"
        lines.append(f"    {'reconciles':<38} {verdict:>10}")
        lines.append("  integrity:")
        excluded = "excluded" if self.off_clock_excluded else "INCLUDED"
        for label, value in (
            ("entered after age 0 (acquisition lag)", self.left_truncated_loans),
            ("  of which genuinely seasoned (age 3+)", self.seasoned_loans),
            ("entered after the horizon", self.entered_after_horizon),
            (f"off-clock loans ({excluded})", self.off_clock_loans),
            ("clock reset at any point", self.clock_reset_loans),
            ("  of which within the horizon", self.clock_reset_within_horizon),
            ("missing a reporting month", self.month_gap_loans),
            ("credit event before 90+ DPD", self.credit_event_before_delinquency),
            ("in performance, not origination", self.loans_missing_origination),
            ("in origination, not performance", self.loans_missing_performance),
        ):
            lines.append(f"    {label:<38} {value:>10,}")
        return "\n".join(lines)


def _sql(config: Config) -> str:
    """Render ``person_period.sql`` against the configuration."""
    definition = config.default_definition
    template = Template(
        resources.files("cde.features.sql").joinpath("person_period.sql").read_text()
    )

    def codes(values: tuple[str, ...]) -> str:
        return "(" + ", ".join(f"'{value}'" for value in values) + ")"

    if definition.mode == "delinquency":
        # A charge-off is a default even in the rare case where the delinquency column
        # never showed 90+ DPD first, so take whichever clock fires earlier.
        default_age = "least(first_delinquent_age, credit_event_age)"
    else:
        default_age = "credit_event_age"

    return template.substitute(
        delinquency_threshold=definition.delinquency_threshold_months,
        delinquency_code=definition.delinquency_threshold_code,
        delinquency_reo=DELINQUENCY_REO,
        prepaid_code=ZERO_BALANCE_PREPAID,
        default_codes=codes(definition.zero_balance_default_codes),
        informative_codes=codes(definition.zero_balance_informative_exit_codes),
        horizon=config.horizon.max_loan_age_months,
        default_age_expression=default_age,
        off_clock_filter=(
            "AND NOT e.off_clock" if config.person_period.exclude_off_clock_loans else ""
        ),
    )


def build(config: Config, con: duckdb.DuckDBPyConnection) -> PersonPeriodReport:
    """Build ``performance_classified``, ``loan_exit`` and ``person_period``.

    Requires :func:`cde.data.ingest.ingest` to have run against the same connection.
    """
    con.execute(_sql(config))
    return report(config, con)


def _scalar(con: duckdb.DuckDBPyConnection, query: str) -> int:
    row = con.execute(query).fetchone()
    assert row is not None
    return int(row[0] or 0)


def report(config: Config, con: duckdb.DuckDBPyConnection) -> PersonPeriodReport:
    """Measure the constructed table. Every number here is a stage 1 deliverable."""
    shape = con.execute(
        "SELECT count(*), count(DISTINCT loan_identifier), "
        "count(*) FILTER (WHERE default_event) FROM person_period"
    ).fetchone()
    assert shape is not None

    exit_mix = dict(
        con.execute(
            "SELECT exit_reason, count(*) FROM loan_exit GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
    )
    mean_row = con.execute(
        "SELECT avg(months) FROM (SELECT count(*) AS months FROM person_period "
        "GROUP BY loan_identifier)"
    ).fetchone()
    assert mean_row is not None

    excluded = (
        _scalar(con, "SELECT count(*) FROM loan_exit WHERE off_clock")
        if config.person_period.exclude_off_clock_loans
        else 0
    )
    return PersonPeriodReport(
        cohort_loans=_scalar(con, "SELECT count(DISTINCT loan_identifier) FROM origination"),
        loans_without_eligible_months=_scalar(
            con,
            "SELECT count(*) FROM (SELECT DISTINCT loan_identifier FROM performance) p "
            "ANTI JOIN (SELECT DISTINCT loan_identifier FROM person_period_eligible) e "
            "USING (loan_identifier)",
        ),
        rows=int(shape[0]),
        loans=int(shape[1]),
        events=int(shape[2]),
        exit_mix={str(k): int(v) for k, v in exit_mix.items()},
        mean_months_at_risk=float(mean_row[0] or 0.0),
        left_truncated_loans=_scalar(
            con, "SELECT count(*) FROM loan_exit WHERE first_observed_age > 0"
        ),
        seasoned_loans=_scalar(
            con, "SELECT count(*) FROM loan_exit WHERE first_observed_age >= 3"
        ),
        off_clock_loans=_scalar(con, "SELECT count(*) FROM loan_exit WHERE off_clock"),
        clock_reset_loans=_scalar(con, "SELECT count(*) FROM loan_exit WHERE ever_clock_reset"),
        clock_reset_within_horizon=_scalar(
            con, "SELECT count(*) FROM loan_exit WHERE clock_reset_within_horizon"
        ),
        month_gap_loans=_scalar(con, "SELECT count(*) FROM loan_exit WHERE ever_month_gap"),
        off_clock_excluded=config.person_period.exclude_off_clock_loans,
        entered_after_horizon=_scalar(
            con, "SELECT count(*) FROM loan_exit WHERE entered_after_horizon"
        ),
        credit_event_before_delinquency=_scalar(
            con, "SELECT count(*) FROM loan_exit WHERE credit_event_preceded_delinquency"
        ),
        loans_missing_origination=_scalar(
            con,
            "SELECT count(*) FROM (SELECT DISTINCT loan_identifier FROM performance) p "
            "ANTI JOIN origination o USING (loan_identifier)",
        ),
        loans_missing_performance=_scalar(
            con,
            "SELECT count(*) FROM origination o "
            "ANTI JOIN (SELECT DISTINCT loan_identifier FROM performance) p "
            "USING (loan_identifier)",
        ),
        excluded_loans=excluded,
        mode=config.default_definition.mode,
    )
