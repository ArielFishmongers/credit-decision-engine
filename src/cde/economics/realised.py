"""What a loan actually paid: realised NPV from the observed cash flows.

The counterpart to :mod:`cde.economics.npv`, and the reason both exist.

## Decisions come from predictions; scores come from cash

:func:`cde.economics.npv.loan_npv` values a loan under the model's own hazards. That is
the right input to a *decision* -- it is all a lender knows at the application -- and it
is useless as an *evaluation*. Using it as one does not merely weaken the experiment, it
inverts it.

The measurement that shows why is already in hand from stage 3. On the 2007-2008 book the
uncalibrated model assesses portfolio value at $3,396 a loan and the honestly recalibrated
one at $2,046. Score each arm on its own expected NPV and the **overconfident model wins
by $1,349**, because a model that under-predicts default believes every loan is
profitable. PROJECT_PLAN.md section 5.5 specifies the B0-B4 ablation on "expected NPV per
applicant"; taken literally that would have rewarded precisely the error stage 3 exists to
expose. So this module supplies the realised side, and §6d records the correction.

## What is and is not assumption here

Almost nothing is modelled. The balance path, the note rate and the realised loss are all
observed monthly, and the cash flows fall straight out of them -- a payoff is the balance
dropping to zero, a curtailment an extra-large drop, capitalised arrears after a
modification a *negative* principal payment.

Three things remain conventions, all shared with the expected side so that they cancel in
any comparison between the two:

* the discount rate, per vintage, from :func:`cde.economics.rates.discount_rate_by_vintage`
* the 35bp servicing cost
* the terminal balance booked at par at the horizon (§8 limitation 10)

## What it still misses, and in which direction

A disposition landing beyond the 120-month horizon carries its loss with it, and 7.5% of
eventual loss dispositions do. Those losses are absent here, so realised NPV is
**optimistic** by that much -- the same censoring, in the same direction, as the LGD
measurement in :mod:`cde.economics.lgd`. Reported by :attr:`RealisedNpv.loss_censoring`
rather than left implicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from string import Template

import duckdb
import pandas as pd

from cde.config import Config
from cde.economics.rates import discount_rate_by_vintage


@dataclass(frozen=True)
class RealisedNpv:
    """Per-loan realised NPV, with the legs it is a sum of."""

    frame: pd.DataFrame
    horizon: int
    #: Loans dropped for never reporting a positive balance -- see :func:`measure`.
    unfunded: int = 0

    @property
    def per_loan(self) -> float:
        return float(self.frame["npv"].mean())

    @property
    def total(self) -> float:
        return float(self.frame["npv"].sum())

    def summary(self) -> str:
        f = self.frame
        principal = float(f["advanced_upb"].sum())
        lines = [
            f"realised NPV over {len(f):,} loans, {self.horizon} months on book",
            f"  {'principal advanced':<26} {-principal:>16,.0f}",
        ]
        for column, label in (
            ("pv_interest", "interest received"),
            ("pv_principal", "principal repaid"),
            ("pv_terminal", "balance at the horizon"),
            ("pv_loss", "realised losses"),
            ("pv_servicing", "servicing cost"),
        ):
            value = float(f[column].sum())
            lines.append(f"  {label:<26} {value:>16,.0f}   {value / principal:>7.2%}")
        lines += [
            f"  {'':<26} {'-' * 16}",
            f"  {'NPV':<26} {self.total:>16,.0f}   {self.total / principal:>7.2%}",
            f"  {'NPV per loan':<26} {self.per_loan:>16,.0f}",
            f"  loans with a realised loss  {int((f['realised_loss'] > 0).sum()):>16,}"
            f"   ({(f['realised_loss'] > 0).mean():.2%})",
            f"  loans still alive at horizon{int((f['terminal_balance'] > 0).sum()):>16,}"
            f"   ({(f['terminal_balance'] > 0).mean():.2%})",
        ]
        if self.unfunded:
            lines.append(
                f"  excluded, never funded    {self.unfunded:>16,}"
                f"   (single zero-balance record; no cash flow to value)"
            )
        return "\n".join(lines)


def _sql(config: Config, vintages: tuple[int, ...]) -> str:
    template = Template(
        resources.files("cde.economics.sql").joinpath("realised_npv.sql").read_text()
    )
    rates = discount_rate_by_vintage(config)
    missing = [v for v in vintages if v not in rates.index]
    if missing:
        raise ValueError(f"no discount rate for vintage(s) {missing}")
    values = ", ".join(f"({int(v)}, {float(rates.loc[v])!r})" for v in vintages)
    return template.substitute(
        vintages="(" + ", ".join(str(int(v)) for v in vintages) + ")",
        horizon=config.horizon.max_loan_age_months,
        servicing_bps=config.economics.servicing_cost_annual_bps,
        discount_rates=values,
    )


def measure(
    config: Config, con: duckdb.DuckDBPyConnection, vintages: tuple[int, ...]
) -> RealisedNpv:
    """Realised NPV for every eligible loan in the given cohorts."""
    frame = con.execute(_sql(config, vintages)).df()
    placeholders = ", ".join(str(int(v)) for v in vintages)
    terms = con.execute(
        f"SELECT loan_identifier, original_upb AS principal, original_interest_rate, "
        f"original_loan_term FROM origination WHERE origination_vintage IN ({placeholders})"
    ).df()
    merged = frame.merge(terms, on="loan_identifier", how="left", validate="1:1")
    if merged["principal"].isna().any():
        raise ValueError(
            f"{int(merged['principal'].isna().sum())} valued loans have no origination "
            f"record; the performance and origination tables disagree on the population"
        )
    # Loans that never report a positive balance cannot be valued: 141 of the 129,449
    # test-book loans (0.109%) carry a single performance record showing zero, so there
    # is no cash flow at all. Subtracting their reported principal books a spurious loss
    # averaging -$217,440 each, which is -$237 a loan across the whole book -- 17% of
    # the realised headline, manufactured by 0.1% of it. Dropped and counted.
    unfunded = int((merged["advanced_upb"] <= 0.0).sum())
    merged = merged[merged["advanced_upb"] > 0.0].reset_index(drop=True)

    # Valued against the cash ACTUALLY ADVANCED, not the reported ORIGINAL UPB.
    #
    # ORIGINAL UPB is rounded to the nearest $1,000 as a disclosure measure, and 260
    # loans sit more than 5% below theirs for reasons the files do not explain -- one by
    # $463,000. Using it as the principal turns each of those into a six-figure phantom
    # loss. The first reported balance is the money that left the lender, the cash flows
    # telescope to exactly it, and the resulting NPV is a real economic quantity rather
    # than one contaminated by a rounding convention.
    #
    # The cost is that the realised and expected sides no longer subtract the same
    # principal: the expected side uses ORIGINAL UPB, because that is the field the model
    # sees. The gap is mean -$347 a loan with about $577 of spread. It is a level shift,
    # identical across every decision rule, so it cancels in any comparison BETWEEN
    # rules -- which is what stages 4 and 5 compare -- and it does not cancel in an
    # absolute realised-against-expected difference, where it must be remembered.
    merged["npv"] = (
        merged["pv_interest"]
        + merged["pv_principal"]
        + merged["pv_terminal"]
        + merged["pv_loss"]
        + merged["pv_servicing"]
        - merged["advanced_upb"]
    )
    merged["npv_per_dollar"] = merged["npv"] / merged["advanced_upb"]
    merged["rounding_gap"] = merged["advanced_upb"] - merged["principal"]
    return RealisedNpv(
        frame=merged, horizon=config.horizon.max_loan_age_months, unfunded=unfunded
    )


def loss_censoring(
    con: duckdb.DuckDBPyConnection, config: Config, vintages: tuple[int, ...]
) -> pd.DataFrame:
    """How much realised loss falls outside the horizon, and is therefore missing.

    Realised NPV values the first ``horizon`` months, so a disposition that concludes
    after them carries its loss out of the calculation. The shortfall is optimistic and
    it is not negligible, so it is measured rather than described.
    """
    placeholders = ", ".join(str(int(v)) for v in vintages)
    horizon = config.horizon.max_loan_age_months
    # The month index and the first observed age must both be computed over the FULL
    # history and only then filtered to the disposition rows. Windowing after a
    # `actual_loss IS NOT NULL` filter makes a loan's single disposition row its own
    # first month, so `first_observed_age = 0` matched nothing and this returned an empty
    # frame -- a silently missing diagnostic rather than a wrong number, but the fix is
    # the same lesson as everywhere else here: sequence first, filter second.
    return con.execute(
        f"""
        WITH sequenced AS (
            SELECT
                loan_identifier,
                actual_loss,
                row_number() OVER (PARTITION BY loan_identifier ORDER BY period_date) - 1
                    AS month_index,
                min(loan_age) OVER (PARTITION BY loan_identifier) AS first_observed_age
            FROM performance
            WHERE origination_vintage IN ({placeholders})
        )
        SELECT
            count(*)                                                AS priced_dispositions,
            count(*) FILTER (WHERE month_index < {horizon})         AS inside_horizon,
            sum(actual_loss)                                        AS total_loss,
            sum(actual_loss) FILTER (WHERE month_index < {horizon}) AS loss_inside_horizon
        FROM sequenced
        WHERE first_observed_age = 0 AND actual_loss IS NOT NULL
        """
    ).df()
