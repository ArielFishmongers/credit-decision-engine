"""Loss given default, measured from realised losses rather than assumed.

Closes open question 2. The SQL in ``sql/lgd.sql`` carries the derivation; this module
runs it and reports.

The headline, and the reason this is not a one-line constant: **default here is first
passage to 90+ days past due, and only some of those loans ever produce a loss.** So the
severity the NPV identity needs is

    E[loss | 90+ DPD]  =  P(loss disposition | 90+ DPD)  ×  E[LGD | disposition]

Both factors are measurable, both worsen sharply out of time, and they compound.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from string import Template

import duckdb
import pandas as pd

from cde.config import Config


@dataclass(frozen=True)
class LgdEstimate:
    """Measured severity for one set of vintages."""

    label: str
    vintages: tuple[int, ...]
    reached_default: int
    reached_loss_disposition: int
    usable_for_severity: int
    probability_of_loss: float
    lgd_at_default_balance: float
    lgd_at_disposition: float
    lgd_median: float
    lgd_p10: float
    lgd_p90: float
    total_loss: float
    total_exposure_at_default: float

    @property
    def effective_lgd(self) -> float:
        """Severity per default event, which is what the NPV identity consumes.

        The product of the two measured factors. This is the number that belongs in
        config, and it is roughly a third of the disposition LGD because nearly
        two-thirds of all 90+ DPD loans never generate a loss at all.
        """
        return self.probability_of_loss * self.lgd_at_default_balance

    @property
    def cure_rate(self) -> float:
        """Share of 90+ DPD loans that never reach a loss-generating disposition.

        "Cure" is loose shorthand: it covers borrowers who caught up, loans modified into
        performance, and loans that left the panel by a route carrying no recorded loss.
        """
        return 1.0 - self.probability_of_loss

    def summary(self) -> str:
        return "\n".join(
            [
                f"{self.label}  vintages {list(self.vintages)}",
                f"  reached 90+ DPD in-horizon          {self.reached_default:>9,}",
                f"  ...of which a loss disposition      {self.reached_loss_disposition:>9,}",
                f"  ...usable for severity              {self.usable_for_severity:>9,}",
                f"  P(loss disposition | 90+ DPD)       {self.probability_of_loss:>9.4f}",
                f"  cure / no-loss rate                 {self.cure_rate:>9.4f}",
                f"  LGD vs default-month balance        {self.lgd_at_default_balance:>9.4f}",
                f"  LGD vs disposition balance          {self.lgd_at_disposition:>9.4f}"
                f"   (conventional basis)",
                f"  LGD median / p10 / p90              {self.lgd_median:>9.4f}"
                f" / {self.lgd_p10:.4f} / {self.lgd_p90:.4f}",
                f"  => EFFECTIVE LGD per default event  {self.effective_lgd:>9.4f}",
                f"  total realised loss                 ${self.total_loss / 1e6:>8,.1f}M",
            ]
        )


def _sql(vintages: tuple[int, ...], name: str = "lgd.sql") -> str:
    template = Template(
        resources.files("cde.economics.sql").joinpath(name).read_text()
    )
    codes = "(" + ", ".join(str(int(v)) for v in vintages) + ")"
    return template.substitute(vintages=codes)


def estimate(
    con: duckdb.DuckDBPyConnection, vintages: tuple[int, ...], label: str = ""
) -> LgdEstimate:
    """Measure severity on the given cohorts."""
    row = con.execute(_sql(vintages)).df().iloc[0]
    return LgdEstimate(
        label=label or f"vintages {vintages[0]}-{vintages[-1]}",
        vintages=vintages,
        reached_default=int(row["reached_default"]),
        reached_loss_disposition=int(row["reached_loss_disposition"]),
        usable_for_severity=int(row["usable_for_severity"]),
        probability_of_loss=float(row["probability_of_loss"]),
        lgd_at_default_balance=float(row["lgd_at_default_balance"]),
        lgd_at_disposition=float(row["lgd_at_disposition"]),
        lgd_median=float(row["lgd_median"]),
        lgd_p10=float(row["lgd_p10"]),
        lgd_p90=float(row["lgd_p90"]),
        total_loss=float(row["total_loss"]),
        total_exposure_at_default=float(row["total_exposure_at_default"]),
    )


def estimate_splits(
    config: Config, con: duckdb.DuckDBPyConnection
) -> dict[str, LgdEstimate]:
    """Measure train and test separately.

    **Only the train figure may be used for scoring.** Applying the realised test-period
    severity to test-period loans would be leakage of exactly the kind stage 5 exists to
    avoid: a lender at the decision point in 2006 knows what 2000-2004 losses looked
    like, not what 2006-2008 losses will look like. The test figure is a *result*.
    """
    return {
        "train": estimate(con, config.data.train_vintages, "train"),
        "test": estimate(con, config.data.test_vintages, "test  (a result, never an input)"),
    }


def by_vintage(config: Config, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Severity per vintage, so the crisis deterioration is visible rather than pooled."""
    records = []
    for vintage in config.data.vintages:
        single = estimate(con, (vintage,), str(vintage))
        records.append(
            {
                "vintage": vintage,
                "reached_default": single.reached_default,
                "reached_loss_disposition": single.reached_loss_disposition,
                "probability_of_loss": single.probability_of_loss,
                "lgd_at_default_balance": single.lgd_at_default_balance,
                "lgd_at_disposition": single.lgd_at_disposition,
                "effective_lgd": single.effective_lgd,
            }
        )
    return pd.DataFrame.from_records(records)


def degradation(splits: dict[str, LgdEstimate]) -> str:
    """State how much worse the test book was, and why it compounds.

    The two factors move together and in the same direction, so the error in expected
    loss is their product, not their average. Combined with the frequency gap this is the
    single clearest statement of what out-of-time degradation costs.
    """
    train, test = splits["train"], splits["test"]
    frequency = test.probability_of_loss / train.probability_of_loss
    severity = test.lgd_at_default_balance / train.lgd_at_default_balance
    return "\n".join(
        [
            "severity degradation, train -> test:",
            f"  fewer loans cure       P(loss|90+DPD) {train.probability_of_loss:.4f}"
            f" -> {test.probability_of_loss:.4f}   ({frequency:.2f}x)",
            f"  losses are deeper      LGD            {train.lgd_at_default_balance:.4f}"
            f" -> {test.lgd_at_default_balance:.4f}   ({severity:.2f}x)",
            f"  they compound          effective LGD  {train.effective_lgd:.4f}"
            f" -> {test.effective_lgd:.4f}   ({test.effective_lgd / train.effective_lgd:.2f}x)",
        ]
    )


# --- what the cure blend costs -------------------------------------------------------


@dataclass(frozen=True)
class CureCost:
    """Value the NPV discards by treating a cured loan as if it had prepaid.

    See PROJECT_PLAN.md section 8 limitation 9. Measured rather than asserted, and
    printed by ``cde lgd``, because a magnitude quoted in a document and computed
    nowhere is exactly the thing that went stale when the horizon moved.
    """

    label: str
    reached_default: int
    had_loss_disposition: int
    cured: int
    mean_cost: float
    median_cost: float
    total_cost: float
    mean_default_age: float
    mean_balance_at_default: float

    @property
    def cost_share_of_balance(self) -> float:
        """The cost as a fraction of the balance outstanding when default hit.

        The number that says whether this limitation is large or small. It is bounded
        by roughly the spread between the note rate and the discount rate: a loan
        paying 90bp over the discount rate for the rest of its life is worth a few
        percent more than par, not tens of percent.
        """
        if self.mean_balance_at_default <= 0.0:
            return 0.0
        return self.mean_cost / self.mean_balance_at_default

    def summary(self) -> str:
        return "\n".join(
            [
                f"cure blend, the cost of it  ({self.label})",
                f"  reached 90+ DPD                     {self.reached_default:>9,}",
                f"  ...cured / no recorded loss         {self.cured:>9,}"
                f"   ({self.cured / self.reached_default:.1%})",
                f"  mean months on book at default      {self.mean_default_age:>9.1f}",
                f"  value discarded per cured loan      ${self.mean_cost:>8,.0f}"
                f"   (median ${self.median_cost:,.0f})",
                f"  ...as a share of the balance        {self.cost_share_of_balance:>9.2%}",
                f"  total across cured loans            ${self.total_cost / 1e6:>8,.1f}M",
                "  The NPV books a cured loan's balance at the default month, which is",
                "  the treatment a PREPAYMENT gets. The loan actually goes on paying, and",
                "  above the discount rate, so this UNDERSTATES value. Conservative, but",
                "  it falls hardest on the loans the model scores as risky. The fix is a",
                "  multi-state model, out of scope for v1.",
            ]
        )


def cure_opportunity_cost(
    config: Config,
    con: duckdb.DuckDBPyConnection,
    vintages: tuple[int, ...],
    label: str = "train",
) -> CureCost:
    """Discounted value the NPV identity discards for each loan that cures.

    For a cured loan the identity books ``B(t)`` at the default month. What the loan is
    actually worth from that month is its remaining payments to the horizon plus the
    balance still outstanding there, all discounted. The difference is the cost.

    Uses the same amortisation and discounting functions as :func:`cde.economics.npv.
    loan_npv`, deliberately: a second implementation of either would drift, and the
    whole point of this figure is that it describes the NPV actually in use.
    """
    import numpy as np

    from cde.economics.npv import monthly_payment, scheduled_balance, to_monthly_discount
    from cde.economics.rates import discount_rate_by_vintage

    frame = con.execute(_sql(vintages, "cure_cost.sql")).df()
    reached = len(frame)
    had_loss = int(frame["had_loss_disposition"].sum())
    cured = frame[~frame["had_loss_disposition"]].reset_index(drop=True)
    if cured.empty:
        return CureCost(label, reached, had_loss, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    horizon = config.horizon.max_loan_age_months
    rates = discount_rate_by_vintage(config)
    monthly = to_monthly_discount(
        cured["origination_vintage"].map(rates).to_numpy(dtype=float)
    )
    principal = cured["original_upb"].to_numpy(dtype=float)
    note = cured["original_interest_rate"].to_numpy(dtype=float) / 100.0
    term = cured["original_loan_term"].to_numpy(dtype=float)
    default_age = cured["default_age"].to_numpy(dtype=float)
    payment = monthly_payment(principal, note, term)

    # Ages laid out as a matrix so this is one vectorised pass rather than a loop over
    # loans: row i covers months 0..horizon-1 and is masked to the months at or after
    # loan i's default.
    ages = np.arange(horizon, dtype=float)[None, :]
    live = ages >= default_age[:, None]
    discount = np.power(1.0 + monthly[:, None], -(ages + 1.0))
    continued = np.sum(payment[:, None] * discount * live, axis=1)
    terminal = scheduled_balance(principal, note, term, float(horizon)) * np.power(
        1.0 + monthly, -float(horizon)
    )
    booked = scheduled_balance(principal, note, term, default_age) * np.power(
        1.0 + monthly, -(default_age + 1.0)
    )
    cost = continued + terminal - booked
    return CureCost(
        label=label,
        reached_default=reached,
        had_loss_disposition=had_loss,
        cured=len(cured),
        mean_cost=float(cost.mean()),
        median_cost=float(np.median(cost)),
        total_cost=float(cost.sum()),
        mean_default_age=float(default_age.mean()),
        mean_balance_at_default=float(
            scheduled_balance(principal, note, term, default_age).mean()
        ),
    )
