"""Turning a probability into an approve-or-decline, and where the cutoff belongs.

Stage 4, and claim 3: **the optimal cutoff is an economic quantity, not a statistical
one.** It sits where the marginal applicant's expected profit reaches zero, not where
accuracy or F1 peaks.

## The experiment holds the score fixed and varies only the criterion

Every rule compared here ranks applicants by the *same* predicted cumulative default
probability from the *same* fitted hazard. Nothing about model quality differs between
them, so the comparison isolates the one thing that does: how the score is turned into a
threshold. Comparing a different model under each criterion is the stage 5 ablation and a
different question — mixing the two would leave no way to say which of the model or the
criterion earned the difference.

## Decisions come from predictions; scores come from cash

The rule for each arm is chosen using only information available at the decision point.
The *evaluation* is realised NPV from :mod:`cde.economics.realised` — the cash the loans
actually paid. Scoring a decision by the expected NPV that made it is circular, and stage
3 measured how badly: the uncalibrated model values the 2007–2008 book $1,349 a loan
*above* the honestly recalibrated one, so an expected-NPV scoreboard ranks overconfidence
first. See :mod:`cde.economics.realised` for the full argument.

## Why accuracy cannot produce a cutoff on this problem

Not an empirical accident but arithmetic. Accuracy is
``(true negatives + true positives) / n``, and at a 36-month default rate of a few
percent, declining an applicant can only improve it if the model is confident enough to
put that applicant above 50% — which on rare-event data essentially never happens. So the
accuracy-maximising threshold approves everybody and is identical to the do-nothing arm.
It is reported precisely because it is degenerate: "our model is 94% accurate" is the most
common claim in a public credit notebook and it describes a decision rule that declines
nobody.

F1 is not degenerate, because it ignores true negatives, so it does produce a real
threshold. It is still the wrong one: it weighs a missed default against a wrongly
declined applicant at an implicit exchange rate of 1:1, and the true exchange rate is the
ratio of a default's loss to a good loan's margin, which on this book is nearer 30:1.

## The oracle arm, and why it is labelled rather than quoted

The sweep also reports the approval rate that maximises *realised* profit. That is
hindsight — it is chosen by looking at the answer — so it is not a rule any lender could
run. It bounds what the ranking could have earned with a perfectly placed cutoff, which
is what makes the gap between it and the economic rule interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from cde.calibration.calibrate import Calibrator
from cde.config import Config
from cde.economics.npv import loan_npv
from cde.economics.realised import measure as measure_realised
from cde.eval.calibration import eligible_for_npv, load_loans, projected_paths
from cde.models.hazard import HazardModels

#: Rules compared, in the order they belong in the table.
RULES = ("approve all", "accuracy-max", "F1-max", "expected NPV > 0", "oracle (hindsight)")


def build_frame(
    config: Config,
    models: HazardModels,
    con: duckdb.DuckDBPyConnection,
    vintages: tuple[int, ...],
    *,
    calibrator: Calibrator | None = None,
    loans: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per applicant: the score, the expected NPV, and what actually happened.

    Columns

    ``score``
        predicted cumulative probability of default within
        ``decision.binary_horizon_months``, from the projected competing-risks paths.
        Higher is worse, so a cutoff declines applicants *above* a threshold.
    ``expected_npv``
        the model's value for the loan at origination, in dollars.
    ``realised_npv``
        the cash the loan actually produced, discounted the same way.
    ``defaulted``
        whether it reached 90+ DPD inside the binary horizon. Only defined for loans
        observed that long, so ``resolved`` marks the ones it can be read for.
    """
    horizon = config.decision.binary_horizon_months
    # `loans` is supplied when the caller needs a subset -- the in-time holdout, where
    # the statistical thresholds are chosen. It must already be restricted to loans
    # observed from origination.
    if loans is None:
        loans = eligible_for_npv(load_loans(config, con, vintages=vintages))

    scores: list[pd.Series] = []
    values: list[pd.DataFrame] = []
    for terms, paths in projected_paths(config, models, loans, transform=calibrator):
        at_horizon = paths[paths["loan_age"] == horizon - 1]
        scores.append(at_horizon.set_index("loan_identifier")["cif_default"])
        values.append(loan_npv(config, terms, paths).frame[["loan_identifier", "npv"]])

    frame = pd.DataFrame({"score": pd.concat(scores)}).rename_axis("loan_identifier")
    frame = frame.join(
        pd.concat(values, ignore_index=True).set_index("loan_identifier")["npv"]
    ).rename(columns={"npv": "expected_npv"})

    realised = measure_realised(config, con, vintages).frame.set_index("loan_identifier")
    frame = frame.join(realised["npv"].rename("realised_npv"), how="inner")

    outcome = loans.set_index("loan_identifier")[["exit_age", "exit_reason"]]
    frame = frame.join(outcome, how="left")
    # A loan is readable at the binary horizon if it was observed right through it, or
    # left by an event before it. Anything censored earlier cannot answer the question
    # and is excluded from the accuracy and F1 figures rather than counted as a
    # non-default, which would understate the rate by the share that stopped reporting.
    frame["resolved"] = (frame["exit_age"] >= horizon - 1) | frame["exit_reason"].isin(
        ["default", "prepaid"]
    )
    frame["defaulted"] = (frame["exit_reason"] == "default") & (frame["exit_age"] < horizon)
    return frame.reset_index()


def sweep(config: Config, frame: pd.DataFrame) -> pd.DataFrame:
    """Approve everyone whose score is below a threshold; report what that book did.

    One row per grid point. ``npv_per_applicant`` divides by *every* applicant, not by
    the approved ones: a lender who declines half the book and earns a fine margin on the
    rest has still only deployed half the capital, and dividing by the approved subset
    would rank a tiny, immaculate portfolio above a large profitable one.
    """
    ordered = frame.sort_values("score", kind="stable").reset_index(drop=True)
    applicants = len(ordered)
    realised = ordered["realised_npv"].to_numpy(dtype=float)
    resolved = ordered["resolved"].to_numpy(dtype=bool)
    defaulted = (ordered["defaulted"].to_numpy(dtype=bool)) & resolved

    # Cumulative sums over the score-ordered book give every threshold at once.
    cumulative_npv = np.concatenate([[0.0], np.cumsum(realised)])
    approved_defaults = np.concatenate([[0], np.cumsum(defaulted)])
    approved_resolved = np.concatenate([[0], np.cumsum(resolved)])
    total_defaults = int(defaulted.sum())
    total_resolved = int(resolved.sum())

    grid = np.unique(
        np.linspace(0, applicants, config.decision.sweep_points + 1).astype(int)
    )
    # Declining a loan is a POSITIVE prediction of default, so at approval count k the
    # declined tail is predicted-positive: true positives are the defaults among them.
    declined_defaults = total_defaults - approved_defaults[grid]
    declined_resolved = total_resolved - approved_resolved[grid]
    predicted_positive = declined_resolved
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted_positive > 0, declined_defaults / predicted_positive, 0.0)
        recall = (
            declined_defaults / total_defaults if total_defaults else np.zeros_like(precision)
        )
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    # Accuracy over the resolved population: approved non-defaults plus declined defaults.
    approved_non_defaults = approved_resolved[grid] - approved_defaults[grid]
    accuracy = (approved_non_defaults + declined_defaults) / total_resolved

    return pd.DataFrame(
        {
            "approved": grid,
            "approval_rate": grid / applicants,
            "threshold": [
                float(ordered["score"].iloc[k - 1]) if k > 0 else 0.0 for k in grid
            ],
            "npv_total": cumulative_npv[grid],
            "npv_per_applicant": cumulative_npv[grid] / applicants,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


@dataclass(frozen=True)
class DecisionResult:
    """The rule comparison that settles claim 3."""

    frame: pd.DataFrame
    curve: pd.DataFrame
    rules: pd.DataFrame
    binary_horizon: int

    def summary(self) -> str:
        lines = [
            f"decision rules on {len(self.frame):,} applicants, binary target at "
            f"{self.binary_horizon} months",
            f"  {'rule':<20} {'approves':>9} {'threshold':>10} "
            f"{'realised NPV/applicant':>23} {'vs approve-all':>15}",
        ]
        baseline = float(
            self.rules.loc[self.rules["rule"] == "approve all", "npv_per_applicant"].iloc[0]
        )
        for row in self.rules.itertuples():
            lines.append(
                f"  {row.rule:<20} {row.approval_rate:>8.1%} {row.threshold:>10.5f} "
                f"{row.npv_per_applicant:>23,.0f} "
                f"{row.npv_per_applicant - baseline:>15,.0f}"
            )
        return "\n".join(lines)


def _at(curve: pd.DataFrame, index: int, rule: str) -> dict[str, float | str]:
    row = curve.iloc[index]
    return {
        "rule": rule,
        "approval_rate": float(row["approval_rate"]),
        "threshold": float(row["threshold"]),
        "npv_per_applicant": float(row["npv_per_applicant"]),
        "accuracy": float(row["accuracy"]),
        "f1": float(row["f1"]),
    }


def rules_table(
    config: Config, frame: pd.DataFrame, curve: pd.DataFrame, chooser: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Locate each rule's cutoff, then read the realised result off the test curve.

    ``chooser`` is the curve the *statistical* thresholds are picked on — the in-time
    holdout, so a threshold is never selected by maximising a metric on the same data its
    profit is reported from. Without it the accuracy- and F1-maximising arms are chosen
    with the answer in hand, which flatters them by an amount nobody can bound.
    """
    picking = chooser if chooser is not None else curve
    records = [_at(curve, len(curve) - 1, "approve all")]
    for rule, column in (("accuracy-max", "accuracy"), ("F1-max", "f1")):
        chosen = picking.iloc[int(picking[column].to_numpy().argmax())]
        # Map the chosen SCORE threshold onto the test curve, rather than the chosen
        # approval rate: the threshold is the rule, and the two books have different
        # score distributions, so carrying the rate across would silently redefine it.
        index = int(np.searchsorted(curve["threshold"].to_numpy(), chosen["threshold"]))
        records.append(_at(curve, min(index, len(curve) - 1), rule))

    # The economic rule: approve exactly those whose expected NPV is positive. No sweep
    # and no tuning — under the model this IS the optimum, because a loan with positive
    # expected value should be written and one with negative expected value should not.
    approved = int((frame["expected_npv"] > 0).sum())
    index = int(np.searchsorted(curve["approved"].to_numpy(), approved))
    records.append(_at(curve, min(index, len(curve) - 1), "expected NPV > 0"))
    records.append(
        _at(curve, int(curve["npv_per_applicant"].to_numpy().argmax()), "oracle (hindsight)")
    )
    return pd.DataFrame.from_records(records)


def evaluate(
    config: Config,
    frame: pd.DataFrame,
    *,
    chooser_frame: pd.DataFrame | None = None,
) -> DecisionResult:
    """Sweep the cutoff and locate every rule on the resulting curve."""
    curve = sweep(config, frame)
    chooser = (
        sweep(config, chooser_frame)
        if chooser_frame is not None and config.decision.choose_thresholds_on_holdout
        else None
    )
    return DecisionResult(
        frame=frame,
        curve=curve,
        rules=rules_table(config, frame, curve, chooser),
        binary_horizon=config.decision.binary_horizon_months,
    )
