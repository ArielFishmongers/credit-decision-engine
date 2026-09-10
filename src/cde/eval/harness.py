"""The B0-B4 ablation: where the value actually comes from.

Stage 5, and the single most informative table in the project. Each arm adds exactly one
thing to the one before it, so the gaps decompose the total.

======  ==================================================  ====================================
arm     what it is                                          the gap to the arm above isolates
======  ==================================================  ====================================
B0      approve every applicant                             the floor
B1      binary 36-month classifier, accuracy-maximising     the typical public notebook
B2      the same classifier, profit-maximising              the decision layer alone
B3      discrete-time hazard, profit-maximising             the survival framing (claim 1)
B4a     hazard recalibrated on the in-time holdout          calibration, honestly available
B4b     hazard recalibrated on the first test vintage       calibration, upper bound
======  ==================================================  ====================================

## Everything is held fixed except the thing under test

One population -- the eligible 2007-2008 loans -- for all six arms, because the 2006
cohort is spent fitting B4b's calibrator and an ablation whose arms are scored on
different books measures the books. One set of economics, so every expected NPV uses the
same LGD, discount rate, servicing cost and recovery lag. One training sample: the binary
classifier sees the identical 80% of the training vintages the hazard was fitted on, with
the identical covariates and encoding.

## Decisions come from predictions; scores come from cash

Every arm chooses its cutoff from information available at the decision point, and every
arm is then scored on the *same* realised cash flows. This is not a stylistic preference.
§5.5 originally specified the ablation on expected NPV, and stage 3 measured what that
would have done: the uncalibrated hazard values the book $1,349 a loan above the
recalibrated one, so B3 would beat B4 by construction and the ablation would have ranked
overconfidence first. See :mod:`cde.economics.realised`.

## Two things reported that the spec does not ask for

**An oracle per arm, and each arm's capture of it.** The realised profit of an arm mixes
two things: how well its score *ranks* applicants, and whether its cutoff is in the right
place. The oracle -- the best point on that arm's own ranking, chosen with hindsight --
separates them. An arm with a good ranking and a badly placed cutoff looks identical in
the headline to one with a poor ranking and a perfect cutoff, and they call for opposite
fixes.

**B4 split in two.** Stage 3 found in-time recalibration worth essentially nothing, so a
single B4 fitted honestly would sit on top of B3 and the ablation's most interesting cell
would read as an empty result rather than as a finding. B4a is what a lender could
actually have run; B4b saw 2006 outcomes and is an upper bound. Both are reported because
the gap between them is the point.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cde.calibration.calibrate import (
    Calibrator,
    brier_score,
    expected_calibration_error,
    reliability,
)
from cde.config import Config
from cde.economics.npv import loan_npv
from cde.economics.realised import measure as measure_realised
from cde.eval.calibration import eligible_for_npv, load_loans, projected_paths
from cde.models.binary import TARGET, BinaryModel, implied_monthly_hazard
from cde.models.hazard import HazardModels

#: Arm key -> (label, what the gap above it isolates).
ARMS: dict[str, tuple[str, str]] = {
    "B0": ("approve all", "the floor"),
    "B1": ("binary 36m, accuracy-max", "the typical public notebook"),
    "B2": ("binary 36m, profit-max", "the decision layer alone"),
    "B3": ("hazard, profit-max", "the survival framing (claim 1)"),
    "B4a": ("hazard + in-time recalibration", "calibration, honestly available"),
    "B4b": ("hazard + sequential recalibration", "calibration, upper bound"),
}


@dataclass(frozen=True)
class Ablation:
    """The ablation table, plus the per-arm sweep curves behind it."""

    arms: pd.DataFrame
    curves: dict[str, pd.DataFrame]
    population: int
    binary_horizon: int
    #: One row per applicant: every arm's score and expected NPV, plus the realised
    #: cash. Kept so the failure analysis runs without re-projecting.
    detail: pd.DataFrame | None = None

    def summary(self) -> str:
        floor = float(self.arms.loc[self.arms["arm"] == "B0", "realised"].iloc[0])
        lines = [
            f"B0-B4 ablation on {self.population:,} out-of-time applicants "
            f"(binary target at {self.binary_horizon} months)",
            "",
            f"  {'arm':<5} {'approves':>9} {'realised':>10} {'vs B0':>9} {'step':>8} "
            f"{'expected':>10} {'AUC':>6} {'cap/cut':>8} {'cap/rank':>9}  isolates",
        ]
        previous = floor
        for row in self.arms.itertuples():
            step = row.realised - previous
            lines.append(
                f"  {row.arm:<5} {row.approval_rate:>8.1%} {row.realised:>10,.0f} "
                f"{row.realised - floor:>9,.0f} {step:>8,.0f} "
                f"{row.expected:>10,.0f} {row.auc:>6.3f} {row.capture_cut:>7.1%} "
                f"{row.capture_rank:>8.1%}  {ARMS[row.arm][1]}"
            )
            previous = row.realised
        lines += [
            "",
            "  realised = realised NPV per applicant, in dollars, over EVERY applicant",
            "  expected = the arm's own forecast, shown to expose the gap, never scored on",
            "  cap/cut  = share of the best cutoff on the quantity the arm thresholds.",
            "             Bounded by 100%; a low value means the cutoff is misplaced",
            "  cap/rank = share of the best cutoff on the RISK RANKING alone. Can exceed",
            "             100%, and that is a result: approving on expected NPV also uses",
            "             the loan's balance, rate and term, so the approved set is not a",
            "             prefix of the risk order and can beat it",
        ]
        return "\n".join(lines)


def _score_metrics(
    score: np.ndarray, defaulted: np.ndarray, resolved: np.ndarray, bins: int
) -> dict[str, float]:
    """Discrimination and calibration of one arm's score against the binary outcome.

    Restricted to resolved loans. AUC is reported with Gini alongside, since credit teams
    quote ``2*AUC - 1``, and the two carry identical information.
    """
    usable_score = score[resolved]
    usable_outcome = defaulted[resolved].astype(float)
    if usable_outcome.sum() == 0 or usable_outcome.sum() == usable_outcome.size:
        return {"auc": float("nan"), "gini": float("nan"), "brier": float("nan"),
                "ece": float("nan")}
    auc = float(roc_auc_score(usable_outcome, usable_score))
    # The score is a probability of default over the same window as the outcome, so a
    # Brier score and an ECE are meaningful. For B0 there is no score, and the caller
    # passes the base rate, which makes these read as the no-skill baseline.
    table = reliability(np.clip(usable_score, 0.0, 1.0), usable_outcome, bins=bins)
    return {
        "auc": auc,
        "gini": 2.0 * auc - 1.0,
        "brier": brier_score(np.clip(usable_score, 0.0, 1.0), usable_outcome),
        "ece": expected_calibration_error(table),
    }


def _best_prefix(realised: np.ndarray, ranking: np.ndarray) -> float:
    """Best realised total obtainable by approving a prefix of ``ranking``, per applicant.

    Hindsight, so a bound rather than a rule.
    """
    order = np.argsort(ranking, kind="stable")
    cumulative = np.concatenate([[0.0], np.cumsum(realised[order])])
    return float(cumulative.max() / realised.size)


def _arm_row(
    key: str,
    *,
    approved: np.ndarray,
    realised: np.ndarray,
    expected: np.ndarray,
    score: np.ndarray,
    decision_value: np.ndarray,
    defaulted: np.ndarray,
    resolved: np.ndarray,
    threshold: float,
    bins: int,
) -> dict[str, float | str]:
    """Realised profit, the arm's own forecast, and two oracles that mean different things.

    ``capture_rank`` measures the arm against the best cutoff on its **risk ranking** —
    approve the safest k applicants. ``capture_cut`` measures it against the best cutoff
    on the quantity it **actually thresholds**, which for every profit-maximising arm is
    expected NPV rather than risk.

    The two differ, and the difference is a result rather than bookkeeping. An arm's
    approval set is ``expected NPV > 0``, and that is *not* a prefix of the risk
    ranking: expected NPV also depends on the loan's balance, rate and term, so a
    slightly riskier loan can be worth writing while a safer one is not. An arm can
    therefore **beat the risk-ranking oracle**, and B4b does — which is direct evidence
    that pricing on value uses information that ranking on risk discards.

    Only ``capture_cut`` is bounded by 1. Reporting a single "capture" that could exceed
    100% while being described as a share is the kind of number that quietly discredits a
    table, so both are carried.
    """
    applicants = realised.size
    floor_value = float(realised.sum() / applicants)
    achieved = float(realised[approved].sum() / applicants)
    oracle_rank = _best_prefix(realised, score)
    # Descending, because a higher expected NPV is a better applicant.
    oracle_cut = _best_prefix(realised, -decision_value)

    def _share(oracle: float) -> float:
        headroom = oracle - floor_value
        return (achieved - floor_value) / headroom if headroom > 1e-9 else float("nan")

    return {
        "arm": key,
        "label": ARMS[key][0],
        "approval_rate": float(approved.mean()),
        "threshold": threshold,
        "realised": achieved,
        "expected": float(expected[approved].sum() / applicants),
        "oracle_rank": oracle_rank,
        "oracle_cut": oracle_cut,
        "capture_rank": _share(oracle_rank),
        "capture_cut": _share(oracle_cut),
        **_score_metrics(score, defaulted, resolved, bins),
    }


def _project(
    config: Config,
    models: HazardModels,
    loans: pd.DataFrame,
    *,
    calibrator: Calibrator | None,
) -> pd.DataFrame:
    """Per-loan 36-month cumulative incidence and expected NPV under one hazard variant."""
    window = config.decision.binary_horizon_months
    scores: list[pd.Series] = []
    values: list[pd.DataFrame] = []
    for terms, paths in projected_paths(config, models, loans, transform=calibrator):
        at_window = paths[paths["loan_age"] == window - 1]
        scores.append(at_window.set_index("loan_identifier")["cif_default"])
        values.append(loan_npv(config, terms, paths).frame[["loan_identifier", "npv"]])
    out = pd.DataFrame({"score": pd.concat(scores)}).rename_axis("loan_identifier")
    return out.join(
        pd.concat(values, ignore_index=True).set_index("loan_identifier")["npv"]
    ).rename(columns={"npv": "expected"})


def _flat_hazard_npv(
    config: Config,
    models: HazardModels,
    loans: pd.DataFrame,
    probability: pd.Series,
    prepayment_rate: float,
) -> pd.DataFrame:
    """Expected NPV for B2: the binary probability spread evenly over the window.

    The binary classifier says a loan has a 6% chance of defaulting sometime in three
    years and says nothing about when. Something must fill that gap before a cash flow
    can be discounted, and a constant hazard is the neutral choice. Prepayment gets the
    portfolio-average monthly rate — a single number, because a binary default classifier
    carries no prepayment information at all.

    This is what a competent analyst with a binary model and an NPV would build, not a
    straw man, and the B2-to-B3 gap is therefore the value of knowing the shape rather
    than the value of having any model at all.
    """
    window = config.decision.binary_horizon_months
    horizon = config.horizon.max_loan_age_months
    monthly = implied_monthly_hazard(probability.to_numpy(dtype=float), window)
    frame = pd.DataFrame(
        {
            "loan_identifier": np.repeat(probability.index.to_numpy(), horizon),
            "loan_age": np.tile(np.arange(horizon, dtype=np.int32), len(probability)),
            "flat_default": np.repeat(monthly, horizon),
        }
    )
    hazards = {
        "default": frame["flat_default"].to_numpy(dtype=float),
        "prepayment": np.full(len(frame), prepayment_rate, dtype=float),
    }
    paths = models.survival(frame, hazards)
    # rename_axis before reset_index, deliberately: `probability` is indexed by a bare
    # numpy array of identifiers, so its index carries no name and reset_index() would
    # produce a column called "index". loan_npv then rejects the frame for missing
    # loan_identifier -- which it did, after three expensive projections had already run.
    terms = (
        loans.set_index("loan_identifier")
        .loc[
            probability.index,
            ["original_upb", "original_interest_rate", "original_loan_term",
             "discount_rate"],
        ]
        .rename_axis("loan_identifier")
        .reset_index()
    )
    return loan_npv(config, terms, paths).frame[["loan_identifier", "npv"]].rename(
        columns={"npv": "expected"}
    )


def build(
    config: Config,
    models: HazardModels,
    binary: BinaryModel,
    con: duckdb.DuckDBPyConnection,
    *,
    vintages: tuple[int, ...],
    accuracy_threshold: float,
    in_time: Calibrator | None,
    sequential: Calibrator | None,
    prepayment_rate: float,
    progress: bool = True,
) -> Ablation:
    """Run every arm on one population and one set of economics."""
    from cde.models.binary import loan_level_frame

    bins = config.calibration.bins
    loans = eligible_for_npv(load_loans(config, con, vintages=vintages))

    outcome = loan_level_frame(config, con, vintages).set_index("loan_identifier")
    realised_frame = measure_realised(config, con, vintages).frame.set_index("loan_identifier")

    variants: dict[str, pd.DataFrame] = {}
    for key, calibrator in (("B3", None), ("B4a", in_time), ("B4b", sequential)):
        if key != "B3" and calibrator is None:
            continue
        if progress:
            print(f"  projecting {key}", flush=True)
        variants[key] = _project(config, models, loans, calibrator=calibrator)

    if progress:
        print("  scoring the binary arms", flush=True)
    binary_p = pd.Series(
        binary.probability(loans), index=loans["loan_identifier"].to_numpy(), name="score"
    )
    b2 = _flat_hazard_npv(config, models, loans, binary_p, prepayment_rate).set_index(
        "loan_identifier"
    )

    # One index for every arm, so no arm is scored on a population another was not.
    shared = realised_frame.index
    for table in variants.values():
        shared = shared.intersection(table.index)
    shared = shared.intersection(b2.index).intersection(outcome.index)
    shared = pd.Index(sorted(shared))

    realised = realised_frame.loc[shared, "npv"].to_numpy(dtype=float)
    defaulted = outcome.loc[shared, TARGET].to_numpy(dtype=bool)
    resolved = outcome.loc[shared, "resolved"].to_numpy(dtype=bool)
    base_rate = float(defaulted[resolved].mean())
    approve_all = np.ones(len(shared), dtype=bool)

    records: list[dict[str, float | str]] = [
        _arm_row(
            "B0",
            approved=approve_all,
            realised=realised,
            expected=np.zeros(len(shared)),
            # B0 has no model, so its "score" is the base rate for everybody. That makes
            # its AUC exactly 0.5 and its ECE the calibration of a constant forecast,
            # which is the correct no-skill reading rather than a missing value.
            score=np.full(len(shared), base_rate),
            decision_value=np.zeros(len(shared)),
            defaulted=defaulted,
            resolved=resolved,
            threshold=1.0,
            bins=bins,
        )
    ]

    binary_score = binary_p.loc[shared].to_numpy(dtype=float)
    b2_expected = b2.loc[shared, "expected"].to_numpy(dtype=float)
    records.append(
        _arm_row(
            "B1",
            approved=binary_score < accuracy_threshold,
            realised=realised,
            expected=b2_expected,
            score=binary_score,
            # B1 thresholds the score itself, so its two oracles coincide.
            decision_value=-binary_score,
            defaulted=defaulted,
            resolved=resolved,
            threshold=accuracy_threshold,
            bins=bins,
        )
    )
    records.append(
        _arm_row(
            "B2",
            approved=b2_expected > 0.0,
            realised=realised,
            expected=b2_expected,
            score=binary_score,
            decision_value=b2_expected,
            defaulted=defaulted,
            resolved=resolved,
            threshold=float("nan"),
            bins=bins,
        )
    )
    for key, table in variants.items():
        expected = table.loc[shared, "expected"].to_numpy(dtype=float)
        records.append(
            _arm_row(
                key,
                approved=expected > 0.0,
                realised=realised,
                expected=expected,
                score=table.loc[shared, "score"].to_numpy(dtype=float),
                decision_value=expected,
                defaulted=defaulted,
                resolved=resolved,
                threshold=float("nan"),
                bins=bins,
            )
        )

    order = {key: i for i, key in enumerate(ARMS)}
    arms = pd.DataFrame.from_records(records)
    arms = arms.sort_values("arm", key=lambda s: s.map(order)).reset_index(drop=True)
    detail = pd.DataFrame(
        {
            "loan_identifier": shared,
            "origination_vintage": outcome.loc[shared, "origination_vintage"].to_numpy(),
            "realised": realised,
            "defaulted": defaulted,
            "resolved": resolved,
            "score_binary": binary_score,
            "expected_B2": b2_expected,
        }
    )
    for key, table in variants.items():
        detail[f"score_{key}"] = table.loc[shared, "score"].to_numpy(dtype=float)
        detail[f"expected_{key}"] = table.loc[shared, "expected"].to_numpy(dtype=float)
    return Ablation(
        arms=arms,
        curves={},
        population=len(shared),
        binary_horizon=config.decision.binary_horizon_months,
        detail=detail,
    )


def accuracy_maximising_threshold(
    score: np.ndarray, defaulted: np.ndarray, resolved: np.ndarray
) -> tuple[float, float]:
    """The threshold on ``score`` that maximises accuracy, and the accuracy there.

    Returns a threshold to be used as "approve if score < threshold". Found by walking
    every candidate cut of the sorted scores rather than fixing 0.5, so the arm gets the
    best accuracy actually available to it and cannot be accused of a rigged comparison.

    **Chosen on whichever sample is passed in, which must not be the sample the arm's
    profit is reported from.** The caller passes the in-time holdout. Picking it on the
    test book and then quoting test profit at it is choosing the answer and reporting it.

    On rare-event data the result is degenerate and that is the finding, not a bug:
    accuracy is ``(TN + TP) / n``, true negatives dominate at a low base rate, so every
    applicant declined trades a certain true negative for a probable false positive and
    accuracy falls monotonically as approval falls. The maximum sits at full approval.
    """
    usable = np.asarray(resolved, dtype=bool)
    values = np.asarray(score, dtype=float)[usable]
    outcome = np.asarray(defaulted, dtype=bool)[usable]
    if values.size == 0:
        raise ValueError("no resolved loans to choose a threshold on")
    order = np.argsort(values, kind="stable")
    sorted_outcome = outcome[order]
    total = sorted_outcome.size
    total_defaults = int(sorted_outcome.sum())
    # Approving the k lowest scores: correct = approved non-defaults + declined defaults.
    approved_defaults = np.concatenate([[0], np.cumsum(sorted_outcome)])
    approved = np.arange(total + 1)
    correct = (approved - approved_defaults) + (total_defaults - approved_defaults)
    accuracy = correct / total
    best = int(accuracy.argmax())
    sorted_values = values[order]
    # A threshold strictly above the last approved score, so "score < threshold" admits
    # exactly the intended k loans and nothing more.
    if best >= total:
        threshold = float(np.nextafter(sorted_values[-1], np.inf))
    else:
        threshold = float(sorted_values[best])
    return threshold, float(accuracy[best])


def failure_analysis(
    detail: pd.DataFrame, arm: str = "B3", *, deciles: int = 10
) -> dict[str, pd.DataFrame]:
    """Where the model is wrong, broken out by cohort and by its own risk ranking.

    Two cuts, because they answer different questions and the headline answers neither.

    **By vintage** names which cohorts break it. A model wrong by a similar margin
    everywhere is mis-specified; one wrong on particular cohorts has met a calendar
    effect its origination covariates cannot represent, which is the age-period-cohort
    problem and has no fix inside this feature set.

    **By its own predicted-risk decile** separates a level error from a shape error. A
    constant expected-minus-realised gap across deciles is a level error and
    recalibration addresses it; a gap that varies with risk is a shape error, and a
    two-parameter monotone map cannot fix that however well it is fitted.

    ``gap`` is expected minus realised NPV per loan, so a positive number means the model
    expected more value than arrived -- the direction that loses money.
    """
    expected, score = f"expected_{arm}", f"score_{arm}"
    if expected not in detail.columns:
        raise KeyError(f"detail has no {expected!r}; arm {arm!r} was not run")

    by_vintage = (
        detail.groupby("origination_vintage")
        .agg(
            loans=("realised", "size"),
            expected_npv=(expected, "mean"),
            realised_npv=("realised", "mean"),
            default_rate=("defaulted", "mean"),
        )
        .reset_index()
    )
    by_vintage["gap"] = by_vintage["expected_npv"] - by_vintage["realised_npv"]

    working = detail.copy()
    # duplicates="drop" because tied scores would otherwise raise rather than merge the
    # affected bin edges.
    working["decile"] = pd.qcut(working[score], deciles, labels=False, duplicates="drop")
    by_risk = (
        working.groupby("decile")
        .agg(
            loans=("realised", "size"),
            mean_score=(score, "mean"),
            expected_npv=(expected, "mean"),
            realised_npv=("realised", "mean"),
            default_rate=("defaulted", "mean"),
        )
        .reset_index()
    )
    by_risk["gap"] = by_risk["expected_npv"] - by_risk["realised_npv"]
    return {"by_vintage": by_vintage, "by_risk": by_risk}
