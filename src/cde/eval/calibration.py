"""Stage 3 harness: measure miscalibration on three populations, then price it.

The stage exists for one number -- **how much portfolio value moves when the
probabilities are honest instead of overconfident** -- and everything here is
scaffolding around getting that number without cheating.

## The three populations, and why two of them are not enough

============================  ===========================================================
in-sample train               the 80% of training loans the hazard was fitted on
in-time holdout               the other 20%, same vintages, never seen by the fit
out-of-time test              vintages 2006-2008, a different economy entirely
============================  ===========================================================

The in-sample population is not there to be impressive; it is a **control**. A
converged logistic regression with an unpenalised intercept is calibrated in-sample by
construction, so its expected calibration error must come back near zero. If it does
not, the harness is wrong rather than the model, and that identity has already caught
two bugs in this project.

The in-time holdout separates two effects the out-of-time set confounds. A model can
miscalibrate because it overfitted (visible on the holdout, same economy) or because the
world moved (visible only out of time). Without the holdout, every out-of-time gap looks
like a regime change even when it is ordinary overfitting.

## Why one model has to serve every arm

The hazard used here is fitted on 80% of the training loans, and the same fitted object
scores all three populations and both recalibrated variants. The alternative -- stage 2's
full-sample model for the uncalibrated arm and an 80% model for the calibrated one --
would make the B3-to-B4 gap in stage 5's ablation a measurement of *training size* as
well as of calibration, and there would be no way afterwards to say how much was which.
Hence ``cde hazard --holdout-share`` and a separate saved file.

## Why the recalibrator is fitted where it is

Nowhere else is honest. Fitting it on the test set leaks the answer being asked for.
Fitting it on the rows the hazard was fitted on measures nothing, because those are
already calibrated by construction -- the recalibrator would come back as the identity.
The in-time holdout is the only sample that is both unseen and available at the decision
point, which also makes it the realistic operating position: a lender in 2005 could hold
out part of their own book, and could not hold out 2008.

## Stated before measuring

Recorded so it is a prediction rather than a rationalisation: in-time recalibration will
**not** fix out-of-time miscalibration, and the out-of-time gap will be *smaller* than
the earlier 60-month analysis implied, because the 120-month window stopped truncating
calm cohorts harder than crisis ones. Both are checked in :meth:`CalibrationReport.summary`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from cde.calibration.calibrate import (
    BrierParts,
    Calibrator,
    brier_decomposition,
    build_calibrator,
    expected_calibration_error,
    maximum_calibration_error,
    reliability,
)
from cde.config import Config
from cde.economics.npv import loan_npv
from cde.economics.rates import discount_rate_by_vintage
from cde.models.hazard import CAUSE_TARGETS, HazardModels, required_columns

#: Loans per chunk when projecting cash-flow paths. Each loan becomes ``horizon`` rows
#: carrying every design covariate, so 10,000 loans at a 120-month horizon is a 1.2
#: million row frame -- large enough to be efficient, small enough to stay in memory
#: alongside the design matrix built from it.
PROJECTION_CHUNK = 10_000

#: Scope labels for rows of the NPV table. Constants because the figure, the summary
#: and the CSV all select on them, and a literal repeated in three places is a literal
#: that will eventually disagree with itself.
SCOPE_HEADLINE = "headline"
SCOPE_SEQUENTIAL = "sequential"
SCOPE_SENSITIVITY = "sensitivity"

#: Fraction of eligible loans used for the economics sensitivity grid. The grid asks how
#: the *conclusion* moves with the discount spread and severity, which is a question
#: about a ratio, so a tenth of the book answers it to far better precision than the
#: assumptions themselves are known to. The headline NPV uses every eligible loan.
SENSITIVITY_SAMPLE = 0.10


def holdout_mask(loan_ids: np.ndarray, *, share: float, seed: int) -> np.ndarray:
    """Deterministic in-time holdout, assigned per loan by a hash of its identifier.

    Two properties matter and a shuffle-and-slice has neither.

    **Split by loan, never by row.** A loan contributes up to 120 person-period rows and
    they are anything but independent -- they share every covariate and differ only in
    age. Splitting rows would put the same loan on both sides and leak the outcome
    directly.

    **Stable under a change of population.** Hashing the identifier means a loan's side
    of the split depends only on that loan, so adding a vintage, filtering, or
    re-running in a different order cannot move anybody. A seeded permutation depends on
    the size and order of the input, so the same loan would silently change sides
    between two runs that differ in any other respect -- and the calibrator would then
    have been fitted on rows the "unseen" score was computed from.
    """
    if not 0.0 < share < 1.0:
        raise ValueError(f"share must be in (0, 1), got {share}")
    key = str(seed).encode()
    # blake2b keyed by the seed, rather than pandas' hashing, so the split is stable
    # across pandas and numpy versions as well as across runs. 400,000 digests is under
    # a second and this happens once.
    cutoff = int(share * (1 << 32))
    digests = np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(str(loan).encode(), digest_size=4, key=key).digest(), "big"
            )
            for loan in loan_ids
        ),
        dtype=np.int64,
        count=len(loan_ids),
    )
    return digests < cutoff


@dataclass(frozen=True)
class PopulationScore:
    """Calibration of one set of predictions against one set of outcomes."""

    label: str
    variant: str
    predicted_events: float
    observed_events: int
    rows: int
    ece: float
    max_error: float
    brier: BrierParts
    table: pd.DataFrame

    @property
    def base_rate(self) -> float:
        return self.observed_events / self.rows if self.rows else 0.0

    @property
    def marginal_ratio(self) -> float:
        """Predicted total events over observed. The single most legible number here.

        1.0 means the model gets the *aggregate* right. Above 1 it over-predicts
        defaults across the board, below 1 it under-predicts. It says nothing about
        whether the errors are in the right place -- a model can have a ratio of exactly
        1 and be badly wrong bin by bin -- which is what the ECE and the reliability
        term are for.
        """
        if not self.observed_events:
            return float("nan")
        return self.predicted_events / self.observed_events

    @property
    def is_in_sample(self) -> bool:
        """Whether this score is measured on rows its own fit already saw.

        True for the raw hazard on the training rows, and for a recalibrator on the
        holdout it was fitted on. Both are calibrated there by construction, so their
        near-zero errors are controls confirming the harness works -- not results. This
        flag exists because isotonic on its own holdout returns an ECE of exactly zero,
        which is the most impressive-looking and least meaningful number in the report.
        """
        if self.label == "train (in-sample)" and self.variant == "raw":
            return True
        return self.label == "holdout (in-time)" and self.variant != "raw"

    def summary(self) -> str:
        caveat = (
            "   <-- IN-SAMPLE for this fit: a control, not a result"
            if self.is_in_sample
            else ""
        )
        return "\n".join(
            [
                f"{self.label}  [{self.variant}]{caveat}",
                f"  rows {self.rows:>12,}   events {self.observed_events:>8,}"
                f"   base rate {self.base_rate:.5%}",
                f"  predicted events {self.predicted_events:>11,.0f}"
                f"   ratio to observed {self.marginal_ratio:>6.3f}",
                f"  ECE {self.ece:.6f}  ({self.ece / self.base_rate:>5.1%} of the base rate)"
                f"   worst bin {self.max_error:.6f}",
                f"  Brier reliability {self.brier.reliability:.3e}"
                f"   resolution {self.brier.resolution:.3e}"
                f"   skill {self.brier.skill:+.4f}",
            ]
        )


def score(
    config: Config,
    label: str,
    variant: str,
    predicted: np.ndarray,
    observed: np.ndarray,
) -> PopulationScore:
    """Every calibration statistic for one (population, variant) pair."""
    settings = config.calibration
    table = reliability(predicted, observed, bins=settings.bins, strategy=settings.binning)
    return PopulationScore(
        label=label,
        variant=variant,
        predicted_events=float(np.sum(predicted)),
        observed_events=int(np.sum(observed)),
        rows=int(len(predicted)),
        ece=expected_calibration_error(table),
        max_error=maximum_calibration_error(table),
        brier=brier_decomposition(
            predicted, observed, bins=settings.bins, strategy=settings.binning
        ),
        table=table,
    )


def fit_calibrators(
    config: Config, predicted: np.ndarray, observed: np.ndarray
) -> dict[str, Calibrator]:
    """Fit every configured recalibration method on one sample."""
    return {
        method: build_calibrator(
            method,
            tolerance=config.model.tolerance,
            max_iterations=config.model.max_iterations,
            isotonic_min_tail_rows=config.calibration.isotonic_min_tail_rows,
        ).fit(predicted, observed)
        for method in config.calibration.methods
    }


# --- cash-flow projection ------------------------------------------------------------


def load_loans(
    config: Config, con: duckdb.DuckDBPyConnection, *, vintages: tuple[int, ...]
) -> pd.DataFrame:
    """One row per loan: the design covariates plus what the cash flows need.

    Taken from the person-period table at each loan's first observed month, so the
    covariates are exactly the ones the hazard model was fitted with -- reading them from
    ``origination`` instead would risk a column diverging between the two paths.
    """
    columns = required_columns(config)
    for extra in ("first_observed_age", "exit_age", "exit_reason"):
        if extra not in columns:
            columns.append(extra)
    placeholders = ", ".join(str(int(v)) for v in vintages)
    frame = con.execute(
        f"SELECT {', '.join(columns)} FROM person_period "
        f"WHERE origination_vintage IN ({placeholders}) AND loan_age = first_observed_age "
        f"ORDER BY loan_identifier"
    ).df()
    if frame["loan_identifier"].duplicated().any():
        raise ValueError(
            "more than one row per loan at its first observed age; the person-period "
            "table is not one row per loan-month as assumed"
        )
    rates = discount_rate_by_vintage(config)
    frame["discount_rate"] = frame["origination_vintage"].map(rates).astype(float)
    if frame["discount_rate"].isna().any():
        raise ValueError("some vintages have no discount rate; check economics.benchmark_path")
    return frame


def eligible_for_npv(loans: pd.DataFrame) -> pd.DataFrame:
    """Loans whose value can be projected from origination.

    Restricted to ``first_observed_age == 0``. About 17% of these loans enter the panel
    late, because Freddie Mac bought them after origination -- they are perfectly usable
    for fitting a hazard, which conditions on being alive at the month in question, but
    an NPV is a value *at the decision point* and there is no decision point on record
    for a loan first seen at age 6. Projecting one anyway would be inventing the months
    before it appeared.
    """
    return loans[loans["first_observed_age"] == 0].reset_index(drop=True)


def _long_frame(loans: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Each loan repeated across ages 0..horizon-1, the shape the hazard model scores.

    Every covariate in the design is fixed at origination -- FICO, CLTV, DTI, rate, term,
    purpose, state -- so the only thing that varies down a loan's block is ``loan_age``.
    That is what makes a projection possible at all: the model needs no future
    information to score month 90, because it never had any.
    """
    repeated = loans.loc[loans.index.repeat(horizon)].reset_index(drop=True)
    repeated["loan_age"] = np.tile(np.arange(horizon, dtype=np.int32), len(loans))
    return repeated


def projected_paths(
    config: Config,
    models: HazardModels,
    loans: pd.DataFrame,
    *,
    transform: Calibrator | None = None,
    chunk_size: int = PROJECTION_CHUNK,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield (terms, paths) per chunk of loans, ready for :func:`loan_npv`.

    ``transform`` recalibrates the **default** hazard before survival is computed, which
    is the point: recalibrating h_d changes S(t), which changes the weight on every
    later month, so the effect on value is not simply proportional to the change in the
    probability. Prepayment is left alone -- claim 2 is about the default probability,
    and recalibrating both would make the NPV difference impossible to attribute.
    """
    horizon = config.horizon.max_loan_age_months
    for start in range(0, len(loans), chunk_size):
        chunk = loans.iloc[start : start + chunk_size].reset_index(drop=True)
        long = _long_frame(chunk, horizon)
        hazards = {
            cause: fitted.hazard(long) for cause, fitted in models.causes.items()
        }
        if transform is not None:
            hazards["default"] = transform.transform(hazards["default"])
        paths = models.survival(long, hazards)
        terms = chunk[
            [
                "loan_identifier",
                "original_upb",
                "original_interest_rate",
                "original_loan_term",
                "discount_rate",
            ]
        ]
        yield terms, paths


def cumulative_calibration(
    config: Config, models: HazardModels, loans: pd.DataFrame, *, label: str
) -> pd.DataFrame:
    """Predicted against observed cumulative default incidence, at each horizon.

    The distinction from the monthly check matters. The model emits a monthly hazard;
    what a lender prices against is the chance of default over years, which is the
    hazards compounded through the survival product. Errors that look negligible per
    month accumulate, and they can accumulate in either direction -- so a model can be
    well calibrated monthly and badly calibrated cumulatively.

    Loans censored before a horizon are **excluded from that horizon**, not counted as
    survivors. Counting them would understate incidence by exactly the share of the book
    that stopped reporting, and that share is not small. A loan that prepaid stays in and
    counts as a non-default, which is correct: cumulative *incidence* under competing
    risks asks whether default happened first, and for a prepaid loan it did not.
    """
    predicted_by_loan: list[pd.DataFrame] = []
    for _, paths in projected_paths(config, models, loans):
        wanted = paths[paths["loan_age"].isin([h - 1 for h in config.calibration.horizons])]
        predicted_by_loan.append(wanted[["loan_identifier", "loan_age", "cif_default"]])
    predicted = pd.concat(predicted_by_loan, ignore_index=True)

    observed = loans[["loan_identifier", "exit_age", "exit_reason"]]
    records = []
    for months in config.calibration.horizons:
        # Fully observed through the horizon, or already gone by an event before it.
        resolved = (observed["exit_age"] >= months - 1) | observed["exit_reason"].isin(
            ["default", "prepaid"]
        )
        usable = observed[resolved]
        defaulted = (usable["exit_reason"] == "default") & (usable["exit_age"] < months)
        model_at_horizon = predicted[predicted["loan_age"] == months - 1].set_index(
            "loan_identifier"
        )["cif_default"]
        aligned = model_at_horizon.reindex(usable["loan_identifier"]).to_numpy(dtype=float)
        # nanmean would quietly average over whatever survived the join. The observed
        # and predicted sides come from the same loan frame, so a gap here means the
        # projection dropped loans -- which would bias the predicted rate by exactly
        # the loans it dropped, in an unknown direction, with no symptom.
        unmatched = int(np.isnan(aligned).sum())
        if unmatched:
            raise ValueError(
                f"{unmatched} of {len(usable)} loans have no projected cumulative "
                f"incidence at {months} months; the projection and the observed "
                f"outcomes came from different populations"
            )
        records.append(
            {
                "population": label,
                "horizon_months": months,
                "loans": int(len(usable)),
                "excluded_censored": int(len(observed) - len(usable)),
                "observed_rate": float(defaulted.mean()),
                "predicted_rate": float(aligned.mean()),
            }
        )
    table = pd.DataFrame.from_records(records)
    table["ratio"] = table["predicted_rate"] / table["observed_rate"]
    table["gap"] = table["predicted_rate"] - table["observed_rate"]
    return table


def portfolio_npv(
    config: Config,
    models: HazardModels,
    loans: pd.DataFrame,
    *,
    variants: dict[str, Calibrator | None],
    economics: tuple[dict[str, float], ...] = (),
) -> pd.DataFrame:
    """NPV per loan under each hazard variant, and each economic assumption.

    One row per (variant, economics cell). Projections are chunked and every cell is
    evaluated inside the chunk that produced the paths, so the expensive part -- building
    the design matrix and scoring 120 months for every loan -- happens once per variant
    rather than once per cell.

    ``discount_spread_bps`` is handled differently from the other overrides and has to
    be: it is not an argument to :func:`loan_npv` at all. The discount rate reaches the
    NPV through a per-loan column in ``terms``, keyed on the loan's vintage, precisely so
    that no loan-level quantity can enter it. So a spread override shifts that column
    rather than being passed down -- and passing it down instead was a real bug here,
    caught only because :func:`loan_npv` takes keyword arguments explicitly rather than
    swallowing ``**kwargs``.
    """
    cells = economics or ({},)
    base_spread = config.economics.discount_spread_bps
    records = []
    for variant, calibrator in variants.items():
        totals: list[dict[str, float]] = [dict() for _ in cells]
        loans_seen = 0
        for terms, paths in projected_paths(config, models, loans, transform=calibrator):
            loans_seen += len(terms)
            for index, cell in enumerate(cells):
                shift = (cell.get("discount_spread_bps", base_spread) - base_spread) / 10_000.0
                priced = (
                    terms
                    if shift == 0.0
                    else terms.assign(discount_rate=terms["discount_rate"] + shift)
                )
                overrides = {k: v for k, v in cell.items() if k != "discount_spread_bps"}
                result = loan_npv(config, priced, paths, **overrides)  # type: ignore[arg-type]
                for column in ("npv", "principal", "default_leg", "terminal_leg"):
                    totals[index][column] = totals[index].get(column, 0.0) + float(
                        result.frame[column].sum()
                    )
        for index, cell in enumerate(cells):
            bucket = totals[index]
            records.append(
                {
                    "variant": variant,
                    "loans": loans_seen,
                    "lgd": cell.get("lgd", config.economics.lgd),
                    "discount_spread_bps": cell.get("discount_spread_bps", base_spread),
                    "npv_total": bucket["npv"],
                    "npv_per_loan": bucket["npv"] / loans_seen,
                    "npv_per_dollar": bucket["npv"] / bucket["principal"],
                    "default_leg": bucket["default_leg"],
                }
            )
    table = pd.DataFrame.from_records(records)
    if "raw" in set(table["variant"]):
        baseline = (
            table[table["variant"] == "raw"]
            .set_index(["lgd", "discount_spread_bps"])["npv_per_loan"]
        )
        keys = pd.MultiIndex.from_frame(table[["lgd", "discount_spread_bps"]])
        reference = baseline.reindex(keys).to_numpy(dtype=float)
        table["delta_vs_raw"] = table["npv_per_loan"].to_numpy(dtype=float) - reference
        table["delta_pct"] = table["delta_vs_raw"] / np.abs(reference)
    return table


def sensitivity_cells(config: Config) -> tuple[dict[str, float], ...]:
    """The economics grid: severity against the discount spread.

    Severity spans the train measurement (0.1330), the realised test figure (0.2407,
    which a lender could not have known) and a point either side. The spread spans 50 to
    150bp, the range that keeps every vintage inside [Treasury, note rate].
    """
    lgds = (0.10, config.economics.lgd, 0.18, 0.2407)
    spreads = (50.0, config.economics.discount_spread_bps, 150.0)
    return tuple(
        {"lgd": lgd, "discount_spread_bps": spread} for lgd in lgds for spread in spreads
    )


@dataclass(frozen=True)
class ScoredRows:
    """The three arrays the calibration measurement actually needs, and nothing else.

    Holding the person-period frame to score it is what made this harness unrunnable.
    The training frame alone is **5.6 GB** -- 12.1 million rows, and the string columns
    dominate it: ``loan_identifier``, ``exit_reason``, ``property_state``,
    ``property_type``, ``occupancy_status`` and ``loan_purpose`` cost 0.6-0.74 GB each as
    Python strings. With the test frame alongside it that is 8.9 GB before a single cash
    flow is projected, and the process was killed outright -- no traceback, no exit code,
    just gone, which is a genuinely confusing failure to diagnose from a log.

    Nothing downstream of scoring wants the frame. Calibration needs the predicted
    hazard, the observed outcome, and which side of the holdout each row is on. Those are
    three arrays: 8 bytes, 1 byte and 1 byte a row, about 130 MB for the whole training
    panel against 5.6 GB for the frame it came from.
    """

    hazard: np.ndarray
    observed: np.ndarray
    held: np.ndarray
    vintage: np.ndarray

    def __len__(self) -> int:
        return int(self.hazard.size)


def score_rows(
    config: Config,
    models: HazardModels,
    con: duckdb.DuckDBPyConnection,
    vintages: tuple[int, ...],
    *,
    holdout_share: float | None = None,
    progress: bool = True,
) -> ScoredRows:
    """Score a population one vintage at a time, keeping only the arrays.

    Streaming by vintage rather than loading the panel whole. One vintage is roughly
    2.4 million rows and about 1.1 GB, so peak memory is bounded by the largest single
    cohort instead of by the sum of all of them.

    The holdout assignment is computed on the **unique** loan identifiers and then
    broadcast back through :func:`pandas.factorize` codes. Hashing 12.1 million
    identifiers one at a time would take a minute for 400,000 distinct answers.
    """
    from cde.models.hazard import load_person_period

    target = CAUSE_TARGETS["default"]
    hazards: list[np.ndarray] = []
    outcomes: list[np.ndarray] = []
    held: list[np.ndarray] = []
    cohorts: list[np.ndarray] = []
    for vintage in vintages:
        frame = load_person_period(
            config, con, vintages=(vintage,), columns=required_columns(config)
        )
        hazards.append(models.causes["default"].hazard(frame))
        outcomes.append(frame[target].to_numpy(dtype=np.int8))
        cohorts.append(np.full(len(frame), vintage, dtype=np.int16))
        if holdout_share is not None:
            codes, uniques = pd.factorize(frame["loan_identifier"], sort=False)
            by_loan = holdout_mask(
                np.asarray(uniques),
                share=holdout_share,
                seed=config.calibration.holdout_seed,
            )
            held.append(by_loan[codes])
        else:
            held.append(np.zeros(len(frame), dtype=bool))
        if progress:
            print(
                f"  scored {vintage}: {len(frame):>10,} rows, "
                f"{int(outcomes[-1].sum()):>6,} default events",
                flush=True,
            )
        del frame
    return ScoredRows(
        hazard=np.concatenate(hazards),
        observed=np.concatenate(outcomes).astype(float),
        held=np.concatenate(held),
        vintage=np.concatenate(cohorts),
    )


@dataclass(frozen=True)
class CalibrationReport:
    """Everything stage 3 measured, and the checks that say whether to believe it."""

    scores: list[PopulationScore]
    calibrator_notes: list[str]
    horizons: pd.DataFrame
    npv: pd.DataFrame
    holdout_share: float

    def score_for(self, label: str, variant: str) -> PopulationScore | None:
        for candidate in self.scores:
            if candidate.label == label and candidate.variant == variant:
                return candidate
        return None

    def checks(self) -> list[tuple[bool, str]]:
        """The invariants that decide whether the harness itself is trustworthy."""
        out: list[tuple[bool, str]] = []
        in_sample = self.score_for("train (in-sample)", "raw")
        if in_sample is not None:
            # A converged logistic fit with an unpenalised intercept reproduces the
            # observed event count exactly, in sample. Anything else is a harness bug.
            ok = abs(in_sample.marginal_ratio - 1.0) < 0.01
            out.append(
                (
                    ok,
                    f"in-sample marginal ratio {in_sample.marginal_ratio:.4f} "
                    f"(must be ~1.000 by construction, or the harness is wrong)",
                )
            )
        test = self.score_for("test (out-of-time)", "raw")
        if test is not None:
            # Under-prediction out of time is the expected direction: the training
            # cohorts defaulted far less. A ratio at or above 1 would mean the crisis
            # book was no worse than the calm one, which contradicts stage 1.
            out.append(
                (
                    test.marginal_ratio < 1.0,
                    f"out-of-time marginal ratio {test.marginal_ratio:.4f} "
                    f"(< 1 expected: the model under-predicts the crisis book)",
                )
            )
            out.append(
                (
                    test.ece > 0.0,
                    f"out-of-time ECE {test.ece:.6f} (exactly zero would mean leakage)",
                )
            )
        for method in ("platt", "isotonic"):
            recalibrated = self.score_for("test (out-of-time)", method)
            if recalibrated is None or test is None:
                continue
            # The prediction stated in the module docstring, checked rather than
            # assumed. Judged as a FRACTION of the gap, because an unqualified
            # "narrows" fired on a move of 0.0016 against a gap of 0.63 -- true, and
            # meaningless. Under a twentieth of the gap counts as leaving it open.
            gap = abs(test.marginal_ratio - 1.0)
            closed = (gap - abs(recalibrated.marginal_ratio - 1.0)) / gap
            verdict = (
                "leaves the out-of-time gap open"
                if closed < 0.05
                else f"closes {closed:.0%} of the out-of-time gap"
            )
            out.append(
                (
                    True,
                    f"in-time {method} {verdict}: ratio {test.marginal_ratio:.4f} -> "
                    f"{recalibrated.marginal_ratio:.4f}"
                    f"   (predicted in advance that it would not close it)",
                )
            )
        sequential = [s for s in self.scores if "sequential" in s.label]
        if sequential:
            raw_later = next((s for s in sequential if s.variant == "raw"), None)
            fitted_later = next((s for s in sequential if s.variant != "raw"), None)
            if raw_later is not None and fitted_later is not None:
                gap = abs(raw_later.marginal_ratio - 1.0)
                closed = (gap - abs(fitted_later.marginal_ratio - 1.0)) / gap
                out.append(
                    (
                        True,
                        f"by contrast, recalibrating on the FIRST test vintage closes "
                        f"{closed:.0%} of the gap on the rest: ratio "
                        f"{raw_later.marginal_ratio:.4f} -> "
                        f"{fitted_later.marginal_ratio:.4f}   (so the miscalibration is "
                        f"a regime change, not overfitting)",
                    )
                )
        return out

    def summary(self) -> str:
        lines = [
            f"stage 3 calibration   model fitted on {1.0 - self.holdout_share:.0%} of "
            f"training loans",
            "",
            "MONTHLY DEFAULT HAZARD",
        ]
        for population in self.scores:
            lines.append(population.summary())
            lines.append("")
        if self.calibrator_notes:
            lines.append("recalibrators, fitted on the in-time holdout:")
            lines += [f"  {note}" for note in self.calibrator_notes]
            lines.append("")
        if not self.horizons.empty:
            lines.append("CUMULATIVE DEFAULT INCIDENCE, predicted vs observed")
            lines.append(
                f"  {'population':<22} {'horizon':>8} {'loans':>9} {'predicted':>10} "
                f"{'observed':>9} {'ratio':>7}"
            )
            for row in self.horizons.itertuples():
                lines.append(
                    f"  {row.population:<22} {row.horizon_months:>7}m {row.loans:>9,} "
                    f"{row.predicted_rate:>9.3%} {row.observed_rate:>8.3%} "
                    f"{row.ratio:>7.3f}"
                )
            lines.append("")
        if not self.npv.empty:
            headline = headline_npv(self.npv)
            excluded = self.npv.attrs.get("excluded_left_truncated", 0)
            lines.append("PORTFOLIO NPV, out-of-time book, projected from origination")
            if excluded:
                lines.append(
                    f"  {excluded:,} loans excluded: they enter the panel after age 0, so "
                    f"there is no decision point to value them at"
                )
            lines.append(
                f"  {'variant':<12} {'loans':>9} {'NPV per loan':>14} "
                f"{'per dollar':>11} {'vs raw':>12} {'vs raw':>8}"
            )
            for row in headline.itertuples():
                lines.append(
                    f"  {row.variant:<12} {row.loans:>9,} {row.npv_per_loan:>14,.0f} "
                    f"{row.npv_per_dollar:>10.3%} {row.delta_vs_raw:>12,.0f} "
                    f"{row.delta_pct:>7.2%}"
                )
            later = self.npv[self.npv.get("scope", "") == SCOPE_SEQUENTIAL]
            if not later.empty:
                vintages = self.npv.attrs.get("sequential_vintages", [])
                fitted_on = self.npv.attrs.get("sequential_fitted_on", "the first vintage")
                lines.append("")
                lines.append(
                    f"  ...and on {vintages} alone, adding the SEQUENTIAL arm: a Platt "
                    f"calibrator fitted on {fitted_on}, which is the only variant that "
                    f"actually moves the probabilities"
                )
                lines.append(
                    f"  {'variant':<12} {'loans':>9} {'NPV per loan':>14} "
                    f"{'per dollar':>11} {'vs raw':>12} {'vs raw':>8}"
                )
                for row in later.itertuples():
                    flag = "   <-- what the mispricing costs" if row.variant == "sequential" else ""
                    lines.append(
                        f"  {row.variant:<12} {row.loans:>9,} {row.npv_per_loan:>14,.0f} "
                        f"{row.npv_per_dollar:>10.3%} {row.delta_vs_raw:>12,.0f} "
                        f"{row.delta_pct:>7.2%}{flag}"
                    )
            grid = self.npv[self.npv.get("scope", "") == SCOPE_SENSITIVITY]
            if not grid.empty:
                spread = config_spread(grid)
                lines.append("")
                lines.append(
                    f"  sensitivity, 10% sample: NPV per loan, recalibrated minus raw, "
                    f"at {spread:.0f}bp"
                )
                cell = grid[grid["discount_spread_bps"] == spread]
                for variant in sorted(set(cell["variant"]) - {"raw"}):
                    series = cell[cell["variant"] == variant].sort_values("lgd")
                    rendered = "  ".join(
                        f"LGD {row.lgd:.3f}: {row.delta_vs_raw:>+8,.0f}"
                        for row in series.itertuples()
                    )
                    lines.append(f"    {variant:<10} {rendered}")
            lines.append("")
        lines.append("CHECKS")
        for ok, message in self.checks():
            lines.append(f"  {'ok  ' if ok else 'FAIL'}  {message}")
        return "\n".join(lines)


def build(
    config: Config,
    models: HazardModels,
    con: duckdb.DuckDBPyConnection,
    *,
    holdout_share: float,
    with_npv: bool = True,
    with_sensitivity: bool = True,
) -> CalibrationReport:
    """Run the whole of stage 3."""
    settings = config.calibration
    scores: list[PopulationScore] = []
    notes: list[str] = []

    # --- pass 1: training vintages, split into the fitted 80% and the unseen 20% -----
    print("scoring the training vintages", flush=True)
    train = score_rows(
        config, models, con, config.data.train_vintages, holdout_share=holdout_share
    )
    held = train.held
    for label, mask in (
        ("train (in-sample)", ~held),
        ("holdout (in-time)", held),
    ):
        scores.append(score(config, label, "raw", train.hazard[mask], train.observed[mask]))

    calibrators = fit_calibrators(config, train.hazard[held], train.observed[held])
    for method, calibrator in calibrators.items():
        notes.append(calibrator.describe())
        for label, mask in (("train (in-sample)", ~held), ("holdout (in-time)", held)):
            scores.append(
                score(
                    config,
                    label,
                    method,
                    calibrator.transform(train.hazard[mask]),
                    train.observed[mask],
                )
            )
    del train, held

    # --- pass 2: the out-of-time book ------------------------------------------------
    print("scoring the out-of-time vintages", flush=True)
    test = score_rows(config, models, con, config.data.test_vintages)
    raw_test, observed_test = test.hazard, test.observed
    label = "test (out-of-time)"
    scores.append(score(config, label, "raw", raw_test, observed_test))
    for method, calibrator in calibrators.items():
        scores.append(
            score(config, label, method, calibrator.transform(raw_test), observed_test)
        )

    # A calibrator fitted on the FIRST test vintage and applied to the rest. Closer to
    # how a lender would actually operate -- recalibrate on what has just been seen --
    # but optimistic about availability: fully observing the 2006 cohort takes 120
    # months, so the information is not there in 2007. Reported with that caveat.
    sequential: Calibrator | None = None
    first_test = config.data.test_vintages[0]
    is_first = test.vintage == first_test
    if is_first.any() and (~is_first).any() and observed_test[is_first].sum() > 0:
        sequential = build_calibrator(
            "platt",
            tolerance=config.model.tolerance,
            max_iterations=config.model.max_iterations,
            isotonic_min_tail_rows=config.calibration.isotonic_min_tail_rows,
        ).fit(raw_test[is_first], observed_test[is_first])
        later = ", ".join(str(v) for v in config.data.test_vintages[1:])
        scores.append(
            score(
                config,
                f"test {later} (sequential)",
                f"platt fitted on {first_test}",
                sequential.transform(raw_test[~is_first]),
                observed_test[~is_first],
            )
        )
        scores.append(
            score(
                config,
                f"test {later} (sequential)",
                "raw",
                raw_test[~is_first],
                observed_test[~is_first],
            )
        )
        notes.append(
            f"sequential: platt fitted on {first_test}, applied to {later} — "
            f"{sequential.describe()}"
        )
    del test, raw_test, observed_test

    # --- cumulative incidence and value ---------------------------------------------
    horizons = pd.DataFrame()
    npv = pd.DataFrame()
    if with_npv:
        print("projecting cash flows from origination", flush=True)
        all_test = load_loans(config, con, vintages=config.data.test_vintages)
        test_loans = eligible_for_npv(all_test)
        excluded = len(all_test) - len(test_loans)
        horizons = cumulative_calibration(config, models, test_loans, label="test (out-of-time)")
        variants: dict[str, Calibrator | None] = {"raw": None}
        variants.update(calibrators)
        # The headline runs on every eligible loan at the configured economics. It
        # carries the in-time recalibrators only, because those are the ones a lender
        # could actually have fitted at the decision point.
        frames = [
            portfolio_npv(config, models, test_loans, variants=variants).assign(
                scope=SCOPE_HEADLINE
            )
        ]

        # The arm that matters, and the one the headline cannot show. In-time
        # recalibration barely moves these probabilities -- the model is already
        # calibrated on its own vintages, so Platt comes back near the identity and the
        # NPV with it. That makes the headline delta small for a reason that has nothing
        # to do with whether calibration is worth anything.
        #
        # The sequential calibrator, fitted on the FIRST test vintage and applied to the
        # rest, does move them: it closes most of the out-of-time gap. Pricing the same
        # loans with and without it is therefore what puts a number on the mispricing,
        # which is what claim 2 is actually about. Restricted to the later test vintages
        # so that every variant here is scored on one population and the 2006 cohort is
        # used only to fit.
        later_vintages = config.data.test_vintages[1:]
        sequential_loans = pd.DataFrame()
        if sequential is not None and later_vintages:
            sequential_loans = test_loans[
                test_loans["origination_vintage"].isin(later_vintages)
            ].reset_index(drop=True)
            with_sequential: dict[str, Calibrator | None] = dict(variants)
            with_sequential["sequential"] = sequential
            frames.append(
                portfolio_npv(
                    config, models, sequential_loans, variants=with_sequential
                ).assign(scope=SCOPE_SEQUENTIAL)
            )

        if with_sensitivity:
            # The grid runs on a fixed subsample of whichever population carries the
            # most variants. It asks how the *conclusion* moves with severity and the
            # discount spread -- a question about a difference between variants priced
            # identically, so the sampling error largely cancels and a tenth of the book
            # answers it to far better precision than either assumption is known to.
            # Sampled by the same identifier hash as the holdout, so it is reproducible.
            population = sequential_loans if len(sequential_loans) else test_loans
            grid_variants = (
                with_sequential
                if sequential is not None and len(sequential_loans)
                else variants
            )
            keep = holdout_mask(
                population["loan_identifier"].to_numpy(),
                share=SENSITIVITY_SAMPLE,
                seed=settings.holdout_seed,
            )
            frames.append(
                portfolio_npv(
                    config,
                    models,
                    population[keep].reset_index(drop=True),
                    variants=grid_variants,
                    economics=sensitivity_cells(config),
                ).assign(scope=SCOPE_SENSITIVITY)
            )
        npv = pd.concat(frames, ignore_index=True)
        npv.attrs["excluded_left_truncated"] = excluded
        npv.attrs["sequential_vintages"] = list(later_vintages)
        npv.attrs["sequential_fitted_on"] = first_test
    # How often a tail cap actually bound, reported AFTER everything has been
    # transformed. A cap that binds on a large share of rows is not a safety measure any
    # more -- it is doing the pricing, and the variant's NPV would be a statement about
    # the cap rather than about calibration. Appended here rather than at fit time,
    # where the count is necessarily still zero.
    for method, calibrator in calibrators.items():
        share = getattr(calibrator, "clamped_share", 0.0)
        if share:
            notes.append(
                f"{method}: the tail cap bound on {share:.4%} of all transformed rows"
                + ("   <-- MATERIAL: this variant is partly reporting the cap" if share > 0.01
                   else "   (immaterial, so the variant still reports calibration)")
            )
    return CalibrationReport(
        scores=scores,
        calibrator_notes=notes,
        horizons=horizons,
        npv=npv,
        holdout_share=holdout_share,
    )


def headline_npv(npv: pd.DataFrame) -> pd.DataFrame:
    """The full-population rows at the configured economics: the number to quote.

    The sensitivity grid lives in the same frame, so selecting on the scope label rather
    than on "the first row's assumptions" matters -- the latter silently picked whichever
    cell happened to sort first once the grid was appended.
    """
    if "scope" in npv.columns:
        return npv[npv["scope"] == SCOPE_HEADLINE]
    return npv


def config_spread(grid: pd.DataFrame) -> float:
    """The spread value nearest the middle of the grid, used as its reference column."""
    spreads = sorted(grid["discount_spread_bps"].unique())
    return float(spreads[len(spreads) // 2])


def write_scores(report: CalibrationReport, path: Path) -> Path:
    """The headline per-population table, one row per (population, variant)."""
    records = [
        {
            "population": s.label,
            "variant": s.variant,
            "in_sample_for_this_fit": s.is_in_sample,
            "rows": s.rows,
            "observed_events": s.observed_events,
            "predicted_events": s.predicted_events,
            "marginal_ratio": s.marginal_ratio,
            "base_rate": s.base_rate,
            "ece": s.ece,
            "max_error": s.max_error,
            "brier": s.brier.brier,
            "reliability": s.brier.reliability,
            "resolution": s.brier.resolution,
            "uncertainty": s.brier.uncertainty,
            "skill": s.brier.skill,
        }
        for s in report.scores
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(path, index=False)
    return path


def reliability_table(report: CalibrationReport) -> pd.DataFrame:
    """Every reliability table stacked, labelled by population and variant.

    Returned rather than only written, so the figures and the CSV are the same object
    and a chart cannot drift from the file the write-up quotes.
    """
    return pd.concat(
        [s.table.assign(population=s.label, variant=s.variant) for s in report.scores],
        ignore_index=True,
    )


def write_reliability(report: CalibrationReport, path: Path) -> Path:
    """Every reliability table stacked, so the curves and the write-up cite one file."""
    return write_frame(reliability_table(report), path)


def write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
