"""Discrete-time hazard model, fitted as logistic regression on the person-period data.

Stage 2. Loan age enters flexibly (splines, never a linear term) -- see
:mod:`cde.models.features` for why.

## Why logistic regression on person-period rows *is* a survival model

This is the part worth being able to say out loud. Each row asks a single binary
question: given that this loan was still alive entering month t, did it default
during month t? The fitted probability is therefore the discrete-time hazard
h(t | x) directly, not a "probability of default" in the loose sense. Censoring
needs no special handling at all -- a loan simply stops contributing rows -- which
is why no Cox partial likelihood or interval-censoring machinery appears anywhere
here. Fitting is ordinary maximum likelihood on a Bernoulli likelihood, and it
happens to be exactly the survival likelihood.

## Competing risks

Two models, same design matrix, different target column: one for default and one
for prepayment. Each estimates a **cause-specific hazard** -- the probability of
leaving this month *for that reason* -- and each is estimated correctly whether or
not the other is modelled, because the other cause's exits are simply rows that
stop appearing. What needs both is survival and the cash flows:

    S(t) = PROD (1 - h_default(u) - h_prepay(u))

A loan that repays early stops paying interest, which is a real term in the NPV, so
stage 4 needs the prepayment hazard as much as the default one. Stage 1 measured
prepayment at 77-94% of every cohort inside ten years, which is what moved this
from optional to required.

## Two things deliberately NOT done

**No class balancing, no resampling, no `class_weight`.** Default events are 0.08%
of rows on the training vintages, and every instinct says to rebalance. Doing so
would multiply the predicted
probabilities by a constant factor and destroy their *level* while leaving their
*ranking* untouched -- so AUC would be unchanged and the NPV calculation that
consumes the probability would be wrong for every loan in the book. That is claim 2
of this project, and fitting it wrong here to make a metric look better would be
exactly the mistake the project exists to demonstrate.

**Barely any regularisation.** L2 shrinks predictions toward the base rate, which is
miscalibration introduced by the fitting procedure. Stage 3 measures miscalibration
and recalibrates; it should be measuring the model's, not the penalty's.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cde.config import Config
from cde.models.features import DesignSpec, build_matrix, fit_spec

#: Cause name -> the person-period column holding its event indicator.
CAUSE_TARGETS = {"default": "default_event", "prepayment": "prepayment_event"}


def _row_order(frame: pd.DataFrame) -> np.ndarray | None:
    """The permutation putting rows in (loan, ascending age) order, or None if already there.

    :meth:`HazardModels.survival` accumulates with ``cumprod``, ``shift`` and
    ``cumsum``, and all three walk rows in the order they physically sit in the
    frame -- none of them consults ``loan_age``. So the row order *is* the time
    axis, and getting it wrong produces plausible numbers rather than an error.

    Returning None for an already-ordered frame is not a micro-optimisation: the
    reorder itself is quick, but applying it copies every column, which on the full
    training frame is a couple of gigabytes. The check is a few numpy passes.

    Ordered means two things, both required:

    * loan blocks are contiguous and sorted, so a loan's rows are never scattered
      through the frame. Without this, the adjacent-row check below would pass
      vacuously on a frame like [(A,5), (B,1), (A,1)] -- every adjacent pair
      crosses a loan boundary, yet A's rows run 5 then 1.
    * within a loan, age strictly increases from one row to the next.
    """
    if len(frame) <= 1:
        return None
    loans = frame["loan_identifier"].to_numpy()
    ages = frame["loan_age"].to_numpy()
    boundary = loans[1:] != loans[:-1]
    ascending = bool(np.all((ages[1:] > ages[:-1]) | boundary))
    if ascending and frame["loan_identifier"].is_monotonic_increasing:
        return None
    keys = frame[["loan_identifier", "loan_age"]].reset_index(drop=True)
    return np.asarray(
        keys.sort_values(["loan_identifier", "loan_age"], kind="stable").index.to_numpy()
    )


def _reject_month_gaps(frame: pd.DataFrame) -> None:
    """Refuse a frame missing months inside a loan's observed span.

    The second, quieter assumption. ``shift(1)`` takes the *previous row*, so it
    reads as "the previous month" only if no month is missing: on a loan jumping
    from age 5 to age 7, S(t-1) for month 7 would silently be S(5), skipping a
    month of exposure.

    A loan starting at age 6 is fine and common -- 17% of these loans enter the
    panel late because Freddie Mac bought them after origination. What is rejected
    is a hole in the middle, which the person-period table cannot produce (loans
    are censored at reporting gaps) but a filtered frame can.
    """
    if len(frame) <= 1:
        return
    loans = frame["loan_identifier"].to_numpy()
    ages = frame["loan_age"].to_numpy()
    same_loan = loans[1:] == loans[:-1]
    jumps = same_loan & (ages[1:] - ages[:-1] > 1)
    if jumps.any():
        first = int(np.flatnonzero(jumps)[0])
        raise ValueError(
            f"{int(jumps.sum())} month gap(s) inside a loan's span, first at "
            f"{loans[first]!r} between ages {ages[first]} and {ages[first + 1]}. "
            f"Survival would treat the row before the gap as the previous month and "
            f"skip the missing exposure. Pass contiguous months, or censor at the gap."
        )


@dataclass(frozen=True)
class CauseModel:
    """A fitted cause-specific hazard model."""

    cause: str
    model: LogisticRegression
    spec: DesignSpec
    events: int
    rows: int
    predicted_events: float
    iterations: int

    @property
    def marginal_gap(self) -> float:
        """Relative gap between predicted and observed event counts.

        Must be ~0. A converged logistic regression with an unpenalised intercept
        satisfies sum(y - p) = 0 exactly, so anything material here means the
        optimiser stopped early -- not that the model is overconfident. Surfaced
        because the difference matters enormously for stage 3 and is invisible
        otherwise.
        """
        return (self.predicted_events - self.events) / self.events if self.events else 0.0

    @property
    def base_rate(self) -> float:
        return self.events / self.rows if self.rows else 0.0

    def hazard(self, frame: pd.DataFrame) -> np.ndarray:
        """Predicted monthly hazard for each person-period row."""
        matrix = build_matrix(self.spec, frame)
        return np.asarray(self.model.predict_proba(matrix)[:, 1], dtype=float)

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": self.spec.feature_names,
                "coefficient": self.model.coef_[0],
            }
        ).sort_values("coefficient", key=abs, ascending=False)


@dataclass(frozen=True)
class HazardModels:
    """The cause-specific models that together describe how a loan leaves."""

    causes: dict[str, CauseModel]
    spec: DesignSpec
    #: Copied from config at fit time so the summary can flag a fit that stopped
    #: because it ran out of iterations rather than because it converged.
    #:
    #: Deliberately has no default. It used to default to 0, which made the check
    #: below (`iterations >= max_iterations`) true for every fit, so the summary
    #: claimed every cause had hit the cap. `fit()` always passes the real value so
    #: it never fired in practice, but a warning that is always on is worse than no
    #: warning: it trains you to ignore it.
    max_iterations: int

    def summary(self) -> str:
        lines = [self.spec.summary(), "", "fitted cause-specific hazards:"]
        for cause, fitted in self.causes.items():
            flag = "" if abs(fitted.marginal_gap) < 1e-3 else "   <-- CHECK CONVERGENCE"
            hit_cap = fitted.iterations >= self.max_iterations
            cap = "  <-- HIT THE ITERATION CAP" if hit_cap else ""
            lines.append(
                f"  {cause:<12} {fitted.events:>8,} events / {fitted.rows:>12,} rows"
                f"   base rate {fitted.base_rate:.5%}"
            )
            lines.append(
                f"  {'':<12} lbfgs iterations {fitted.iterations:>5}{cap}"
                f"   marginal gap {fitted.marginal_gap:+.5%}{flag}"
            )
        return "\n".join(lines)

    def survival(
        self, frame: pd.DataFrame, hazards: dict[str, np.ndarray] | None = None
    ) -> pd.DataFrame:
        """Per-loan survival and cause-specific cumulative incidence by month.

        Row order is the time axis here, so it is enforced rather than assumed:
        rows are reordered into (loan, ascending age) if they are not already, and
        the result is handed back in the caller's original row order. A frame with
        months missing inside a loan's span is rejected outright -- see
        :func:`_row_order` and :func:`_reject_month_gaps`.

        ``hazards`` substitutes already-computed hazards for the models' own, keyed by
        cause and given **in the caller's row order**. Stage 3 needs it: a recalibrated
        hazard has to be pushed back through the survival product, because recalibrating
        h_d changes S(t) and therefore every later month's contribution. Passing them in
        rather than recalibrating inside keeps this the only implementation of the
        recursion -- an independent one in the calibration harness is exactly how two
        sets of numbers drift apart.
        """
        order = _row_order(frame)
        working = frame if order is None else frame.iloc[order]
        _reject_month_gaps(working)

        if hazards is not None:
            unknown = set(hazards) - set(self.causes)
            if unknown:
                raise KeyError(f"hazards given for unmodelled cause(s) {sorted(unknown)}")
            missing = set(self.causes) - set(hazards)
            if missing:
                raise KeyError(
                    f"hazards missing cause(s) {sorted(missing)}; survival needs every "
                    f"cause, since S(t) is the product over all of them"
                )
            for cause, values in hazards.items():
                if len(values) != len(frame):
                    raise ValueError(
                        f"hazards[{cause!r}] has {len(values)} rows, frame has {len(frame)}"
                    )

        out = working[["loan_identifier", "loan_age"]].copy()
        for cause, fitted in self.causes.items():
            if hazards is None:
                out[f"hazard_{cause}"] = fitted.hazard(working)
            else:
                supplied = np.asarray(hazards[cause], dtype=float)
                out[f"hazard_{cause}"] = supplied if order is None else supplied[order]

        leaving = sum(out[f"hazard_{cause}"] for cause in self.causes)
        if (leaving >= 1.0).any():
            # Two hazards that sum past one would make survival negative. It cannot
            # happen from two logistic fits at realistic rates, but a silent negative
            # survival would poison every NPV downstream, so it fails loudly.
            raise ValueError(
                "cause-specific hazards sum to >= 1 for some rows; survival would go negative"
            )

        # Vectorised within loan: cumprod for S(t), then shift for S(t-1). A
        # groupby-apply here would be correct and roughly two orders of magnitude
        # slower across a quarter of a million loans.
        out["survival"] = (1.0 - leaving).groupby(out["loan_identifier"]).cumprod()
        out["survival_entering"] = (
            out.groupby("loan_identifier", sort=False)["survival"].shift(1).fillna(1.0)
        )
        for cause in self.causes:
            increments = out["survival_entering"] * out[f"hazard_{cause}"]
            out[f"cif_{cause}"] = increments.groupby(out["loan_identifier"]).cumsum()

        if order is None:
            return out
        # Back to the caller's row order. Positional rather than by index, because a
        # frame built by concatenation can carry duplicate index labels and reindex
        # would then multiply rows instead of permuting them.
        return out.iloc[np.argsort(order)]


def fit(config: Config, frame: pd.DataFrame) -> HazardModels:
    """Fit every cause-specific hazard on the given person-period rows."""
    spec = fit_spec(config, frame)
    matrix = build_matrix(spec, frame)
    causes: dict[str, CauseModel] = {}
    for cause, target in CAUSE_TARGETS.items():
        if target not in frame.columns:
            raise KeyError(f"person-period table has no {target!r} column for cause {cause!r}")
        outcome = frame[target].to_numpy(dtype=int)
        model = LogisticRegression(
            C=config.model.l2_inverse_strength,
            max_iter=config.model.max_iterations,
            tol=config.model.tolerance,
            solver="lbfgs",
            class_weight=None,  # deliberate; see the module docstring
        )
        model.fit(matrix, outcome)
        causes[cause] = CauseModel(
            cause=cause,
            model=model,
            spec=spec,
            events=int(outcome.sum()),
            rows=int(outcome.size),
            predicted_events=float(model.predict_proba(matrix)[:, 1].sum()),
            iterations=int(model.n_iter_[0]),
        )
    return HazardModels(causes=causes, spec=spec, max_iterations=config.model.max_iterations)


def calibration_by_age(models: HazardModels, frame: pd.DataFrame) -> pd.DataFrame:
    """Predicted versus observed hazard, by month on book, per cause.

    The first diagnostic to run, and the one that decides whether the spline basis
    is adequate. If the model's mean predicted hazard at each loan age does not
    track the observed rate at that age, the age basis is too coarse -- and no
    amount of covariate work fixes that.

    This is in-sample agreement, so it is a specification check, not evidence the
    model generalises. Stage 5 does that.
    """
    out = frame[["loan_age"]].copy()
    for cause, fitted in models.causes.items():
        out[f"predicted_{cause}"] = fitted.hazard(frame)
        out[f"observed_{cause}"] = frame[CAUSE_TARGETS[cause]].to_numpy(dtype=float)
    aggregations = {"at_risk": ("loan_age", "size")}
    for cause in models.causes:
        aggregations[f"predicted_{cause}"] = (f"predicted_{cause}", "mean")
        aggregations[f"observed_{cause}"] = (f"observed_{cause}", "mean")
    return out.groupby("loan_age").agg(**aggregations).reset_index()


def load_person_period(
    config: Config,
    con: duckdb.DuckDBPyConnection,
    *,
    vintages: tuple[int, ...],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read the person-period rows for the given vintages into memory.

    Only the columns the model needs, because the full table carries every
    origination field and pulling all of them would triple the footprint.
    """
    needed = columns if columns is not None else required_columns(config)
    placeholders = ", ".join(str(int(v)) for v in vintages)
    query = (
        f"SELECT {', '.join(needed)} FROM person_period "
        f"WHERE origination_vintage IN ({placeholders}) "
        f"ORDER BY loan_identifier, loan_age"
    )
    return con.execute(query).df()


def required_columns(config: Config) -> list[str]:
    """The person-period columns stage 2 reads. Anything else is left on disk."""
    model = config.model
    base = ["loan_identifier", "origination_vintage", "loan_age", "exit_reason"]
    base += list(CAUSE_TARGETS.values())
    numeric = set(model.spline_columns) | set(model.linear_columns)
    # log_original_upb is derived, so ask for the column it is derived from.
    numeric.discard("log_original_upb")
    numeric.add("original_upb")
    ordered = base + sorted(numeric) + sorted(model.categorical_columns)
    return list(dict.fromkeys(ordered))
