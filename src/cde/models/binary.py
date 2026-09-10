"""The comparison model: binary "did it default within 36 months".

Stage 5's B1 and B2 arms, and deliberately the thing most public credit notebooks build.
It exists to be beaten, but it has to be beaten **fairly**, so it is given every advantage
the hazard model has except the one under test.

## What it shares with the hazard model, and why that matters

Same covariates, same spline and one-hot encoding, same L2 strength, same tight optimiser
tolerance, same training loans — the identical 80% of the training vintages, so neither
model has seen a row the other has not. The only differences are the two that define the
arm:

* **one row per loan instead of one row per loan-month.** So there is no time axis, and
  :func:`cde.models.features.fit_spec` is called with ``include_loan_age=False`` rather
  than fed a column of zeros.
* **a binary target over a fixed window** instead of a monthly hazard, so it cannot say
  *when*.

Without that discipline the B2-to-B3 gap would measure a different feature set, or a
different training sample, and there would be no way afterwards to say how much was the
survival framing. Claim 1 is the whole point of the comparison, and it is only measurable
if everything else is held fixed.

## Censoring is where a binary target starts to hurt

A monthly hazard handles censoring structurally: a loan stops contributing rows and
nothing else changes. A binary target cannot. A loan observed for 20 months has no
readable answer to "did it default within 36", and the two available fixes are both wrong
— dropping it discards information and biases toward loans that survived long enough to be
observed, while calling it a non-default understates the rate by exactly the share of the
book that stopped reporting.

This model drops them, and :attr:`BinaryModel.unresolved` reports how many. On these
mature cohorts it is a small number, which is itself worth saying: the binary target's
censoring problem is mild *here* only because the data runs ten years past origination.
On a live book, where most loans are young, it would dominate.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cde.config import Config
from cde.models.features import DesignSpec, build_matrix, fit_spec
from cde.models.hazard import required_columns

#: Column holding the binary target.
TARGET = "defaulted_within"


@dataclass(frozen=True)
class BinaryModel:
    """A fitted binary default classifier over a fixed window."""

    model: LogisticRegression
    spec: DesignSpec
    horizon_months: int
    events: int
    rows: int
    predicted_events: float
    iterations: int
    unresolved: int

    @property
    def base_rate(self) -> float:
        return self.events / self.rows if self.rows else 0.0

    @property
    def marginal_gap(self) -> float:
        """Predicted minus observed events, relative. Must be ~0 for a converged fit."""
        return (self.predicted_events - self.events) / self.events if self.events else 0.0

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        """P(default within the window) for each loan."""
        matrix = build_matrix(self.spec, frame)
        return np.asarray(self.model.predict_proba(matrix)[:, 1], dtype=float)

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"feature": self.spec.feature_names, "coefficient": self.model.coef_[0]}
        ).sort_values("coefficient", key=abs, ascending=False)

    def summary(self) -> str:
        return "\n".join(
            [
                f"binary default-within-{self.horizon_months}m classifier",
                f"  {self.rows:>9,} loans   {self.events:>7,} events"
                f"   base rate {self.base_rate:.4%}",
                f"  {self.unresolved:>9,} dropped as unresolved at the window",
                f"  features {len(self.spec.feature_names)}"
                f"   (no loan-age block: one row per loan)",
                f"  lbfgs iterations {self.iterations:>4}"
                f"   marginal gap {self.marginal_gap:+.5%}",
            ]
        )


def loan_level_frame(
    config: Config,
    con: duckdb.DuckDBPyConnection,
    vintages: tuple[int, ...],
) -> pd.DataFrame:
    """One row per loan, with the binary target attached.

    Taken from the person-period table at each loan's first observed month, so the
    covariates are byte-for-byte the ones the hazard model was fitted with. Reading them
    from ``origination`` instead would risk a column diverging between the two arms and
    quietly become part of what the ablation measures.
    """
    window = config.decision.binary_horizon_months
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
        raise ValueError("more than one row per loan at its first observed age")
    frame["resolved"] = (frame["exit_age"] >= window - 1) | frame["exit_reason"].isin(
        ["default", "prepaid"]
    )
    frame[TARGET] = (frame["exit_reason"] == "default") & (frame["exit_age"] < window)
    return frame


def fit(config: Config, frame: pd.DataFrame) -> BinaryModel:
    """Fit the binary classifier on the resolved loans of ``frame``."""
    window = config.decision.binary_horizon_months
    unresolved = int((~frame["resolved"]).sum())
    usable = frame[frame["resolved"]].reset_index(drop=True)
    outcome = usable[TARGET].to_numpy(dtype=int)
    if outcome.sum() == 0:
        raise ValueError(
            f"no loan defaulted within {window} months in this sample; the binary arm "
            f"has nothing to fit"
        )
    spec = fit_spec(config, usable, include_loan_age=False)
    matrix = build_matrix(spec, usable)
    model = LogisticRegression(
        C=config.model.l2_inverse_strength,
        max_iter=config.model.max_iterations,
        tol=config.model.tolerance,
        solver="lbfgs",
        # No class weighting, for the same reason the hazard model has none: it would
        # multiply the predicted probabilities by a constant and destroy their level
        # while leaving the ranking untouched. B2 consumes the probability itself.
        class_weight=None,
    )
    model.fit(matrix, outcome)
    return BinaryModel(
        model=model,
        spec=spec,
        horizon_months=window,
        events=int(outcome.sum()),
        rows=int(outcome.size),
        predicted_events=float(model.predict_proba(matrix)[:, 1].sum()),
        iterations=int(model.n_iter_[0]),
        unresolved=unresolved,
    )


def implied_monthly_hazard(probability: np.ndarray, window: int) -> np.ndarray:
    """Turn a window probability into the constant monthly hazard that reproduces it.

        1 - (1 - h)^window = p        =>        h = 1 - (1 - p)^(1/window)

    This is the **only** way a binary classifier can be given to a cash-flow model, and
    the assumption it forces is precisely what claim 1 is about. The binary target says a
    loan has a 6% chance of defaulting sometime in three years and says nothing about
    when; a month-2 default and a month-35 default destroy very different amounts of
    value, so something has to fill the gap. A constant hazard is the neutral choice —
    it spreads the risk evenly — and it is wrong in a knowable direction, because the
    measured hazard is strongly humped rather than flat.

    B2 is therefore not a straw man. It is what a competent analyst with a binary
    classifier and an NPV model would actually build, and the B2-to-B3 gap is the value
    of knowing the shape.
    """
    clipped = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0 - 1e-12)
    return np.asarray(1.0 - np.power(1.0 - clipped, 1.0 / float(window)), dtype=float)
