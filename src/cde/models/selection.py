"""Model-selection evidence for stage 2, made reproducible.

Two questions that came up when the fitted default hazard visibly missed the last
ten months on book, and that call for opposite responses -- so neither was answered
by eye.

1. **Is the loan-age spline too coarse?** Answered by refitting across knot counts
   and comparing BIC, which charges for each extra parameter and so does not reward
   flexibility for its own sake.

2. **Or is the misfit cohort-specific?** Answered by breaking the residuals out by
   vintage. A single age curve fitted to five cohorts that each met different
   calendar conditions cannot match all five, however many knots it has -- and no
   covariate available at origination can fix that, because the missing information
   is *when the loan lived*, not *who the borrower was*.

The measured answer was (2): BIC is minimised at the smallest basis tried, while the
2004 cohort's late-age residuals sit at +1.65 standard errors against -0.06 to -0.92
for every other training cohort. 2004's months 45-120 span calendar 2008 onward;
every other training cohort had passed that age band before the crisis. So adding
knots would have chased a calendar effect with an age variable.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cde.config import Config
from cde.models.features import build_matrix, fit_spec
from cde.models.hazard import CAUSE_TARGETS


def knot_study(
    config: Config,
    frame: pd.DataFrame,
    cause: str = "default",
    knot_counts: tuple[int, ...] = (6, 9, 12, 16),
) -> pd.DataFrame:
    """Refit one cause across loan-age knot counts and score each basis.

    Two BIC columns, reported together rather than choosing between them. BIC's
    complexity penalty scales with log(n), and on person-period data it is genuinely
    unclear what n should be: there are 12.1 million rows but only 9,250 events, and
    the effective information is far nearer the event count. Where the two disagree
    the choice would need arguing; here they agree, so it does not matter.
    """
    outcome = frame[CAUSE_TARGETS[cause]].to_numpy(dtype=int)
    rows, events = int(outcome.size), int(outcome.sum())
    records = []
    for knots in knot_counts:
        candidate = replace(config, model=replace(config.model, loan_age_spline_knots=knots))
        spec = fit_spec(candidate, frame)
        matrix = build_matrix(spec, frame)
        model = LogisticRegression(
            solver="lbfgs",
            C=candidate.model.l2_inverse_strength,
            tol=candidate.model.tolerance,
            max_iter=candidate.model.max_iterations,
        )
        model.fit(matrix, outcome)
        predicted = model.predict_proba(matrix)[:, 1]
        log_likelihood = float(
            np.sum(outcome * np.log(predicted) + (1 - outcome) * np.log1p(-predicted))
        )
        parameters = matrix.shape[1] + 1
        records.append(
            {
                "knots": knots,
                "parameters": parameters,
                "log_likelihood": log_likelihood,
                "bic_by_rows": -2 * log_likelihood + parameters * np.log(rows),
                "bic_by_events": -2 * log_likelihood + parameters * np.log(events),
                "marginal_gap": (predicted.sum() - events) / events,
            }
        )
    return pd.DataFrame.from_records(records)


def residuals_by_vintage(
    frame: pd.DataFrame, predicted: np.ndarray, cause: str = "default"
) -> pd.DataFrame:
    """Standardised residuals per (vintage, month on book).

    ``z`` is (observed - predicted) in units of the binomial standard error, so it is
    comparable across cells of very different size. A cohort whose ``z`` is
    systematically one-signed is being mispredicted as a cohort, which no amount of
    age flexibility addresses.
    """
    working = pd.DataFrame(
        {
            "vintage": frame["origination_vintage"].to_numpy(),
            "loan_age": frame["loan_age"].to_numpy(),
            "observed": frame[CAUSE_TARGETS[cause]].to_numpy(dtype=float),
            "predicted": predicted,
        }
    )
    grouped = (
        working.groupby(["vintage", "loan_age"])
        .agg(at_risk=("observed", "size"), observed=("observed", "mean"),
             predicted=("predicted", "mean"))
        .reset_index()
    )
    standard_error = np.sqrt(
        grouped["predicted"] * (1.0 - grouped["predicted"]) / grouped["at_risk"]
    )
    grouped["residual"] = grouped["observed"] - grouped["predicted"]
    grouped["z"] = grouped["residual"] / standard_error.replace(0.0, np.nan)
    return grouped


#: Fraction of the horizon above which ages count as "late" for the cohort diagnostic.
#: Three quarters, so a 120-month horizon means ages 90+ and a 60-month one means 45+.
LATE_AGE_FRACTION = 0.75


def late_age_threshold(horizon_months: int) -> int:
    """Where "late ages" begins, as a fraction of the horizon rather than a constant.

    This used to be a bare default of 45, chosen as the last quarter of a 60-month
    window. When the horizon doubled, 45 silently became the last 63% of the window: the
    diagnostic averaged ages 45-120 while the CLI still printed "ages 45-60", and the
    2004 cohort's signal halved from +3.3 to +1.65 purely from the widened window rather
    than from anything about the data.
    """
    return int(LATE_AGE_FRACTION * horizon_months)


def cohort_bias(residuals: pd.DataFrame, late_age: int) -> pd.DataFrame:
    """Mean standardised residual per cohort, overall and at late ages.

    ``late_age`` is required, not defaulted: a horizon-independent constant here is what
    made the diagnostic quietly measure a different thing after the horizon moved. Get it
    from :func:`late_age_threshold`.

    The returned column is named ``mean_z_late_ages`` regardless of the threshold, with
    the threshold itself carried in ``late_age``, so a caller cannot print a stale
    hard-coded label.
    """
    overall = residuals.groupby("vintage")["z"].mean().rename("mean_z_all_ages")
    late = (
        residuals[residuals["loan_age"] >= late_age]
        .groupby("vintage")["z"]
        .mean()
        .rename("mean_z_late_ages")
    )
    out = pd.concat([overall, late], axis=1).reset_index()
    out["late_age"] = late_age
    return out
