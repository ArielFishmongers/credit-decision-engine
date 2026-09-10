"""Turn person-period rows into a design matrix for the hazard model.

Stage 2. Three groups of columns, each treated differently and each for a reason:

* **Loan age** gets a cubic spline basis. This is the one non-negotiable choice in
  stage 2. A single linear term in loan age asserts that the monthly default risk
  rises (or falls) at a constant rate for ten years, and the measured hazard does
  nothing of the kind -- it is humped, and the hump sits at month 22 for the 2008
  cohort and month 44 for the 2006 one. A slope fitted through that is wrong at
  every point. Monthly dummies were the obvious alternative and lose: 9,250 default
  events spread over 121 ages leaves the first months with almost none, so those
  dummies would be unstable or perfectly separated.

* **Non-linear predictors** get the same spline treatment. FICO is the clearest
  case -- the log-odds gap between 620 and 660 is not the gap between 760 and 800 --
  and the same holds for loan-to-value and debt-to-income around their policy
  thresholds.

* **Everything else** is linear or one-hot.

Two behaviours worth knowing about, both automatic:

* Columns that are entirely missing, or that hold a single value, are dropped with
  a reason. They are not hard-coded, because which columns are dead depends on the
  vintages selected: ``VANTAGESCORE 4.0`` is 100% null on 2000-2008 originations but
  populated on recent ones, and ``AMORTIZATION TYPE`` is constant here only because
  the Standard dataset filters to fixed-rate loans.

* Missing numeric values are median-imputed **and** flagged with an indicator, so
  the model can use the fact that a value was missing. On this data a missing DTI is
  not missing at random -- Freddie Mac suppresses ratios above 65%, which is to say
  it suppresses exactly the riskiest borrowers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import SplineTransformer

from cde.config import Config

LOAN_AGE = "loan_age"
#: Derived before the matrix is built. Loan size spans two orders of magnitude and
#: its effect is closer to proportional than additive, so the log is the natural scale.
LOG_UPB = "log_original_upb"


@dataclass
class DesignSpec:
    """A fitted feature specification. Reused verbatim on the test set.

    Holding the spline knots, imputation medians and category levels here -- rather
    than recomputing them per split -- is what stops the test set from being encoded
    against its own statistics, which would leak information across the out-of-time
    boundary.
    """

    spline_transformers: dict[str, SplineTransformer] = field(default_factory=dict)
    medians: dict[str, float] = field(default_factory=dict)
    #: Mean and standard deviation of each linear column, learned on the training
    #: rows only. Applied on both splits so the test set is never encoded against
    #: its own statistics.
    means: dict[str, float] = field(default_factory=dict)
    deviations: dict[str, float] = field(default_factory=dict)
    categories: dict[str, list[str]] = field(default_factory=dict)
    linear_columns: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    #: Whether the loan-age spline block is part of this design.
    #:
    #: False for the stage 5 binary comparison model, which has one row per loan and no
    #: time axis at all. Carried on the spec rather than passed to
    #: :func:`build_matrix` separately so a matrix can never be built under a different
    #: assumption from the spec it was encoded with.
    #:
    #: The alternative -- feeding ``loan_age = 0`` for every row -- reuses more code and
    #: is worse: the age knots come from config and bypass the dead-column check, so it
    #: would silently add seven constant basis columns collinear with the intercept, and
    #: a reader would have no way to tell the binary model had no time axis.
    include_loan_age: bool = True
    dropped: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"design matrix: {len(self.feature_names)} features"]
        groups = {
            "loan-age spline": sum(1 for n in self.feature_names if n.startswith("age_spline")),
            "predictor splines": sum(
                1 for n in self.feature_names if "_spline_" in n and not n.startswith("age_spline")
            ),
            "linear": sum(1 for n in self.feature_names if n.startswith("lin_")),
            "missing indicators": sum(1 for n in self.feature_names if n.endswith("_is_missing")),
            "one-hot": sum(1 for n in self.feature_names if n.startswith("cat_")),
        }
        for label, count in groups.items():
            lines.append(f"  {label:<20} {count:>4}")
        if self.dropped:
            lines.append("  dropped:")
            for column, reason in sorted(self.dropped.items()):
                lines.append(f"    {column:<48} {reason}")
        return "\n".join(lines)


def add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "original_upb" in out.columns:
        out[LOG_UPB] = np.log(out["original_upb"].clip(lower=1.0))
    return out


def _usable(frame: pd.DataFrame, column: str, spec: DesignSpec) -> bool:
    """Reject a column that cannot inform the model, and say why."""
    if column not in frame.columns:
        spec.dropped[column] = "not present in the person-period table"
        return False
    series = frame[column]
    if series.isna().all():
        spec.dropped[column] = "entirely missing on these vintages"
        return False
    if series.nunique(dropna=True) < 2:
        only = series.dropna().iloc[0] if not series.dropna().empty else "?"
        spec.dropped[column] = f"single value ({only!r}) -- no variation to fit"
        return False
    return True


def fit_spec(
    config: Config, frame: pd.DataFrame, *, include_loan_age: bool = True
) -> DesignSpec:
    """Learn the encoding from the training rows only.

    ``include_loan_age=False`` drops the loan-age spline block, for a model with one row
    per loan and no time axis. Every other column is encoded identically, which is what
    lets the stage 5 ablation attribute the B2-to-B3 gap to the survival framing rather
    than to a different feature set.
    """
    model = config.model
    spec = DesignSpec(include_loan_age=include_loan_age)
    frame = add_derived(frame)

    if include_loan_age:
        knots = np.linspace(
            0.0, float(config.horizon.max_loan_age_months), model.loan_age_spline_knots
        )
        age_spline = SplineTransformer(
            degree=3, knots=knots.reshape(-1, 1), extrapolation="linear", include_bias=False
        )
        age_spline.fit(frame[[LOAN_AGE]].to_numpy(dtype=float))
        spec.spline_transformers[LOAN_AGE] = age_spline

    for column in model.spline_columns:
        if not _usable(frame, column, spec):
            continue
        values = frame[column].astype(float)
        spec.medians[column] = float(values.median())
        filled = values.fillna(spec.medians[column]).to_numpy(dtype=float).reshape(-1, 1)
        # Knots at quantiles rather than evenly spaced: FICO is dense in 680-780 and
        # sparse below 600, and evenly spaced knots would spend resolution where
        # there are no loans.
        quantiles = np.linspace(0.0, 1.0, model.loan_age_spline_knots)
        column_knots = np.unique(np.quantile(filled, quantiles)).reshape(-1, 1)
        transformer = SplineTransformer(
            degree=3 if len(column_knots) > 3 else 1,
            knots=column_knots,
            extrapolation="linear",
            include_bias=False,
        )
        transformer.fit(filled)
        spec.spline_transformers[column] = transformer

    for column in model.linear_columns:
        if not _usable(frame, column, spec):
            continue
        spec.linear_columns.append(column)
        values = frame[column].astype(float)
        spec.medians[column] = float(values.median())
        # Standardised, and this is not cosmetic. ORIGINAL LOAN TERM runs 60-360
        # while the spline bases and one-hots sit in [0, 1]; that 360x spread wrecks
        # the conditioning of the problem, and lbfgs stopped on its relative
        # function tolerance -- reporting no warning -- with the fitted
        # probabilities still ~1% off the observed event count. It also makes the
        # L2 penalty coherent: unscaled, a feature measured in hundreds gets a tiny
        # coefficient and is effectively unpenalised, while a 0/1 dummy is not.
        filled = values.fillna(spec.medians[column])
        spec.means[column] = float(filled.mean())
        deviation = float(filled.std(ddof=0))
        spec.deviations[column] = deviation if deviation > 0.0 else 1.0

    loans = frame.drop_duplicates("loan_identifier")
    for column in model.categorical_columns:
        if not _usable(frame, column, spec):
            continue
        shares = loans[column].astype("string").fillna("MISSING").value_counts(normalize=True)
        kept = sorted(shares[shares >= model.min_category_share].index.tolist())
        if len(kept) < 2:
            spec.dropped[column] = (
                f"no level reaches the {model.min_category_share:.1%} floor"
            )
            continue
        spec.categories[column] = kept

    spec.feature_names = _names(spec)
    return spec


def _names(spec: DesignSpec) -> list[str]:
    names: list[str] = []
    if spec.include_loan_age:
        age = spec.spline_transformers[LOAN_AGE]
        names += [f"age_spline_{i}" for i in range(age.n_features_out_)]
    for column, transformer in spec.spline_transformers.items():
        if column == LOAN_AGE:
            continue
        names += [f"{column}_spline_{i}" for i in range(transformer.n_features_out_)]
        names.append(f"{column}_is_missing")
    for column in spec.linear_columns:
        names.append(f"lin_{column}")
        names.append(f"{column}_is_missing")
    for column, levels in spec.categories.items():
        # First level dropped as the reference, so the intercept is interpretable.
        names += [f"cat_{column}={level}" for level in levels[1:]]
        names.append(f"cat_{column}=OTHER")
    return names


def build_matrix(spec: DesignSpec, frame: pd.DataFrame) -> sparse.csr_matrix:
    """Encode rows using an already-fitted spec.

    Returned sparse: the one-hot block is overwhelmingly zeros, and on 12.1 million
    training rows a dense float64 matrix of this width would not fit in memory.
    """
    frame = add_derived(frame)
    blocks: list[sparse.spmatrix] = []

    if spec.include_loan_age:
        age = spec.spline_transformers[LOAN_AGE]
        blocks.append(
            sparse.csr_matrix(
                age.transform(frame[[LOAN_AGE]].to_numpy(dtype=float)), dtype=np.float32
            )
        )

    for column, transformer in spec.spline_transformers.items():
        if column == LOAN_AGE:
            continue
        values = frame[column].astype(float)
        missing = values.isna().to_numpy()
        filled = values.fillna(spec.medians[column]).to_numpy(dtype=float).reshape(-1, 1)
        blocks.append(sparse.csr_matrix(transformer.transform(filled), dtype=np.float32))
        blocks.append(sparse.csr_matrix(missing.astype(np.float32).reshape(-1, 1)))

    for column in spec.linear_columns:
        values = frame[column].astype(float)
        missing = values.isna().to_numpy()
        filled = values.fillna(spec.medians[column])
        standardised = (
            (filled - spec.means[column]) / spec.deviations[column]
        ).to_numpy(dtype=np.float32)
        blocks.append(sparse.csr_matrix(standardised.reshape(-1, 1)))
        blocks.append(sparse.csr_matrix(missing.astype(np.float32).reshape(-1, 1)))

    for column, levels in spec.categories.items():
        observed = frame[column].astype("string").fillna("MISSING")
        for level in levels[1:]:
            blocks.append(
                sparse.csr_matrix((observed == level).to_numpy(np.float32).reshape(-1, 1))
            )
        # A level unseen in training, or one too rare to keep, lands in OTHER rather
        # than silently encoding as the reference level.
        blocks.append(
            sparse.csr_matrix(
                (~observed.isin(levels)).to_numpy(np.float32).reshape(-1, 1)
            )
        )

    matrix = sparse.hstack(blocks, format="csr", dtype=np.float32)
    if matrix.shape[1] != len(spec.feature_names):
        raise ValueError(
            f"design matrix has {matrix.shape[1]} columns but the spec names "
            f"{len(spec.feature_names)}"
        )
    return matrix
