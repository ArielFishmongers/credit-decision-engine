"""Is a predicted 2% actually 2%? Measurement first, then correction.

Stage 3. This module is deliberately free of any knowledge of loans, vintages or
hazards: everything here takes a vector of predicted probabilities and a vector of
0/1 outcomes. The loan-level orchestration lives in :mod:`cde.eval.calibration`.

## Why this stage exists at all

Claim 2 of the project. The NPV identity consumes ``h_d(t)`` as a *number*, not as a
rank, so a model with excellent AUC and a systematic 30% overstatement of every
probability misprices every loan in the book while looking perfect on the metric most
credit notebooks report. Discrimination and calibration are different properties and
one does not imply the other.

## Discrimination versus calibration, since the words are easy to confuse

* **Discrimination** -- can the model sort? Do the loans it scores riskiest default
  more often than the ones it scores safest? Measured by AUC. Invariant to any
  monotone rescaling of the predictions, which is precisely the problem: multiply
  every prediction by three and AUC does not move.
* **Calibration** -- is the *level* right? Among rows predicted 2%, do 2% default?
  Measured here. Destroyed by exactly the monotone rescaling AUC ignores.

## Why the base rate makes the naive approach useless

The monthly default hazard averages about 0.08% on the training vintages. Two
consequences drive the design of everything below.

1. **Uniform bins do not work.** Twenty equal-width bins over [0, 1] put essentially
   every row in the first bin, so the reliability "curve" is a single point.
   Quantile bins put an equal number of rows in each. Hence
   ``calibration.binning: quantile``.
2. **The raw Brier score is uninformative.** At a base rate ``b``, forecasting ``b``
   for everything scores ``b(1-b)`` ~= 0.0008, and no realistic model improves on
   that by much in absolute terms -- so the raw score is dominated by how rare the
   event is and two very different models look identical. The Murphy decomposition
   below separates the part that is about calibration from the part that is about
   the base rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

#: Predictions are clipped into this range before any logit. A hazard of exactly 0 or
#: 1 maps to an infinite log-odds, and while the logistic model cannot emit either, a
#: recalibrated or externally supplied vector can.
PROBABILITY_FLOOR = 1e-12
PROBABILITY_CEILING = 1.0 - 1e-12


def _validate(predicted: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to float/int arrays and refuse the inputs that produce silent nonsense."""
    p = np.asarray(predicted, dtype=float)
    o = np.asarray(observed, dtype=float)
    if p.shape != o.shape:
        raise ValueError(f"predicted has shape {p.shape} but observed has shape {o.shape}")
    if p.ndim != 1:
        raise ValueError(f"expected 1-D arrays, got {p.ndim}-D")
    if p.size == 0:
        raise ValueError("no rows to calibrate")
    if not np.isfinite(p).all():
        raise ValueError(f"{int((~np.isfinite(p)).sum())} predicted values are not finite")
    if ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("predicted values must be probabilities in [0, 1]")
    unique = np.unique(o)
    if not np.isin(unique, (0.0, 1.0)).all():
        raise ValueError(f"observed must be 0/1, found values {unique[:5]}")
    return p, o


def _bin_assignments(
    predicted: np.ndarray, bins: int, strategy: str
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each row to a bin, returning (index per row, the edges actually used).

    Duplicate edges are collapsed, so the realised bin count can be lower than asked
    for -- with a discrete or heavily tied prediction vector, several quantiles land on
    the same value. That is reported rather than worked around: silently returning
    twenty bins of which fifteen are empty would make the reliability table look richer
    than the data supports.
    """
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(predicted, np.linspace(0.0, 1.0, bins + 1))
    else:  # pragma: no cover - config validation rejects this first
        raise ValueError(f"unknown binning strategy {strategy!r}")
    edges = np.unique(edges)
    if edges.size < 2:
        # Every prediction identical: one bin spanning the single value.
        edges = np.array([edges[0], np.nextafter(edges[0], np.inf)])
    index = np.clip(np.searchsorted(edges, predicted, side="right") - 1, 0, edges.size - 2)
    return index, edges


def reliability(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    bins: int,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """The reliability table: predicted against observed frequency, by bin.

    One row per non-empty bin, with

    * ``mean_predicted`` -- what the model said, averaged over the bin
    * ``observed_rate`` -- what actually happened
    * ``standard_error`` -- binomial standard error of ``observed_rate``
    * ``z`` -- (observed - predicted) in units of that standard error

    ``z`` is the column to read. A gap of 0.4 percentage points means nothing without
    knowing whether the bin holds 40 rows or 400,000, and ``z`` supplies exactly that
    -- the same idiom as :func:`cde.models.selection.residuals_by_vintage`, for the
    same reason. Bins whose observed rate is zero get a null ``z`` rather than an
    infinite one.
    """
    p, o = _validate(predicted, observed)
    index, edges = _bin_assignments(p, bins, strategy)
    frame = pd.DataFrame({"bin": index, "predicted": p, "observed": o})
    table = (
        frame.groupby("bin")
        .agg(
            rows=("observed", "size"),
            events=("observed", "sum"),
            mean_predicted=("predicted", "mean"),
            observed_rate=("observed", "mean"),
            min_predicted=("predicted", "min"),
            max_predicted=("predicted", "max"),
        )
        .reset_index()
    )
    table["events"] = table["events"].astype(int)
    table["bin_low"] = edges[table["bin"].to_numpy()]
    table["bin_high"] = edges[table["bin"].to_numpy() + 1]
    # Standard error of the observed rate under the model's own predicted probability,
    # not under the observed one. Using the observed rate would give a zero standard
    # error to every empty bin and declare a perfect fit there.
    table["standard_error"] = np.sqrt(
        table["mean_predicted"] * (1.0 - table["mean_predicted"]) / table["rows"]
    )
    table["residual"] = table["observed_rate"] - table["mean_predicted"]
    table["z"] = table["residual"] / table["standard_error"].replace(0.0, np.nan)
    return table


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Count-weighted mean absolute gap between predicted and observed, over bins.

    The headline single number, and it is on the scale of the probabilities themselves:
    at a 0.08% base rate an ECE of 0.0004 is not small, it is half the base rate. Always
    read it next to the base rate, never on its own.
    """
    weights = table["rows"].to_numpy(dtype=float)
    gaps = (table["observed_rate"] - table["mean_predicted"]).abs().to_numpy(dtype=float)
    return float(np.average(gaps, weights=weights))


def maximum_calibration_error(table: pd.DataFrame) -> float:
    """Worst single-bin gap. ECE can hide a badly wrong bin holding few rows."""
    return float((table["observed_rate"] - table["mean_predicted"]).abs().max())


@dataclass(frozen=True)
class BrierParts:
    """The Murphy decomposition of the Brier score.

        Brier  =  reliability  -  resolution  +  uncertainty

    all three on the squared-probability scale, computed against the binned forecast.

    * **reliability** -- the calibration term, and the only one stage 3 can improve.
      Mean squared gap between predicted and observed within a bin. Lower is better;
      zero means perfectly calibrated.
    * **resolution** -- how far the bins' observed rates spread away from the base
      rate. This is the discrimination term. Higher is better, and recalibration
      cannot raise it: any monotone transform of the predictions leaves the bin
      membership, and hence the resolution, essentially untouched. That is the formal
      version of "calibration is not discrimination".
    * **uncertainty** -- ``base_rate * (1 - base_rate)``, the Brier score of forecasting
      the base rate for everything. A property of the data alone; no model changes it.
      At a 0.08% base rate it is about 0.0008, which is why it dominates the raw score.

    ``residual`` is ``brier - (reliability - resolution + uncertainty)``. The identity is
    exact for the *bin-mean* forecast, so this measures how much genuine spread the
    binning threw away. It is reported rather than dropped: a large residual means the
    bins are too coarse to describe the forecast, and the three terms above are then
    describing a cruder model than the one actually fitted.
    """

    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    base_rate: float
    bins_used: int
    rows: int

    @property
    def residual(self) -> float:
        return self.brier - (self.reliability - self.resolution + self.uncertainty)

    @property
    def skill(self) -> float:
        """Brier skill score against forecasting the base rate: ``1 - brier/uncertainty``.

        The scale-free version, and the one to quote at a low base rate. Positive means
        better than the base rate, 0 means no better, negative means worse.
        """
        return 1.0 - self.brier / self.uncertainty if self.uncertainty > 0.0 else 0.0

    def summary(self) -> str:
        return "\n".join(
            [
                f"  rows {self.rows:>12,}   base rate {self.base_rate:.5%}"
                f"   bins used {self.bins_used}",
                f"  Brier          {self.brier:.8f}"
                f"   (skill vs base rate {self.skill:+.4f})",
                f"    reliability  {self.reliability:.8f}   <- the calibration term, "
                f"{self.reliability / self.brier:.1%} of the score",
                f"    resolution   {self.resolution:.8f}   <- discrimination; "
                f"recalibration cannot raise this",
                f"    uncertainty  {self.uncertainty:.8f}   <- the base rate alone",
                f"    residual     {self.residual:+.8f}   <- forecast spread lost to binning",
            ]
        )


def brier_score(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Mean squared error of the probability forecast.

    Computed directly rather than through ``sklearn.metrics.brier_score_loss``, whose
    ``scale_by_half`` argument decides between two conventions differing by a factor of
    two: ``False`` gives the multiclass sum ``(p-y)^2 + ((1-p)-(1-y))^2 == 2(p-y)^2``,
    while ``True``/``"auto"`` gives the conventional binary score. The Murphy
    decomposition in :class:`BrierParts` decomposes the *mean squared error*, so taking
    the other convention would double the score and leave the identity off by exactly
    ``brier``. One line here is cheaper than depending on which default a future sklearn
    ships; ``tests/test_calibrate.py`` asserts the two still agree.
    """
    p, o = _validate(predicted, observed)
    return float(np.mean((p - o) ** 2))


def brier_decomposition(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    bins: int,
    strategy: str = "quantile",
) -> BrierParts:
    """Decompose the Brier score into reliability, resolution and uncertainty."""
    p, o = _validate(predicted, observed)
    table = reliability(p, o, bins=bins, strategy=strategy)
    rows = table["rows"].to_numpy(dtype=float)
    total = float(rows.sum())
    base_rate = float(o.mean())
    predicted_by_bin = table["mean_predicted"].to_numpy(dtype=float)
    observed_by_bin = table["observed_rate"].to_numpy(dtype=float)
    return BrierParts(
        brier=brier_score(p, o),
        reliability=float(np.sum(rows * (predicted_by_bin - observed_by_bin) ** 2) / total),
        resolution=float(np.sum(rows * (observed_by_bin - base_rate) ** 2) / total),
        uncertainty=base_rate * (1.0 - base_rate),
        base_rate=base_rate,
        bins_used=len(table),
        rows=int(total),
    )


# --- recalibration -------------------------------------------------------------------
#
# Both calibrators are monotone, which is what makes them safe to apply to a fitted
# model: neither can reorder two loans. They are NOT equivalent in their effect on
# discrimination, and the difference matters for how stage 5's ablation is read.
#
# Platt is *strictly* monotone, so AUC is unchanged to the last decimal place, and any
# change in NPV is attributable to calibration alone.
#
# Isotonic is only *weakly* monotone: it is a step function, so it maps whole intervals
# to a single value. Measured on this project's synthetic harness it raises AUC by
# 0.007-0.011 -- consistently, across every seed tried. That is not a bug and not noise.
# Isotonic regression works by pooling adjacent violators, which are exactly the regions
# where the raw ranking ran the wrong way against the outcomes; collapsing those to ties
# removes locally wrong orderings, and a tie scores 0.5 where a wrong ordering scores 0.
# So an isotonic B4 arm improves discrimination as well as calibration, and its gap over
# B3 cannot be attributed purely to calibration the way Platt's can. Reported, not
# corrected -- see tests/test_calibrate.py and PROJECT_PLAN.md section 8.
#
# NOT `CalibratedClassifierCV`. It refits the base estimator by cross-validation and
# expects an sklearn estimator interface, whereas predictions here arrive as a plain
# array from `CauseModel.hazard(frame)` -- a fitted object that must not be refitted,
# because refitting it on the holdout would change the very thing being measured.


@runtime_checkable
class Calibrator(Protocol):
    """A fitted monotone map from predicted probability to calibrated probability."""

    name: str

    def fit(self, predicted: np.ndarray, observed: np.ndarray) -> Calibrator: ...

    def transform(self, predicted: np.ndarray) -> np.ndarray: ...

    def describe(self) -> str:
        """One line saying what correction was fitted, for the report and the log."""
        ...


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, PROBABILITY_FLOOR, PROBABILITY_CEILING)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=float)


class PlattCalibrator:
    """Logistic regression of the outcome on ``logit(predicted)``.

    Two parameters, a slope and an intercept, so it can shift the level and stretch or
    compress the spread of the log-odds -- and nothing else. That restraint is the
    point: with 2.4 million holdout rows there is no meaningful overfitting risk, and if
    two parameters fix the miscalibration then the miscalibration was a systematic level
    error, which is a far more useful thing to be able to report than "isotonic fixed
    it".

    A fitted slope near 1 with a non-zero intercept says the model is uniformly off. A
    slope below 1 says it is overconfident -- too extreme at both ends.
    """

    name = "platt"

    def __init__(self, *, tolerance: float = 1e-8, max_iterations: int = 2000) -> None:
        # The same tight tolerance the hazard model uses, for the same reason: sklearn's
        # 1e-4 default stops lbfgs early enough on rare-event data that the fitted
        # probabilities no longer reproduce the observed event count. A recalibrator
        # that has not converged is itself miscalibrated, which would be an absurd way
        # to lose the experiment.
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.model: LogisticRegression | None = None

    def fit(self, predicted: np.ndarray, observed: np.ndarray) -> PlattCalibrator:
        p, o = _validate(predicted, observed)
        if o.sum() == 0:
            raise ValueError(
                "no events in the calibration sample; a calibrator fitted on zero "
                "events would map every prediction to ~0"
            )
        model = LogisticRegression(
            solver="lbfgs",
            # Unpenalised. Regularising a two-parameter recalibrator would shrink it
            # toward the identity, which is exactly the correction being measured.
            # Expressed as C=inf rather than penalty=None: sklearn 1.8 deprecated the
            # latter and will remove it in 1.10.
            C=np.inf,
            tol=self.tolerance,
            max_iter=self.max_iterations,
        )
        model.fit(_logit(p).reshape(-1, 1), o.astype(int))
        self.model = model
        return self

    def transform(self, predicted: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PlattCalibrator.transform before fit")
        p = np.asarray(predicted, dtype=float)
        out = self.model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return np.asarray(out, dtype=float)

    @property
    def slope(self) -> float:
        if self.model is None:
            raise RuntimeError("PlattCalibrator.slope before fit")
        return float(self.model.coef_[0][0])

    @property
    def intercept(self) -> float:
        if self.model is None:
            raise RuntimeError("PlattCalibrator.intercept before fit")
        return float(self.model.intercept_[0])

    def describe(self) -> str:
        return (
            f"platt    slope {self.slope:+.4f}  intercept {self.intercept:+.4f}"
            f"   ({'overconfident' if self.slope < 1.0 else 'underconfident'} "
            f"in the log-odds)"
        )


class IsotonicCalibrator:
    """Free monotone step function fitted by isotonic regression.

    More flexible than Platt: it can correct a *shape* error, where the model is too
    high at one end of the range and too low at the other, which two parameters cannot.
    Where isotonic beats Platt out of time the miscalibration had real structure; where
    it does not, Platt is the better answer for being simpler.

    Two costs, both real. It can fit noise in the holdout. And it is a step function, so
    it collapses whole intervals of predictions to one value -- around 20 distinct
    outputs from 120,000 rows on the synthetic harness -- which discards distinctions
    the hazard model made and, as a side effect, RAISES AUC by pooling the regions where
    the raw ranking was locally wrong. See the note above :class:`PlattCalibrator`: it
    means an isotonic arm is not a clean calibration-only comparison.

    ``out_of_bounds="clip"`` is load-bearing. The default is ``"nan"``, and an
    out-of-time prediction below the smallest or above the largest value seen in the
    holdout is guaranteed to occur -- so the default would silently return NaN for those
    rows and poison every downstream sum. Clipping holds them at the nearest fitted
    value, which never produces a non-number.

    ## The tail, and why clipping alone is not enough

    Clipping is conservative only if the value being clipped *to* is reasonable, and on
    rare-event data the terminal value of an isotonic fit is not. Measured on this
    project's in-time holdout -- 2,427,055 rows carrying 1,834 events -- the top knots
    were estimated on almost nothing:

        x = 0.032054  ->  y = 0.029630     136 calibration rows at or above x
        x = 0.043344  ->  y = 0.029630       2 rows
        x = 0.043626  ->  y = 1.000000       1 row, which happened to default

    So isotonic asserted a **monthly** default hazard of 1.0 -- certain default inside
    the month -- on the strength of a single observation. For scale, the highest observed
    rate in any equal-count bin of ~121,000 rows is 0.004985, making that terminal value
    200 times the largest well-estimated monthly rate anywhere in the data.

    Not a harmless curiosity. Projecting every loan across every month on book reaches
    (loan, age) combinations the observed panel never contains, so the projection's
    hazard range runs past the calibration sample's: 43 of 15.5 million projected rows
    sat above the fit's support and every one was mapped to 1.0. With the prepayment
    hazard added that pushes the total leaving-hazard over 1 and survival negative, which
    the guard in :meth:`cde.models.hazard.HazardModels.survival` refused outright.

    ``min_tail_rows`` addresses the cause rather than the symptom: the output is capped
    at the highest fitted value whose knot is supported by at least that many calibration
    rows. Below about a thousand rows at this base rate the expected event count is under
    one, so a block's observed rate is either 0 or a spike -- noise, and no basis for a
    probability. The cap is disclosed by :meth:`describe` and :attr:`clamped_share`,
    never applied silently.
    """

    name = "isotonic"

    def __init__(self, *, min_tail_rows: int = 0) -> None:
        self.min_tail_rows = min_tail_rows
        self.model: IsotonicRegression | None = None
        #: Fitted value the output is capped at, or None when uncapped.
        self.ceiling: float | None = None
        #: The terminal fitted value before capping, kept so the report can say what was
        #: rejected rather than only what was used.
        self.uncapped_ceiling: float | None = None
        self._clamped = 0
        self._seen = 0

    def fit(self, predicted: np.ndarray, observed: np.ndarray) -> IsotonicCalibrator:
        p, o = _validate(predicted, observed)
        if o.sum() == 0:
            raise ValueError(
                "no events in the calibration sample; a calibrator fitted on zero "
                "events would map every prediction to ~0"
            )
        model = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )
        model.fit(p, o)
        self.model = model
        thresholds = np.asarray(model.X_thresholds_, dtype=float)
        values = np.asarray(model.y_thresholds_, dtype=float)
        self.uncapped_ceiling = float(values.max())
        self.ceiling = None
        if self.min_tail_rows > 0:
            # Calibration rows at or above each knot. searchsorted on the sorted
            # predictions is O(k log n) rather than a k-by-n comparison.
            order = np.sort(p)
            support = len(order) - np.searchsorted(order, thresholds, side="left")
            supported = np.flatnonzero(support >= self.min_tail_rows)
            # values is monotone non-decreasing, so the last supported knot carries the
            # largest defensible value.
            self.ceiling = (
                float(values[supported[-1]]) if supported.size else float(values.min())
            )
        return self

    def transform(self, predicted: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("IsotonicCalibrator.transform before fit")
        out = np.asarray(
            self.model.predict(np.asarray(predicted, dtype=float)), dtype=float
        )
        self._seen += out.size
        if self.ceiling is not None:
            over = out > self.ceiling
            self._clamped += int(over.sum())
            out = np.minimum(out, self.ceiling)
        return out

    @property
    def clamped_share(self) -> float:
        """Share of all transformed rows that hit the tail cap, across every call."""
        return self._clamped / self._seen if self._seen else 0.0

    def describe(self) -> str:
        if self.model is None:
            raise RuntimeError("IsotonicCalibrator.describe before fit")
        knots = np.asarray(self.model.X_thresholds_, dtype=float)
        text = (
            f"isotonic {len(knots)} steps  fitted range "
            f"[{knots.min():.6f}, {knots.max():.6f}]"
        )
        if self.ceiling is None:
            return text + f"   UNCAPPED, terminal value {self.uncapped_ceiling:.6f}"
        return (
            f"{text}   capped at {self.ceiling:.6f} "
            f"(>= {self.min_tail_rows:,} rows of support; terminal value "
            f"{self.uncapped_ceiling:.6f} rejected)"
        )


def build_calibrator(
    method: str,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 2000,
    isotonic_min_tail_rows: int = 0,
) -> Calibrator:
    """Construct an unfitted calibrator by config name."""
    if method == "platt":
        return PlattCalibrator(tolerance=tolerance, max_iterations=max_iterations)
    if method == "isotonic":
        return IsotonicCalibrator(min_tail_rows=isotonic_min_tail_rows)
    raise ValueError(f"unknown calibration method {method!r}")
