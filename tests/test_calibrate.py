"""Calibration measurement and correction.

The trap this file exists to avoid: at a 0.08% base rate almost any calibration number
looks small, so "the ECE is 0.0004" is not evidence of anything on its own. The tests
therefore work on synthetic data where the right answer is known by construction --
generate outcomes from a known probability, distort the probability by a known amount,
and check the machinery recovers what was done to it.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from cde.calibration.calibrate import (
    BrierParts,
    IsotonicCalibrator,
    PlattCalibrator,
    brier_decomposition,
    brier_score,
    build_calibrator,
    expected_calibration_error,
    maximum_calibration_error,
    reliability,
)

#: Close to the real monthly default hazard, so the tests exercise the regime the module
#: was designed for rather than a comfortable 50% one.
BASE_RATE = 0.0008
ROWS = 400_000


def truth(seed: int = 11, rows: int = ROWS, base_rate: float = BASE_RATE) -> tuple:
    """Predicted probabilities that ARE the data-generating probabilities, plus draws.

    Spread over a decade of log-odds around the base rate, so the reliability curve has
    real resolution to measure and is not a single point.
    """
    rng = np.random.default_rng(seed)
    logit_base = np.log(base_rate / (1.0 - base_rate))
    logits = logit_base + rng.normal(0.0, 1.2, rows)
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(rows) < p).astype(float)
    return p, y


def distort(p: np.ndarray, slope: float, shift: float) -> np.ndarray:
    """Apply a known monotone distortion in the log-odds -- exactly what Platt inverts."""
    logits = np.log(p / (1.0 - p))
    return 1.0 / (1.0 + np.exp(-(slope * logits + shift)))


# --- the reliability table -----------------------------------------------------------


def test_uniform_bins_produce_a_curve_driven_by_a_handful_of_loans() -> None:
    """The reason ``calibration.binning`` defaults to quantile.

    The failure is not that uniform binning returns too little. It returns a
    plausible-looking multi-point reliability curve, and that is what makes it
    dangerous: at this base rate the first of twenty equal-width bins holds 99.97% of
    the rows, and the remainder hold tens, single digits, or **one loan**. A bin holding
    one loan reports an observed rate of either 0% or 100%, which plots as catastrophic
    miscalibration and is nothing but the loan count.

    So this test asserts the pathology exists, rather than asserting the curve is
    unusable and leaving the reader to take that on trust.
    """
    p, y = truth()
    uniform = reliability(p, y, bins=20, strategy="uniform")
    assert uniform["rows"].max() / uniform["rows"].sum() > 0.99
    assert (uniform["rows"] < 100).sum() >= 3
    # At least one bin whose observed rate is decided by fewer than five loans.
    assert (uniform["rows"] < 5).any()


def test_quantile_bins_hold_equal_counts() -> None:
    """What quantile binning buys: every bin's standard error is comparable, so the
    ``z`` column means the same thing in every row of the table."""
    p, y = truth()
    quantile = reliability(p, y, bins=20, strategy="quantile")
    assert len(quantile) == 20
    assert quantile["rows"].std() / quantile["rows"].mean() < 0.01


def test_a_perfectly_calibrated_forecast_has_near_zero_error() -> None:
    """When the prediction IS the generating probability, the only gap is sampling noise.

    The bound is set from the noise floor rather than picked: with 20,000 rows a bin and
    a rate near the base rate, the binomial standard error on a bin is about 2e-4.
    """
    p, y = truth()
    table = reliability(p, y, bins=20, strategy="quantile")
    assert expected_calibration_error(table) < 2e-4
    # And the standardised residuals should look like standard normal draws, not like a
    # one-signed bias. This is the check that survives the low base rate.
    assert abs(table["z"].mean()) < 1.0
    assert table["z"].abs().max() < 4.0


def test_a_known_distortion_is_detected_and_signed() -> None:
    """Double every prediction and the reliability table must say so."""
    p, y = truth()
    honest = reliability(p, y, bins=20, strategy="quantile")
    overconfident = reliability(np.clip(p * 2.0, 0, 1), y, bins=20, strategy="quantile")
    assert expected_calibration_error(overconfident) > 5 * expected_calibration_error(honest)
    # Predictions above outcomes, so residuals are negative throughout.
    assert (overconfident["residual"] < 0).mean() > 0.9
    assert maximum_calibration_error(overconfident) > maximum_calibration_error(honest)


def test_non_probability_and_non_binary_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="probabilities in"):
        reliability(np.array([0.5, 1.5]), np.array([0.0, 1.0]), bins=2)
    with pytest.raises(ValueError, match="observed must be 0/1"):
        reliability(np.array([0.5, 0.5]), np.array([0.0, 2.0]), bins=2)
    with pytest.raises(ValueError, match="not finite"):
        reliability(np.array([0.5, np.nan]), np.array([0.0, 1.0]), bins=2)
    with pytest.raises(ValueError, match="shape"):
        reliability(np.array([0.5]), np.array([0.0, 1.0]), bins=2)


# --- the Murphy decomposition --------------------------------------------------------


def test_the_murphy_identity_holds_for_the_binned_forecast() -> None:
    """``Brier = reliability - resolution + uncertainty``, exactly, up to binning.

    The residual is the forecast spread the binning discards, and it is reported rather
    than hidden -- so this test checks it is *small*, which is the claim that twenty
    quantile bins describe this forecast adequately.
    """
    p, y = truth()
    parts = brier_decomposition(p, y, bins=20, strategy="quantile")
    assert isinstance(parts, BrierParts)
    reconstructed = parts.reliability - parts.resolution + parts.uncertainty
    assert parts.brier == pytest.approx(reconstructed + parts.residual)
    assert abs(parts.residual) < 0.05 * parts.brier


def test_uncertainty_is_the_base_rate_alone() -> None:
    p, y = truth()
    parts = brier_decomposition(p, y, bins=20)
    assert parts.uncertainty == pytest.approx(parts.base_rate * (1.0 - parts.base_rate))
    # And it dominates the raw score at this base rate, which is exactly why the raw
    # score is not reported on its own.
    assert parts.uncertainty > 0.9 * parts.brier


def test_our_brier_matches_sklearns_binary_convention() -> None:
    """Pins the convention rather than trusting it.

    ``brier_score_loss(..., scale_by_half=False)`` returns the multiclass sum, which for
    two classes is ``2*(p-y)^2`` -- twice the conventional score. The Murphy
    decomposition decomposes the mean squared error, so taking that convention would
    double the Brier score and leave the identity off by exactly one Brier score. If a
    future sklearn moves its default, this test says so instead of the number moving.
    """
    p, y = truth(seed=3, rows=50_000)
    assert brier_score(p, y) == pytest.approx(float(np.mean((p - y) ** 2)))
    assert brier_score(p, y) == pytest.approx(brier_score_loss(y, p, scale_by_half=True))
    assert brier_score_loss(y, p, scale_by_half=False) == pytest.approx(2 * brier_score(p, y))


def test_platt_improves_reliability_and_leaves_resolution_untouched() -> None:
    """The formal version of "calibration is not discrimination".

    Resolution is a property of how the bins separate outcomes. A *strictly* monotone
    transform leaves quantile bin membership unchanged, so it cannot improve resolution
    however much it improves reliability. Platt is strictly monotone, so this holds to
    six decimal places.

    Deliberately not parametrised over isotonic, which is only weakly monotone -- see
    :func:`test_isotonic_also_improves_discrimination_which_platt_does_not`.
    """
    p, y = truth()
    distorted = distort(p, 0.7, 1.5)
    before = brier_decomposition(distorted, y, bins=20)
    after = brier_decomposition(
        PlattCalibrator().fit(distorted, y).transform(distorted), y, bins=20
    )
    assert after.reliability < before.reliability
    assert after.resolution == pytest.approx(before.resolution, rel=1e-6)


# --- the calibrators -----------------------------------------------------------------


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_a_calibrator_recovers_a_known_distortion(method: str) -> None:
    p, y = truth()
    distorted = distort(p, 0.65, 1.8)
    broken = brier_decomposition(distorted, y, bins=20)
    calibrator = build_calibrator(method).fit(distorted, y)
    fixed = brier_decomposition(calibrator.transform(distorted), y, bins=20)
    assert fixed.reliability < 0.05 * broken.reliability


def test_platt_reports_the_distortion_it_inverted() -> None:
    """The interpretable payoff of a two-parameter calibrator.

    Distorting the log-odds by slope ``a`` means Platt should fit slope ``1/a`` to undo
    it, so the fitted coefficient is a readable statement about what was wrong: a slope
    above 1 says the model was under-dispersed, below 1 that it was overconfident.
    """
    p, y = truth()
    calibrator = PlattCalibrator().fit(distort(p, 0.5, 0.0), y)
    assert calibrator.slope == pytest.approx(2.0, rel=0.1)
    assert "slope" in calibrator.describe()


def test_platt_leaves_the_ranking_exactly_alone() -> None:
    """Platt is strictly monotone, so AUC cannot move at all.

    This is what makes a Platt B4 arm interpretable in stage 5: discrimination is
    provably held fixed, so the whole NPV difference is attributable to calibration.
    """
    p, y = truth(seed=5, rows=120_000)
    distorted = distort(p, 0.7, 1.0)
    calibrated = PlattCalibrator().fit(distorted, y).transform(distorted)
    assert roc_auc_score(y, calibrated) == pytest.approx(roc_auc_score(y, distorted), abs=1e-9)


def test_isotonic_also_improves_discrimination_which_platt_does_not() -> None:
    """A finding, not a defect -- and a caveat on the ablation.

    Isotonic regression pools *adjacent violators*: the intervals where the observed
    rate runs backwards against the prediction, which is exactly where the raw ranking
    was wrong. Collapsing those to a single value turns a wrongly-ordered pair (scoring
    0 in the AUC) into a tie (scoring 0.5), so AUC rises. Measured at +0.006 to +0.011
    across every seed tried, with only ~20 distinct output values surviving from 120,000
    rows.

    The consequence: an isotonic B4 arm improves discrimination as well as calibration,
    so its gap over B3 is not a clean measurement of what calibration alone is worth.
    Platt's is. That asymmetry is why both are reported rather than only the better one.
    """
    gains = []
    for seed in (1, 2, 3, 5, 8):
        p, y = truth(seed=seed, rows=120_000)
        distorted = distort(p, 0.7, 1.0)
        isotonic = IsotonicCalibrator().fit(distorted, y).transform(distorted)
        gains.append(roc_auc_score(y, isotonic) - roc_auc_score(y, distorted))
        # The mechanism, visible directly: a step function with very few steps.
        assert len(np.unique(isotonic)) < 100
    assert all(gain > 0.0 for gain in gains), gains
    assert 0.003 < float(np.mean(gains)) < 0.03, gains


def test_isotonic_clips_outside_its_fitted_range_rather_than_returning_nan() -> None:
    """``out_of_bounds="clip"`` is load-bearing, not a style choice.

    sklearn's default is ``"nan"``. Out-of-time predictions outside the range seen in
    the in-time holdout are guaranteed to occur, and NaN would propagate silently into
    every downstream sum -- an NPV of nan, or worse, a nansum that quietly drops loans.
    """
    p, y = truth(seed=7, rows=80_000)
    calibrator = IsotonicCalibrator().fit(p, y)
    extreme = np.array([0.0, p.min() / 10.0, p.max() * 10.0, 1.0])
    out = calibrator.transform(extreme)
    assert np.isfinite(out).all()
    assert ((out >= 0.0) & (out <= 1.0)).all()
    # Monotone and flat at the ends, which is what clipping means.
    assert out[0] <= out[1] <= out[2] <= out[3]
    assert out[2] == pytest.approx(out[3])


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_a_calibrator_refuses_a_sample_with_no_events(method: str) -> None:
    """Fitting on zero events maps everything to ~0 and reads as a wonderful NPV."""
    p, _ = truth(seed=9, rows=1_000)
    with pytest.raises(ValueError, match="no events"):
        build_calibrator(method).fit(p, np.zeros(1_000))


def test_transform_before_fit_is_an_error_not_a_default() -> None:
    for calibrator in (PlattCalibrator(), IsotonicCalibrator()):
        with pytest.raises(RuntimeError, match="before fit"):
            calibrator.transform(np.array([0.01, 0.02]))


def test_an_unknown_method_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown calibration method"):
        build_calibrator("beta")


# --- the isotonic tail ---------------------------------------------------------------


def tail_spike(rows: int = 200_000, seed: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """A rare-event sample whose single highest prediction happens to default.

    Reproduces the real shape of the failure: at a low base rate, the top of the
    prediction range holds a handful of rows, and whether the topmost one defaulted is
    a coin toss. Isotonic reads that coin toss as certainty.
    """
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0001, 0.03, rows)
    y = (rng.random(rows) < p).astype(float)
    p[-1] = 0.05  # the highest prediction in the sample...
    y[-1] = 1.0   # ...and it defaulted
    return p, y


def test_uncapped_isotonic_asserts_certainty_from_a_single_row() -> None:
    """The bug, pinned. **This is what the code does without the cap.**

    On the real in-time holdout — 2.4 million rows, 1,834 events — the isotonic fit's
    terminal knot was supported by exactly one row, which had defaulted, so isotonic
    returned a *monthly* default hazard of 1.0. The highest well-estimated monthly rate
    anywhere in that data is 0.004985, so the terminal value was 200 times too large,
    from n = 1.

    Nothing about the output looks wrong in isolation: 1.0 is a valid probability and
    isotonic is behaving exactly as specified. It only becomes visible downstream, when
    `h_default + h_prepayment` exceeds 1 and survival would go negative.
    """
    p, y = tail_spike()
    uncapped = IsotonicCalibrator().fit(p, y)
    assert uncapped.uncapped_ceiling == pytest.approx(1.0)
    assert uncapped.transform(np.array([0.05]))[0] == pytest.approx(1.0)
    # And clipping does not save it: the value clipped TO is the problem.
    assert uncapped.transform(np.array([0.9]))[0] == pytest.approx(1.0)
    assert "UNCAPPED" in uncapped.describe()


def test_the_tail_cap_refuses_a_value_its_own_sample_cannot_support() -> None:
    p, y = tail_spike()
    capped = IsotonicCalibrator(min_tail_rows=1_000).fit(p, y)
    assert capped.ceiling is not None
    assert capped.ceiling < 0.5
    assert capped.transform(np.array([0.05, 0.9, 1.0])).max() == pytest.approx(capped.ceiling)
    # The rejected value is kept, so the report can say what was refused and not merely
    # what was used.
    assert capped.uncapped_ceiling == pytest.approx(1.0)
    assert "capped at" in capped.describe()
    assert "rejected" in capped.describe()


def test_the_cap_lands_on_a_knot_with_the_required_support() -> None:
    """The cap is data-derived, not a magic number: it is the highest fitted value whose
    knot at least ``min_tail_rows`` calibration rows sit at or above."""
    p, y = tail_spike()
    for minimum in (100, 1_000, 10_000):
        capped = IsotonicCalibrator(min_tail_rows=minimum).fit(p, y)
        thresholds = np.asarray(capped.model.X_thresholds_, dtype=float)  # type: ignore[union-attr]
        values = np.asarray(capped.model.y_thresholds_, dtype=float)  # type: ignore[union-attr]
        knot = thresholds[np.argmin(np.abs(values - capped.ceiling))]
        assert int((p >= knot).sum()) >= minimum, minimum
    # A stricter requirement can only lower the ceiling.
    ceilings = [
        IsotonicCalibrator(min_tail_rows=m).fit(p, y).ceiling for m in (100, 1_000, 10_000)
    ]
    assert ceilings == sorted(ceilings, reverse=True), ceilings


def test_the_cap_reports_how_often_it_bound() -> None:
    """Applied silently, a cap is just another undisclosed assumption."""
    p, y = tail_spike()
    capped = IsotonicCalibrator(min_tail_rows=1_000).fit(p, y)
    assert capped.clamped_share == 0.0
    capped.transform(np.concatenate([np.full(99, 1e-4), np.array([0.9])]))
    assert capped.clamped_share == pytest.approx(0.01)


def test_the_cap_leaves_the_calibration_it_was_added_to_protect_alone() -> None:
    """It must bind only in the tail. If it moved the bulk of the distribution it would
    be undoing the recalibration rather than making it safe.

    How much it can bind is not a free parameter to be guessed at: the cap sits at the
    last knot with ``min_tail_rows`` rows above it, so at most about
    ``min_tail_rows / n`` of the sample can be capped. That relationship is what is
    asserted here, rather than a threshold picked to make the test pass.
    """
    p, y = truth()
    distorted = distort(p, 0.65, 1.8)
    broken = brier_decomposition(distorted, y, bins=20)
    capped = IsotonicCalibrator(min_tail_rows=1_000).fit(distorted, y)
    fixed = brier_decomposition(capped.transform(distorted), y, bins=20)
    assert fixed.reliability < 0.05 * broken.reliability
    structural_bound = 1_000 / len(p)
    assert capped.clamped_share <= structural_bound
    # ...and the recalibration still works, which is the point of measuring both.
    assert capped.clamped_share < 0.01
