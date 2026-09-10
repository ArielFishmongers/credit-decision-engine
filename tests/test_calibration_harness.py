"""The stage 3 harness: the split, the scoring, and who is eligible to be valued.

The properties tested here are the ones whose failure would be invisible. A holdout
split that leaks puts the same loan on both sides and every out-of-time number comes
back flattering; an in-sample control that is not actually in-sample removes the one
check that says the harness works at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cde.config import load_config
from cde.eval.calibration import (
    SCOPE_HEADLINE,
    SCOPE_SENSITIVITY,
    SCOPE_SEQUENTIAL,
    _long_frame,
    eligible_for_npv,
    headline_npv,
    holdout_mask,
    score,
    sensitivity_cells,
)

LOANS = np.array([f"F00Q1{i:07d}" for i in range(20_000)])


def test_the_holdout_is_close_to_the_requested_share() -> None:
    for share in (0.1, 0.2, 0.5):
        held = holdout_mask(LOANS, share=share, seed=7)
        assert held.mean() == pytest.approx(share, abs=0.02), share


def test_the_split_is_reproducible_across_calls() -> None:
    first = holdout_mask(LOANS, share=0.2, seed=7)
    second = holdout_mask(LOANS, share=0.2, seed=7)
    assert np.array_equal(first, second)


def test_two_seeds_give_independent_splits() -> None:
    """Not merely different -- *independent*, which is the stronger property.

    The seed keys the hash, so two seeds partition the same loans without reference to
    each other: a loan in the seed-7 holdout has the same ~20% chance of being in the
    seed-8 holdout as any other loan. If the overlap came back near 100% the seed would
    barely be entering the hash, and if it came back near 0% the two splits would be
    anti-correlated -- either would mean the keying is wrong.
    """
    a = holdout_mask(LOANS, share=0.2, seed=7)
    b = holdout_mask(LOANS, share=0.2, seed=8)
    assert not np.array_equal(a, b)
    assert (a & b).sum() / a.sum() == pytest.approx(0.2, abs=0.03)


def test_a_loans_side_of_the_split_does_not_depend_on_who_else_is_in_it() -> None:
    """**The property a shuffle-and-slice does not have, and the reason for hashing.**

    A seeded permutation depends on the size and order of its input, so filtering the
    population, adding a vintage or simply reordering the query would move loans between
    the fitted side and the "unseen" side. The recalibrator would then have been fitted
    on rows the holdout score was computed from -- leakage, with no symptom except
    surprisingly good numbers.

    Hashing each identifier independently makes the assignment a property of the loan.
    """
    full = holdout_mask(LOANS, share=0.2, seed=7)
    subset_order = np.random.default_rng(0).permutation(len(LOANS))[:5_000]
    shuffled = LOANS[subset_order]
    assert np.array_equal(holdout_mask(shuffled, share=0.2, seed=7), full[subset_order])


def test_a_loans_months_never_straddle_the_split() -> None:
    """Split by loan, never by row.

    A loan contributes up to 120 person-period rows sharing every covariate. Splitting
    rows would put the same loan on both sides, which leaks the outcome directly.
    """
    frame = pd.DataFrame({"loan_identifier": np.repeat(LOANS[:500], 120)})
    held = holdout_mask(frame["loan_identifier"].to_numpy(), share=0.2, seed=7)
    per_loan = pd.Series(held).groupby(frame["loan_identifier"].to_numpy()).nunique()
    assert (per_loan == 1).all()


def test_an_impossible_share_is_refused() -> None:
    for share in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="share must be in"):
            holdout_mask(LOANS, share=share, seed=7)


# --- scoring -------------------------------------------------------------------------


def test_score_reports_the_marginal_ratio_and_flags_in_sample_controls() -> None:
    config = load_config()
    rng = np.random.default_rng(3)
    predicted = rng.uniform(0.0001, 0.01, 200_000)
    observed = (rng.random(200_000) < predicted).astype(float)

    honest = score(config, "test (out-of-time)", "raw", predicted, observed)
    assert honest.marginal_ratio == pytest.approx(1.0, abs=0.05)
    assert not honest.is_in_sample

    # Halve every prediction: the ratio must halve with it.
    halved = score(config, "test (out-of-time)", "raw", predicted / 2.0, observed)
    assert halved.marginal_ratio == pytest.approx(honest.marginal_ratio / 2.0, rel=0.01)


def test_the_two_in_sample_controls_are_labelled_as_such() -> None:
    """Isotonic scored on the holdout it was fitted on returns an ECE of exactly zero.

    That is the most impressive-looking and least meaningful number in the whole report,
    so the flag that marks it a control rather than a result is load-bearing.
    """
    config = load_config()
    predicted = np.full(1_000, 0.01)
    observed = np.zeros(1_000)
    observed[:10] = 1.0
    assert score(config, "train (in-sample)", "raw", predicted, observed).is_in_sample
    assert score(config, "holdout (in-time)", "platt", predicted, observed).is_in_sample
    # ...and the two that are genuinely out of sample are not.
    assert not score(config, "holdout (in-time)", "raw", predicted, observed).is_in_sample
    assert not score(config, "test (out-of-time)", "platt", predicted, observed).is_in_sample


# --- eligibility and the grid --------------------------------------------------------


def test_only_loans_observed_from_origination_can_be_valued() -> None:
    """An NPV is a value at the decision point, and a loan first seen at age 6 has none
    on record. Fitting a hazard on it is fine -- that conditions on being alive at the
    month in question -- but projecting a value from a month that was never observed
    would be inventing it."""
    loans = pd.DataFrame(
        {"loan_identifier": ["a", "b", "c"], "first_observed_age": [0, 6, 0]}
    )
    assert list(eligible_for_npv(loans)["loan_identifier"]) == ["a", "c"]


def test_the_sensitivity_grid_spans_the_train_and_test_severities() -> None:
    """The grid has to contain the realised out-of-time severity even though a lender
    could not have known it, because "would the conclusion survive if severity turned
    out to be what it actually was" is the question an interviewer asks."""
    config = load_config()
    cells = sensitivity_cells(config)
    lgds = {cell["lgd"] for cell in cells}
    assert config.economics.lgd in lgds
    assert 0.2407 in lgds  # the measured 2006-2008 figure
    spreads = {cell["discount_spread_bps"] for cell in cells}
    assert config.economics.discount_spread_bps in spreads
    assert len(cells) == len(lgds) * len(spreads)


def test_headline_npv_selects_by_scope_not_by_the_first_row() -> None:
    """Three scopes share one frame — the full-population headline, the later-vintage
    arm that carries the sequential calibrator, and the sensitivity grid. Selecting on
    "the first row's assumptions" silently picked whichever cell happened to sort
    first."""
    npv = pd.DataFrame(
        {
            "variant": ["raw", "raw", "raw"],
            "lgd": [0.133, 0.10, 0.133],
            "discount_spread_bps": [100.0, 50.0, 100.0],
            "npv_per_loan": [1.0, 2.0, 3.0],
            "scope": [SCOPE_SENSITIVITY, SCOPE_HEADLINE, SCOPE_SEQUENTIAL],
        }
    )
    assert list(headline_npv(npv)["npv_per_loan"]) == [2.0]


def test_the_scope_labels_are_distinct() -> None:
    """They are the join keys between the harness, the summary and the figure."""
    assert len({SCOPE_HEADLINE, SCOPE_SEQUENTIAL, SCOPE_SENSITIVITY}) == 3


# --- the projection frame ------------------------------------------------------------


def test_the_projection_repeats_covariates_without_altering_them() -> None:
    """A projected row must be indistinguishable from an observed one at the same age.

    The hazard model is scored twice on different frames — the observed person-period
    rows for the calibration measurement, and a projected frame for the cash flows. If
    building the projection changed a covariate's dtype or value, the two would disagree
    and the NPV would be priced off a slightly different model than the one measured,
    with nothing to show it. So the projection is a strict repeat: every column identical
    down a loan's block, ``loan_age`` the only thing that moves.
    """
    loans = pd.DataFrame(
        {
            "loan_identifier": ["a", "b"],
            "classic_fico": np.array([720, 680], dtype="int64"),
            "original_interest_rate": np.array([6.5, 7.25], dtype="float64"),
            "occupancy_status": ["P", "S"],
            "original_upb": [200_000.0, 150_000.0],
            "loan_age": np.array([0, 6], dtype="int32"),
        }
    )
    long = _long_frame(loans, 12)
    assert len(long) == 24

    for column in loans.columns:
        if column == "loan_age":
            continue
        assert long[column].dtype == loans[column].dtype, column
        for i, loan in enumerate(loans["loan_identifier"]):
            block = long[long["loan_identifier"] == loan][column]
            assert block.nunique() == 1, column
            assert block.iloc[0] == loans[column].iloc[i], column

    # loan_age is overwritten with 0..horizon-1 per loan, regardless of what the source
    # row carried -- an NPV is a value at origination, so the projection starts there.
    for loan in ("a", "b"):
        ages = long[long["loan_identifier"] == loan]["loan_age"].to_numpy()
        np.testing.assert_array_equal(ages, np.arange(12))


def test_the_projection_keeps_loans_in_contiguous_ascending_blocks() -> None:
    """``HazardModels.survival`` treats row order as the time axis, so a projection that
    interleaved loans or descended in age would compute survival across loan
    boundaries — and produce plausible numbers doing it."""
    loans = pd.DataFrame(
        {"loan_identifier": ["a", "b", "c"], "original_upb": [1.0, 2.0, 3.0]}
    )
    long = _long_frame(loans, 5)
    ids = long["loan_identifier"].to_numpy()
    ages = long["loan_age"].to_numpy()
    assert list(ids) == ["a"] * 5 + ["b"] * 5 + ["c"] * 5
    boundary = ids[1:] != ids[:-1]
    assert bool(np.all((ages[1:] > ages[:-1]) | boundary))


# --- figure captions -----------------------------------------------------------------


def test_vintage_labels_come_out_readable() -> None:
    """Captions are baked into the PNG, so a stale cohort label is invisible to every
    lint and test in the repo. Hard-coded vintages went stale twice here — once when the
    horizon moved, once when the control arm reused the crisis arm's caption — so the
    label is derived from config through this function and checked."""
    from cde.eval.figures import vintage_label

    assert vintage_label([2000, 2001, 2002, 2003, 2004]) == "2000–2004"
    assert vintage_label([2006, 2007, 2008]) == "2006–2008"
    assert vintage_label([2000, 2001, 2002]) == "2000–2002"
    assert vintage_label([2004]) == "2004"
    assert vintage_label([]) == ""
    # Non-contiguous must NOT render as a range: 2005 is deliberately omitted from this
    # project's splits, and "2004–2006" would silently claim it was included.
    assert vintage_label([2004, 2006]) == "2004, 2006"
    assert vintage_label([2008, 2006, 2007]) == "2006–2008"
