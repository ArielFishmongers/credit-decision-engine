"""The stage 2 hazard model.

The synthetic fixture is far too small to say anything about credit risk, so nothing
here asserts a coefficient's sign or size. What it does assert are the properties
that must hold for the output to be a hazard at all, plus the two behaviours that
would otherwise fail silently: dead columns being dropped with a reason, and the
fitted probabilities summing to the observed event count.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cde.config import Config, load_config
from cde.data.ingest import connect, ingest
from cde.features.person_period import build as build_person_period
from cde.models.features import build_matrix, fit_spec
from cde.models.hazard import CAUSE_TARGETS, fit, load_person_period, required_columns
from tests.conftest import SyntheticLoan


@pytest.fixture
def rows(synthetic: tuple[Path, list[SyntheticLoan]]) -> Iterator[tuple[Config, pd.DataFrame]]:
    config_path, _ = synthetic
    config = load_config(config_path)
    with connect(config.database) as con:
        ingest(config, con)
        build_person_period(config, con)
        frame = load_person_period(
            config, con, vintages=config.data.vintages, columns=required_columns(config)
        )
    yield config, frame


def test_required_columns_are_real_and_unique(rows: tuple[Config, pd.DataFrame]) -> None:
    config, frame = rows
    needed = required_columns(config)
    assert len(needed) == len(set(needed))
    assert set(needed) <= set(frame.columns)
    for target in CAUSE_TARGETS.values():
        assert target in needed


def test_dead_columns_are_dropped_with_a_reason(rows: tuple[Config, pd.DataFrame]) -> None:
    """Which columns are dead depends on the vintages, so this must be detected at
    fit time rather than hard-coded: VANTAGESCORE 4.0 is entirely null on 2000-2008
    originations but populated on recent ones."""
    config, frame = rows
    frame = frame.copy()
    frame["classic_fico"] = np.nan  # as VANTAGESCORE 4.0 is on the real train set
    frame["original_interest_rate"] = 6.0  # as AMORTIZATION TYPE is: one value only
    spec = fit_spec(config, frame)
    assert "entirely missing" in spec.dropped["classic_fico"]
    assert "single value" in spec.dropped["original_interest_rate"]
    assert "classic_fico" not in spec.spline_transformers
    assert "original_interest_rate" not in spec.linear_columns


def test_matrix_width_matches_the_named_features(rows: tuple[Config, pd.DataFrame]) -> None:
    config, frame = rows
    spec = fit_spec(config, frame)
    matrix = build_matrix(spec, frame)
    assert matrix.shape == (len(frame), len(spec.feature_names))


def test_loan_age_always_gets_a_spline_basis(rows: tuple[Config, pd.DataFrame]) -> None:
    """The one non-negotiable choice in stage 2. A linear age term would assert a
    constant monthly change in risk over ten years, and the measured hazard is
    humped."""
    config, frame = rows
    spec = fit_spec(config, frame)
    assert sum(1 for n in spec.feature_names if n.startswith("age_spline")) >= 4
    assert "lin_loan_age" not in spec.feature_names


def test_unseen_category_levels_fall_into_other(rows: tuple[Config, pd.DataFrame]) -> None:
    """Out-of-time test rows can carry a level absent from training. It must land in
    OTHER, not silently encode as the reference level."""
    config, frame = rows
    spec = fit_spec(config, frame)
    if not spec.categories:
        pytest.skip("fixture retained no categorical column")
    column = next(iter(spec.categories))
    unseen = frame.copy()
    unseen[column] = "NEVER_SEEN"
    matrix = build_matrix(spec, unseen)
    other = spec.feature_names.index(f"cat_{column}=OTHER")
    assert matrix[:, other].toarray().ravel().min() == 1.0


def test_predicted_events_match_observed_events(rows: tuple[Config, pd.DataFrame]) -> None:
    """A converged logistic regression with an unpenalised intercept satisfies
    sum(y - p) = 0 exactly, whatever the penalty on the slopes.

    This is the check that caught sklearn's default tol=1e-4 leaving the real
    training fit ~1% off at the margin -- an optimiser artefact that would have
    been read in stage 3 as the model being overconfident.
    """
    config, frame = rows
    models = fit(config, frame)
    for cause, fitted in models.causes.items():
        assert abs(fitted.marginal_gap) < 1e-3, f"{cause}: gap {fitted.marginal_gap:.3%}"


def test_sklearn_default_tolerance_really_does_break_this() -> None:
    """Pin the behaviour the config's tolerance exists to avoid, so that a future
    "simplify the config" does not quietly reintroduce it."""
    rng = np.random.default_rng(0)
    n, p = 200_000, 8
    features = rng.normal(size=(n, p))
    beta = rng.normal(scale=0.4, size=p)
    outcome = (
        rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-7.0 + features @ beta)))
    ).astype(int)

    loose = LogisticRegression(solver="lbfgs", C=100.0, tol=1e-4, max_iter=200)
    loose.fit(features, outcome)
    tight = LogisticRegression(solver="lbfgs", C=100.0, tol=1e-8, max_iter=2000)
    tight.fit(features, outcome)

    events = outcome.sum()
    loose_gap = abs(loose.predict_proba(features)[:, 1].sum() - events) / events
    tight_gap = abs(tight.predict_proba(features)[:, 1].sum() - events) / events
    assert loose_gap > 0.01, "sklearn's default no longer misses the margin -- revisit the config"
    assert tight_gap < 1e-4


def test_hazards_are_probabilities(rows: tuple[Config, pd.DataFrame]) -> None:
    config, frame = rows
    models = fit(config, frame)
    for fitted in models.causes.values():
        hazard = fitted.hazard(frame)
        assert np.isfinite(hazard).all()
        assert ((hazard > 0.0) & (hazard < 1.0)).all()


def test_survival_falls_and_incidence_rises(rows: tuple[Config, pd.DataFrame]) -> None:
    config, frame = rows
    models = fit(config, frame)
    curves = models.survival(frame)
    for _, group in curves.groupby("loan_identifier"):
        assert group["survival"].diff().dropna().le(1e-12).all()
        for cause in models.causes:
            assert group[f"cif_{cause}"].diff().dropna().ge(-1e-12).all()
    assert (curves["survival"] >= 0.0).all()
    total = curves["survival"] + sum(curves[f"cif_{c}"] for c in models.causes)
    assert (total - 1.0).abs().max() < 1e-9


def test_the_design_matrix_is_reasonably_scaled(rows: tuple[Config, pd.DataFrame]) -> None:
    """Pins the scaling fix.

    ORIGINAL LOAN TERM runs 60-360 raw, while spline bases and one-hots sit in
    [0, 1]. Unscaled, that spread stopped lbfgs on its relative function tolerance
    -- silently, with no ConvergenceWarning -- leaving the fitted probabilities
    ~1% off the observed event count on the real training set.
    """
    config, frame = rows
    spec = fit_spec(config, frame)
    matrix = build_matrix(spec, frame)
    largest = float(abs(matrix).max())
    assert largest < 50.0, f"largest feature value {largest:,.1f}; linear columns unscaled?"


def test_linear_columns_use_training_statistics_only(
    rows: tuple[Config, pd.DataFrame],
) -> None:
    """The spec is fitted once and reused, so a test set is never standardised
    against its own mean. Re-standardising per split would leak information across
    the out-of-time boundary, which is the one thing stage 5 must not do."""
    config, frame = rows
    spec = fit_spec(config, frame)
    if not spec.linear_columns:
        pytest.skip("fixture retained no linear column")
    column = spec.linear_columns[0]
    shifted = frame.copy()
    shifted[column] = shifted[column].astype(float) + 100.0
    index = spec.feature_names.index(f"lin_{column}")
    original = build_matrix(spec, frame)[:, index].toarray().mean()
    moved = build_matrix(spec, shifted)[:, index].toarray().mean()
    assert moved > original + 1e-6, "shifting the input did not move the encoding"


def test_knot_study_scores_every_basis(rows: tuple[Config, pd.DataFrame]) -> None:
    from cde.models.selection import knot_study

    config, frame = rows
    table = knot_study(config, frame, knot_counts=(4, 6))
    assert list(table["knots"]) == [4, 6]
    # More knots must buy more parameters and cannot reduce the in-sample likelihood.
    assert table["parameters"].is_monotonic_increasing
    assert table["log_likelihood"].iloc[-1] >= table["log_likelihood"].iloc[0] - 1e-6


def test_cohort_bias_flags_a_cohort_specific_residual(
    rows: tuple[Config, pd.DataFrame],
) -> None:
    """The diagnostic that distinguished a calendar effect from a spline problem."""
    from cde.models.selection import cohort_bias, late_age_threshold, residuals_by_vintage

    config, frame = rows
    models = fit(config, frame)
    residuals = residuals_by_vintage(frame, models.causes["default"].hazard(frame))
    assert {"vintage", "loan_age", "z"} <= set(residuals.columns)
    bias = cohort_bias(residuals, late_age_threshold(config.horizon.max_loan_age_months))
    assert set(bias["vintage"]) == set(frame["origination_vintage"].unique())


def test_iteration_cap_warning_does_not_cry_wolf(rows: tuple[Config, pd.DataFrame]) -> None:
    """A fit that converged well inside the cap must not claim it hit the cap.

    `HazardModels.max_iterations` used to default to 0, which made the check
    `iterations >= max_iterations` true for every fit, so the summary flagged every
    cause. A warning that is always on is worse than no warning.
    """
    config, frame = rows
    models = fit(config, frame)
    for fitted in models.causes.values():
        assert fitted.iterations < config.model.max_iterations
    assert "HIT THE ITERATION CAP" not in models.summary()


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_iteration_cap_warning_fires_when_it_should(rows: tuple[Config, pd.DataFrame]) -> None:
    """And the flag must still work: starve the optimiser and it should appear.

    sklearn raises its own ConvergenceWarning here, which is the point -- and worth
    noting that it does so only when the *iteration cap* is hit. The failure that
    actually bit this project was the other exit path, scipy stopping on its
    relative function tolerance, which warns about nothing at all.
    """
    from dataclasses import replace

    config, frame = rows
    starved = replace(config, model=replace(config.model, max_iterations=1))
    models = fit(starved, frame)
    assert "HIT THE ITERATION CAP" in models.summary()


def test_max_iterations_must_be_supplied(rows: tuple[Config, pd.DataFrame]) -> None:
    """No default, so the flag can never be evaluated against a placeholder."""
    from cde.models.hazard import HazardModels

    config, frame = rows
    models = fit(config, frame)
    with pytest.raises(TypeError):
        HazardModels(causes=models.causes, spec=models.spec)  # type: ignore[call-arg]


def test_survival_is_invariant_to_input_row_order(rows: tuple[Config, pd.DataFrame]) -> None:
    """The central property. Row order is the time axis inside survival(), so a
    shuffled frame must not change the answer -- and must come back aligned to the
    order it was passed in."""
    config, frame = rows
    models = fit(config, frame)

    ordered = models.survival(frame)
    shuffled = frame.sample(frac=1.0, random_state=0)
    from_shuffled = models.survival(shuffled)

    # Returned in the caller's order, so the keys line up row for row.
    assert from_shuffled["loan_identifier"].tolist() == shuffled["loan_identifier"].tolist()
    assert from_shuffled["loan_age"].tolist() == shuffled["loan_age"].tolist()

    # And the values agree once both are put on the same (loan, age) key.
    key = ["loan_identifier", "loan_age"]
    left = ordered.set_index(key).sort_index()
    right = from_shuffled.set_index(key).sort_index()
    for column in ("survival", "survival_entering", "cif_default", "cif_prepayment"):
        pd.testing.assert_series_equal(left[column], right[column], check_exact=False, rtol=1e-9)


def test_shuffled_input_would_have_been_silently_wrong(
    rows: tuple[Config, pd.DataFrame],
) -> None:
    """Demonstrates the bug that was fixed, so the guard cannot be removed quietly.

    Running the accumulation on shuffled rows *without* reordering produces numbers
    that are plausible -- between 0 and 1, no NaN, no exception -- and wrong.
    """
    config, frame = rows
    models = fit(config, frame)
    shuffled = frame.sample(frac=1.0, random_state=0)

    correct = models.survival(shuffled).reset_index(drop=True)

    # The old behaviour: accumulate in whatever order the rows arrived.
    naive = shuffled[["loan_identifier", "loan_age"]].copy()
    leaving = pd.Series(
        sum(f.hazard(shuffled) for f in models.causes.values()), index=shuffled.index
    )
    naive["survival"] = (1.0 - leaving).groupby(naive["loan_identifier"]).cumprod()
    naive = naive.reset_index(drop=True)

    assert naive["survival"].between(0.0, 1.0).all(), "the wrong answer looks entirely valid"
    assert not np.allclose(naive["survival"], correct["survival"]), (
        "shuffling no longer changes the naive result -- this test has stopped testing anything"
    )


def test_month_gaps_are_rejected(rows: tuple[Config, pd.DataFrame]) -> None:
    """A hole inside a loan's span makes shift(1) skip a month of exposure."""
    config, frame = rows
    models = fit(config, frame)
    victim = frame["loan_identifier"].iloc[0]
    ages = sorted(frame.loc[frame["loan_identifier"] == victim, "loan_age"])
    if len(ages) < 4:
        pytest.skip("fixture loan too short to punch a gap in")
    middle = ages[len(ages) // 2]
    holed = frame[
        ~((frame["loan_identifier"] == victim) & (frame["loan_age"] == middle))
    ]
    with pytest.raises(ValueError, match="month gap"):
        models.survival(holed)


def test_a_late_starting_loan_is_not_mistaken_for_a_gap(
    rows: tuple[Config, pd.DataFrame],
) -> None:
    """Left truncation is normal -- 17% of real loans enter after age 0 -- and must
    not be confused with a hole in the middle."""
    config, frame = rows
    models = fit(config, frame)
    late = frame[frame["loan_age"] >= 3]
    result = models.survival(late)
    assert len(result) == len(late)
