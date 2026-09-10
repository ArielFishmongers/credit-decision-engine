"""The binary comparison model, and the assumption it forces on a cash-flow model.

The point of these tests is that the binary arm is beaten *fairly*. If it saw different
covariates, a different training sample or a looser optimiser than the hazard model, the
B2-to-B3 gap in the stage 5 ablation would measure that instead of the survival framing —
and claim 1 would be unsupported by an experiment that looked like it supported it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cde.config import load_config
from cde.data.ingest import connect
from cde.features.person_period import build as build_person_period
from cde.models.binary import (
    TARGET,
    fit,
    implied_monthly_hazard,
    loan_level_frame,
)
from cde.models.features import fit_spec
from cde.models.hazard import required_columns
from tests.conftest import SyntheticLoan


@pytest.fixture
def loan_level(synthetic: tuple[Path, list[SyntheticLoan]]):
    from cde.data.ingest import ingest

    config_path, _ = synthetic
    config = load_config(config_path)
    with connect(config.database) as con:
        ingest(config, con)
        build_person_period(config, con)
        yield config, loan_level_frame(config, con, config.data.vintages)


# --- the timing assumption -----------------------------------------------------------


def test_the_implied_hazard_reproduces_the_window_probability() -> None:
    """``1 - (1 - h)^window == p``, which is the only route from a binary classifier to
    a discounted cash flow, and the assumption claim 1 is about."""
    for window in (12, 36, 60):
        for p in (0.0, 0.001, 0.05, 0.25, 0.9):
            h = float(implied_monthly_hazard(np.array([p]), window)[0])
            assert 1.0 - (1.0 - h) ** window == pytest.approx(p, abs=1e-12)
            assert 0.0 <= h <= 1.0


def test_the_implied_hazard_is_monotone_and_survives_certainty() -> None:
    probabilities = np.array([0.0, 0.01, 0.1, 0.5, 0.99, 1.0])
    hazards = implied_monthly_hazard(probabilities, 36)
    assert np.all(np.diff(hazards) > 0)
    # p = 1 must not produce a hazard of exactly 1, which would make survival zero in
    # month 0 and every later cash flow undefined.
    assert hazards[-1] < 1.0
    assert np.isfinite(hazards).all()


# --- fairness to the arm -------------------------------------------------------------


def test_the_binary_design_has_no_loan_age_block(loan_level) -> None:
    """One row per loan means no time axis.

    Feeding ``loan_age = 0`` instead would reuse more code and be worse: the age knots
    come from config and bypass the dead-column check, so seven constant basis columns
    collinear with the intercept would be added silently.
    """
    config, frame = loan_level
    spec = fit_spec(config, frame[frame["resolved"]].reset_index(drop=True),
                    include_loan_age=False)
    assert not spec.include_loan_age
    assert not any(name.startswith("age_spline") for name in spec.feature_names)
    # ...and the hazard model's spec, on the same config, does have it.
    with_age = fit_spec(config, frame)
    assert any(name.startswith("age_spline") for name in with_age.feature_names)
    assert len(with_age.feature_names) > len(spec.feature_names)


def test_the_binary_model_sees_the_same_predictors_as_the_hazard(loan_level) -> None:
    """Everything except the age block must match, or the ablation's B2-to-B3 gap is
    measuring a different feature set."""
    config, frame = loan_level
    usable = frame[frame["resolved"]].reset_index(drop=True)
    binary_names = set(fit_spec(config, usable, include_loan_age=False).feature_names)
    hazard_names = {
        name
        for name in fit_spec(config, frame).feature_names
        if not name.startswith("age_spline")
    }
    assert binary_names == hazard_names


def test_the_fit_converges_and_reproduces_its_own_event_count(loan_level) -> None:
    """Same identity as the hazard model: an unpenalised intercept forces
    ``sum(y - p) = 0`` at the optimum, so a converged fit reproduces the event count."""
    config, frame = loan_level
    model = fit(config, frame)
    if model.events == 0:
        pytest.skip("fixture produced no in-window default")
    assert abs(model.marginal_gap) < 1e-3, model.summary()
    assert model.iterations < config.model.max_iterations


def test_unresolved_loans_are_dropped_and_counted(loan_level) -> None:
    """A loan observed for 20 months cannot answer "did it default within 36". Calling
    it a non-default would understate the rate by the share that stopped reporting."""
    config, frame = loan_level
    model = fit(config, frame)
    assert model.unresolved == int((~frame["resolved"]).sum())
    assert model.rows == int(frame["resolved"].sum())
    assert model.rows + model.unresolved == len(frame)


def test_the_target_only_fires_inside_the_window(loan_level) -> None:
    config, frame = loan_level
    window = config.decision.binary_horizon_months
    fired = frame[frame[TARGET]]
    assert (fired["exit_reason"] == "default").all()
    assert (fired["exit_age"] < window).all()
    # A default beyond the window must NOT count as an in-window event.
    late = frame[(frame["exit_reason"] == "default") & (frame["exit_age"] >= window)]
    assert not late[TARGET].any()


def test_one_row_per_loan_carrying_every_column_the_model_needs(loan_level) -> None:
    config, frame = loan_level
    assert not frame["loan_identifier"].duplicated().any()
    # The fixture's own config, not the project's: `required_columns` is config-derived,
    # and asking the 2-vintage fixture for the production column list demands predictors
    # its config never declared.
    assert set(required_columns(config)) - set(frame.columns) == set()
