"""Saving and reloading fitted models.

The point of persistence here is not convenience. Stages 3, 4 and 5 must all score
with the *same* fitted model, and the design spec must travel with it: its spline
knots, imputation medians and standardisation statistics were learned on the
training rows, and recomputing any of them on the test set would leak information
across the out-of-time boundary.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cde.config import Config, load_config
from cde.data.ingest import connect, ingest
from cde.features.person_period import build as build_person_period
from cde.models.hazard import HazardModels, fit, load_person_period, required_columns
from cde.models.store import StaleModelError, fingerprint, load, save
from tests.conftest import SyntheticLoan


@pytest.fixture
def fitted(
    synthetic: tuple[Path, list[SyntheticLoan]],
) -> Iterator[tuple[Config, pd.DataFrame, HazardModels]]:
    config_path, _ = synthetic
    config = load_config(config_path)
    with connect(config.database) as con:
        ingest(config, con)
        build_person_period(config, con)
        frame = load_person_period(
            config, con, vintages=config.data.vintages, columns=required_columns(config)
        )
    yield config, frame, fit(config, frame)


def test_reloaded_model_predicts_identically(
    fitted: tuple[Config, pd.DataFrame, HazardModels], tmp_path: Path
) -> None:
    """The whole point: same inputs, same numbers, after a round trip."""
    config, frame, models = fitted
    path = save(config, models, tmp_path / "hazard.joblib")
    reloaded, _ = load(config, path)
    for cause in models.causes:
        np.testing.assert_allclose(
            models.causes[cause].hazard(frame), reloaded.causes[cause].hazard(frame)
        )


def test_the_design_spec_survives_the_round_trip(
    fitted: tuple[Config, pd.DataFrame, HazardModels], tmp_path: Path
) -> None:
    """Knots, medians, standardisation statistics and category levels must all come
    back, or the test set would be encoded against different statistics."""
    config, frame, models = fitted
    reloaded, _ = load(config, save(config, models, tmp_path / "hazard.joblib"))
    assert reloaded.spec.feature_names == models.spec.feature_names
    assert reloaded.spec.medians == models.spec.medians
    assert reloaded.spec.means == models.spec.means
    assert reloaded.spec.deviations == models.spec.deviations
    assert reloaded.spec.categories == models.spec.categories


def test_model_card_records_the_assumptions(
    fitted: tuple[Config, pd.DataFrame, HazardModels], tmp_path: Path
) -> None:
    config, frame, models = fitted
    _, card = load(config, save(config, models, tmp_path / "hazard.joblib"))
    assert card.train_vintages == config.data.train_vintages
    assert card.default_mode == config.default_definition.mode
    assert card.horizon_months == config.horizon.max_loan_age_months
    assert set(card.causes) == set(models.causes)
    assert "default" in card.summary()


@pytest.mark.parametrize(
    "change",
    [
        # 72 rather than 36: the configured calibration horizons run to 60 months, and
        # a 36-month observation window cannot measure 60-month cumulative incidence,
        # so Config's cross-section check refuses that pair before the fingerprint is
        # ever consulted. What this case is about is the horizon moving, not which way.
        pytest.param(lambda c: replace(c, horizon=replace(c.horizon, max_loan_age_months=72)),
                     id="horizon"),
        pytest.param(lambda c: replace(c, model=replace(c.model, loan_age_spline_knots=8)),
                     id="knots"),
        pytest.param(
            lambda c: replace(
                c, default_definition=replace(c.default_definition, mode="termination")
            ),
            id="default_definition",
        ),
        pytest.param(
            lambda c: replace(
                c, person_period=replace(c.person_period, exclude_off_clock_loans=False)
            ),
            id="off_clock",
        ),
    ],
)
def test_loading_under_changed_settings_refuses(
    fitted: tuple[Config, pd.DataFrame, HazardModels], tmp_path: Path, change: object
) -> None:
    """A model fitted to one definition of default, scored under another, gives
    perfectly plausible wrong numbers. So it raises instead."""
    config, frame, models = fitted
    path = save(config, models, tmp_path / "hazard.joblib")
    moved = change(config)  # type: ignore[operator]
    with pytest.raises(StaleModelError, match="different settings"):
        load(moved, path)
    # ...but can be loaded deliberately for inspection.
    reloaded, _ = load(moved, path, require_match=False)
    assert set(reloaded.causes) == set(models.causes)


def test_stage_four_assumptions_do_not_invalidate_a_fit(
    fitted: tuple[Config, pd.DataFrame, HazardModels], tmp_path: Path
) -> None:
    """Every economic assumption must be free to vary without forcing a refit.

    The discount rate, severity, the recovery lag and the servicing cost belong to the
    cash-flow calculation, not to the hazard. Stage 4 sweeps all of them, and a refit
    triggered by a sensitivity case would make the swept results incomparable with each
    other -- the opposite of what a sweep is for.

    Every key in ``economics`` gets a line here, so an addition that accidentally lands
    inside the fingerprint is caught by this test rather than by a two-and-a-half-minute
    refit appearing in the middle of a sweep.
    """
    config, frame, models = fitted
    path = save(config, models, tmp_path / "hazard.joblib")
    repriced = replace(
        config,
        economics=replace(
            config.economics,
            annual_discount_rate=0.12,
            discount_spread_bps=400.0,
            # Kept below loss_disposition_probability: the pair implies severity given
            # disposition, and above 1 that is a default recovering more than it owed,
            # which Config refuses outright.
            lgd=0.55,
            loss_disposition_probability=0.95,
            recovery_lag_months=48,
            servicing_cost_annual_bps=120.0,
        ),
    )
    assert fingerprint(repriced) == fingerprint(config)
    load(repriced, path)  # must not raise


def test_calibration_settings_do_not_invalidate_a_fit(
    fitted: tuple[Config, pd.DataFrame, HazardModels], tmp_path: Path
) -> None:
    """Stage 3 assesses a fitted hazard; it never changes what the hazard is.

    So the bin count, the binning strategy, the reported horizons, the holdout and the
    recalibration methods must all be free to move without invalidating a saved model.
    If they were not, every calibration experiment would silently refit the very thing
    it was trying to measure.
    """
    config, frame, models = fitted
    path = save(config, models, tmp_path / "hazard.joblib")
    reassessed = replace(
        config,
        calibration=replace(
            config.calibration,
            bins=50,
            binning="uniform",
            horizons=(6, 18),
            holdout_share=0.35,
            holdout_seed=1,
            methods=("platt",),
        ),
    )
    assert fingerprint(reassessed) == fingerprint(config)
    load(reassessed, path)  # must not raise


def test_missing_file_says_how_to_make_one(
    fitted: tuple[Config, pd.DataFrame, HazardModels], tmp_path: Path
) -> None:
    config, _, _ = fitted
    with pytest.raises(FileNotFoundError, match="cde hazard"):
        load(config, tmp_path / "absent.joblib")
