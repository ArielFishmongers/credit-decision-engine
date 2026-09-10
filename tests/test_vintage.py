"""Invariants of the vintage curves.

These are the properties that make the charts trustworthy. Each one would fail
silently if the SQL were wrong -- a hazard above 1, bands that do not sum to the
cohort, or Kaplan-Meier coming in *below* the cumulative incidence function,
which is mathematically impossible when a competing risk is present.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from cde.config import load_config
from cde.data.ingest import connect, ingest
from cde.eval.vintage import build, summarise, train_test_split
from cde.features.person_period import build as build_person_period
from tests.conftest import SyntheticLoan


@pytest.fixture
def curves(synthetic: tuple[Path, list[SyntheticLoan]]) -> Iterator[pd.DataFrame]:
    config_path, _ = synthetic
    config = load_config(config_path)
    with connect(config.database) as con:
        ingest(config, con)
        build_person_period(config, con)
        yield train_test_split(config, build(config, con))


def test_hazards_are_probabilities(curves: pd.DataFrame) -> None:
    for column in ("hazard_default", "hazard_prepay"):
        assert curves[column].between(0.0, 1.0).all(), column


def test_survival_is_non_increasing(curves: pd.DataFrame) -> None:
    for _, group in curves.sort_values("loan_age").groupby("vintage"):
        assert group["survival"].diff().dropna().le(1e-12).all()


def test_cumulative_incidence_is_non_decreasing(curves: pd.DataFrame) -> None:
    for _, group in curves.sort_values("loan_age").groupby("vintage"):
        assert group["cif_default"].diff().dropna().ge(-1e-12).all()


def test_counting_shares_partition_the_cohort(curves: pd.DataFrame) -> None:
    """The four counting shares must account for exactly the whole cohort.

    This is a real check, unlike S(t) + F_d(t) + F_p(t) = 1, which is an algebraic
    identity of the estimators and holds however wrong the exit classification is.
    Here a shortfall means a loan left the risk set for a reason nothing counted.
    """
    total = (
        curves["not_yet_exited_share"]
        + curves["naive_default"]
        + curves["naive_prepay"]
        + curves["naive_censored"]
    )
    assert (total - 1.0).abs().max() < 1e-9
    assert (curves["not_yet_exited_share"] >= -1e-9).all()


def test_kaplan_meier_never_undershoots_cumulative_incidence(curves: pd.DataFrame) -> None:
    """F_d(t) <= 1 - KM(t), with equality only when the prepayment hazard is zero
    throughout. The inequality is one-directional, so a violation means the
    arithmetic is wrong rather than the data being unusual."""
    assert (curves["km_default"] >= curves["cif_default"] - 1e-9).all()


def test_kaplan_meier_strictly_overshoots_once_prepayment_exists(
    curves: pd.DataFrame,
) -> None:
    prepaying = curves[curves["cif_prepay"] > 0]
    assert not prepaying.empty, "the fixture should contain a prepayment"
    final = prepaying.sort_values("loan_age").groupby("vintage").tail(1)
    assert (final["km_default"] > final["cif_default"]).any()


def test_naive_understates_when_the_cohort_is_censored(curves: pd.DataFrame) -> None:
    """The counting estimator's bias has a known direction.

    A loan censored at month 20 stays in the denominator forever but can never
    contribute to the numerator after it leaves, so the naive rate is pulled below
    the truth. This fixture censors deliberately -- it exercises every exit reason
    -- so the gap must appear, and must appear downward.

    The complementary check, that the two *agree* when nothing is censored, cannot
    be made here because this cohort is censored by construction. It is verified
    against the real 2000-2008 vintages, where every loan is mature at 120 months
    and the largest disagreement across all eight cohorts is 0.00068.
    """
    final = curves.sort_values("loan_age").groupby("vintage").tail(1)
    censored = final[final["naive_censored"] > 0.0]
    assert not censored.empty, "the fixture should censor some loans before the horizon"
    assert (censored["naive_default"] <= censored["cif_default"] + 1e-9).all()


def test_cohort_size_is_constant_within_a_vintage(curves: pd.DataFrame) -> None:
    """The naive denominator must not move as the risk set changes -- the bug that
    inflated the 2000 vintage's naive rate by 28% was exactly this."""
    assert (curves.groupby("vintage")["cohort_size"].nunique() == 1).all()


def test_summary_reports_the_horizon_row(curves: pd.DataFrame) -> None:
    summary = summarise(curves, load_config().horizon.max_loan_age_months)
    assert len(summary.frame) == curves["vintage"].nunique()
    assert "default %" in summary.summary()
