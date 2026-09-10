"""Measured loss given default.

The tests here are about two errors that would both produce plausible NPV numbers:
conflating severity-at-disposition with severity-per-default-event, and letting realised
test-period losses leak into an input the model could not have known.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from cde.config import Config, load_config
from cde.data.ingest import connect, ingest
from cde.economics.lgd import LgdEstimate, by_vintage, degradation, estimate, estimate_splits
from cde.features.person_period import build as build_person_period
from tests.conftest import SyntheticLoan

REAL_DB = Path("data/interim/cde.duckdb")
requires_real_data = pytest.mark.skipif(
    not REAL_DB.is_file(), reason="no ingested data; run `cde ingest`"
)


@pytest.fixture
def built(
    synthetic: tuple[Path, list[SyntheticLoan]],
) -> Iterator[tuple[Config, duckdb.DuckDBPyConnection]]:
    config_path, _ = synthetic
    config = load_config(config_path)
    with connect(config.database) as con:
        ingest(config, con)
        build_person_period(config, con)
        yield config, con


def test_runs_on_a_read_only_connection(
    synthetic: tuple[Path, list[SyntheticLoan]],
) -> None:
    """LGD is a measurement, not a pipeline stage, so it must not need write access.

    The first version created views and failed against a read-only database. Builds and
    closes its own connection first, because DuckDB will not hand out a read-only handle
    while a writer holds the file.
    """
    config_path, _ = synthetic
    config = load_config(config_path)
    with connect(config.database) as con:
        ingest(config, con)
        build_person_period(config, con)
    with connect(config.database, read_only=True) as read_only:
        result = estimate(read_only, config.data.vintages)
    assert result.reached_default >= 0


def test_effective_lgd_is_the_product_of_the_two_factors(
    built: tuple[Config, duckdb.DuckDBPyConnection],
) -> None:
    """The distinction the whole module exists for. Severity per *default event* is
    P(loss | default) × severity per *loss*, because not every default produces a loss."""
    config, con = built
    result = estimate(con, config.data.vintages)
    if result.reached_loss_disposition == 0:
        pytest.skip("synthetic fixture has no loss dispositions")
    assert result.effective_lgd == pytest.approx(
        result.probability_of_loss * result.lgd_at_default_balance
    )
    assert result.cure_rate == pytest.approx(1.0 - result.probability_of_loss)


def test_effective_lgd_never_exceeds_disposition_lgd(
    built: tuple[Config, duckdb.DuckDBPyConnection],
) -> None:
    """Since P(loss | default) <= 1, the effective figure must be the smaller one.
    Getting these the wrong way round would inflate every priced loss."""
    config, con = built
    result = estimate(con, config.data.vintages)
    if result.reached_loss_disposition == 0:
        pytest.skip("synthetic fixture has no loss dispositions")
    assert result.effective_lgd <= result.lgd_at_default_balance + 1e-12


@requires_real_data
def test_config_lgd_matches_what_the_data_says() -> None:
    """Fails if the config value drifts from the measurement — for instance if the train
    vintages change and nobody re-measures."""
    config = load_config()
    with connect(config.database, read_only=True) as con:
        train = estimate(con, config.data.train_vintages, "train")
    assert config.economics.lgd == pytest.approx(train.effective_lgd, abs=5e-4), (
        f"config lgd={config.economics.lgd} but train data says "
        f"{train.effective_lgd:.4f} — re-measure with `cde lgd`"
    )


@requires_real_data
def test_config_lgd_is_the_train_figure_not_the_test_one() -> None:
    """The leakage guard, and the trap the provisional 0.30 fell into.

    0.30 sits close to the realised test-period severity of ~0.24 and 2.3x above the
    train figure of ~0.13. Using it would have made the model look as though it priced
    2006-2008 severity correctly, when nothing available in 2005 could have told it that.
    """
    config = load_config()
    with connect(config.database, read_only=True) as con:
        splits = estimate_splits(config, con)
    train, test = splits["train"], splits["test"]
    assert abs(config.economics.lgd - train.effective_lgd) < abs(
        config.economics.lgd - test.effective_lgd
    ), "config LGD is closer to the realised test severity than to the train estimate"


@requires_real_data
def test_severity_worsened_across_the_out_of_time_boundary() -> None:
    """Falling house prices mean worse recoveries, and fewer borrowers cure. Both factors
    move the same way, so they compound rather than offset."""
    config = load_config()
    with connect(config.database, read_only=True) as con:
        splits = estimate_splits(config, con)
    train, test = splits["train"], splits["test"]
    assert test.probability_of_loss > train.probability_of_loss
    assert test.lgd_at_default_balance > train.lgd_at_default_balance
    # A direction-and-magnitude assertion, not a pinned multiple. This read 2.35x at a
    # 60-month horizon and 1.81x at 120 months, and the difference is not noise: the
    # shorter window truncated train-vintage losses harder than test-vintage ones,
    # because calm-cohort loans defaulted later in life. So the horizon choice was
    # inflating the apparent degradation, and a tight threshold here would encode that
    # artefact as if it were a finding.
    assert test.effective_lgd > 1.5 * train.effective_lgd
    assert "compound" in degradation(splits)


@requires_real_data
def test_per_vintage_severity_is_worse_for_the_crisis_cohorts() -> None:
    config = load_config()
    with connect(config.database, read_only=True) as con:
        table = by_vintage(config, con)
    calm = table[table["vintage"] <= 2004]["effective_lgd"].max()
    crisis = table[table["vintage"] >= 2006]["effective_lgd"].min()
    assert crisis > calm, table.to_string(index=False)


def test_summary_names_the_effective_figure(
    built: tuple[Config, duckdb.DuckDBPyConnection],
) -> None:
    config, con = built
    text = estimate(con, config.data.vintages, "fixture").summary()
    assert "EFFECTIVE LGD" in text
    assert isinstance(estimate(con, config.data.vintages), LgdEstimate)
