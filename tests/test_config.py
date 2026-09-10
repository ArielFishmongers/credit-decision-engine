"""The config is the project's only source of assumptions, so a silently-wrong config
is a silently-wrong result. These tests are about failing loudly."""

from __future__ import annotations

from pathlib import Path

import pytest

from cde.config import ConfigError, load_config
from tests.conftest import write_config


def test_loads_and_derives(config_path: Path) -> None:
    config = load_config(config_path)
    assert config.data.vintages == (2000, 2001)
    assert config.default_definition.mode == "delinquency"
    assert config.database == config.data.interim_dir / "cde.duckdb"


def test_delinquency_threshold_becomes_a_zero_padded_code(config_path: Path) -> None:
    """Three missed payments is status "03" -- 90-119 days past due. The column is text
    and zero-padded, so an unpadded "3" would match nothing."""
    assert load_config(config_path).default_definition.delinquency_threshold_code == "03"


def test_monthly_discount_rate_compounds_to_the_annual_one(config_path: Path) -> None:
    economics = load_config(config_path).economics
    compounded = (1.0 + economics.monthly_discount_rate) ** 12 - 1.0
    assert compounded == pytest.approx(economics.annual_discount_rate)
    # And is deliberately not the r/12 convention used for interest accrual.
    assert economics.monthly_discount_rate != pytest.approx(economics.annual_discount_rate / 12)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"release": "46"}, "release 46"),
        ({"source": "'lending_club'"}, "data.source"),
        ({"mode": "'delinqency'"}, "default_definition.mode"),
        ({"test_vintages": "[1999]"}, "must all precede"),
        ({"train_vintages": "[2000, 2001]"}, "both train and test"),
        ({"max_loan_age_months": "0"}, "must be >= 1"),
        ({"lgd": "1.4"}, "must be in [0, 1]"),
        ({"servicing_cost_annual_bps": "-5"}, "must be >= 0"),
        ({"delinquency_threshold_months": "0"}, "must be in 1..99"),
        ({"zero_balance_default_codes": '["03", "07"]'}, "unknown zero balance codes"),
        ({"zero_balance_default_codes": '["01", "03"]'}, "voluntary payoff, not default"),
        ({"zero_balance_informative_exit_codes": '["03", "16"]'}, "exactly once"),
    ],
)
def test_rejects_inconsistent_config(
    project: Path, overrides: dict[str, str], expected: str
) -> None:
    path = write_config(project, project / "raw", project / "interim", **overrides)
    with pytest.raises(ConfigError) as caught:
        load_config(path)
    assert expected in str(caught.value)


def test_rejects_unknown_and_missing_keys(project: Path) -> None:
    """A key that YAML has but the dataclass does not is the dangerous case: rename an
    assumption and the code silently keeps using the old one."""
    path = project / "typo.yaml"
    # `vintages` is the trap: it was a real key once, so a stale config would carry
    # it and the loader must refuse rather than quietly ignore it.
    path.write_text("""
data: {release: 47, source: sample, train_vintages: [2000], test_vintages: [2001],
       raw_dir: raw, interim_dir: interim, vintages: [2000]}
horizon: {max_loan_age_months: 60}
default_definition: {mode: delinquency, zero_balance_default_codes: ["03"],
                     zero_balance_informative_exit_codes: ["16"],
                     delinquency_threshold_months: 3}
person_period: {exclude_off_clock_loans: true}
model: {loan_age_spline_knots: 4, spline_columns: [], linear_columns: [],
        categorical_columns: [], min_category_share: 0.005,
        l2_inverse_strength: 100.0, tolerance: 1.0e-8, max_iterations: 2000}
economics: {discount_rate_mode: fixed, benchmark_series: DGS5,
            benchmark_path: data/benchmark/treasury_dgs5.csv,
            discount_spread_bps: 200, annual_discount_rate: 0.06, lgd: 0.3,
            loss_disposition_probability: 0.6, recovery_lag_months: 27,
            servicing_cost_annual_bps: 35}
calibration: {bins: 20, binning: quantile, horizons: [12, 24, 36, 60],
              holdout_share: 0.2, holdout_seed: 20260908,
              methods: [platt, isotonic], isotonic_min_tail_rows: 1000}
decision: {binary_horizon_months: 36, sweep_points: 200,
           choose_thresholds_on_holdout: true}
""")
    with pytest.raises(ConfigError) as caught:
        load_config(path)
    assert "unknown keys in 'data'" in str(caught.value)
    assert "vintages" in str(caught.value)


def test_missing_file_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no config at"):
        load_config(tmp_path / "absent.yaml")
