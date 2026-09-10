"""The benchmark yield curve behind the discount rate.

The tests that matter here are not about parsing. They are about two failures that would
produce plausible NPV numbers built on a wrong rate: FRED's non-numeric holiday marker
silently poisoning the column, and the discount rate picking up borrower risk.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from cde.config import ConfigError, load_config
from cde.economics.rates import (
    benchmark_by_vintage,
    discount_rate_by_vintage,
    implied_primary_spread,
    load_benchmark,
    to_monthly,
)


def _real_benchmark() -> Path:
    """The benchmark the *config* asks for, not a hard-coded filename.

    This was pinned to `treasury_dgs5.csv` and not updated when the horizon moved to 120
    months and the series to DGS10. The guard then watched a file the pipeline no longer
    used: deleting the stale CSV would have silently skipped five tests, and deleting the
    live one would have failed them with FileNotFoundError instead of skipping.
    """
    return Path(load_config().economics.benchmark_path)


requires_benchmark = pytest.mark.skipif(
    not Path("configs/default.yaml").is_file() or not _real_benchmark().is_file(),
    reason="benchmark rates absent; run scripts/fetch_benchmark_rates.sh",
)

#: Series name used by the synthetic FRED fixtures below. Deliberately NOT the production
#: series: these tests exercise parsing, and pinning them to production would couple them
#: to a config value they have no business depending on.
FIXTURE_SERIES = "DGS5"


def _write_fred(path: Path, rows: str) -> Path:
    path.write_text(f"observation_date,{FIXTURE_SERIES}\n" + rows)
    return path


def test_holiday_markers_are_coerced_not_silently_kept(tmp_path: Path) -> None:
    """FRED marks non-trading days with "." rather than an empty field.

    Left alone, the column parses as text, and a mean over it either raises or drops
    everything — either way the discount rate is wrong and nothing says so.
    """
    path = _write_fred(
        tmp_path / "t.csv",
        "2000-01-03,6.50\n2000-01-04,.\n2000-01-05,6.60\n",
    )
    monthly = load_benchmark(path)
    assert len(monthly) == 1
    assert monthly["yield_pct"].iloc[0] == pytest.approx(6.55)  # the "." row excluded


def test_all_missing_series_raises_rather_than_returning_nothing(tmp_path: Path) -> None:
    path = _write_fred(tmp_path / "t.csv", "2000-01-03,.\n2000-01-04,.\n")
    with pytest.raises(ValueError, match="no numeric observations"):
        load_benchmark(path)


def test_missing_file_says_how_to_get_one(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fetch_benchmark_rates"):
        load_benchmark(tmp_path / "absent.csv")


def test_uncovered_vintage_is_refused(tmp_path: Path) -> None:
    """Silently reindexing to NaN would propagate into every discounted cash flow."""
    path = _write_fred(tmp_path / "t.csv", "2000-01-03,6.50\n2000-02-01,6.40\n")
    with pytest.raises(ValueError, match="does not cover"):
        benchmark_by_vintage(load_benchmark(path), (2000, 2007))


def test_monthly_rate_compounds_back_to_annual() -> None:
    for annual in (0.0, 0.02, 0.06, 0.12):
        assert (1.0 + to_monthly(annual)) ** 12 - 1.0 == pytest.approx(annual)


def test_monthly_conversion_agrees_with_the_config_property() -> None:
    """Two routes to a monthly rate exist; they must not disagree."""
    config = load_config()
    assert to_monthly(config.economics.annual_discount_rate) == pytest.approx(
        config.economics.monthly_discount_rate
    )


def test_fixed_mode_gives_every_vintage_the_same_rate() -> None:
    config = load_config()
    fixed = replace(config, economics=replace(config.economics, discount_rate_mode="fixed"))
    rates = discount_rate_by_vintage(fixed)
    assert rates.nunique() == 1
    assert rates.iloc[0] == pytest.approx(fixed.economics.annual_discount_rate)


def test_an_unknown_discount_mode_is_refused() -> None:
    config = load_config()
    with pytest.raises(ConfigError, match="discount_rate_mode"):
        replace(config, economics=replace(config.economics, discount_rate_mode="note_relative"))


@requires_benchmark
def test_treasury_mode_varies_by_vintage_and_tracks_the_curve() -> None:
    """The whole point of the mode: 2000 was a high-rate year and 2003 a low one, so a
    discount rate that does not move between them is not doing its job."""
    config = load_config()
    rates = discount_rate_by_vintage(config)
    assert set(rates.index) == set(config.data.vintages)
    assert rates.nunique() > 1
    assert rates[2000] > rates[2003]


@requires_benchmark
def test_implied_primary_spread_lands_in_a_plausible_range() -> None:
    """The guard against a wrong FRED series or an off-by-one join.

    The primary mortgage spread over comparable Treasuries is a documented quantity.
    Measured here at 165-324bp, tightest for 2006 at the peak of the credit boom and
    widest for 2008 in the crisis. A different series, or a join shifted by a year, would
    not reproduce that.
    """
    import duckdb

    config = load_config()
    if not config.database.is_file():
        pytest.skip("no database; run `cde ingest`")
    con = duckdb.connect(str(config.database), read_only=True)
    note = (
        con.execute(
            "SELECT origination_vintage AS vintage, avg(original_interest_rate)/100.0 AS note "
            "FROM origination GROUP BY 1 ORDER BY 1"
        )
        .df()
        .set_index("vintage")["note"]
    )
    con.close()

    spreads = implied_primary_spread(note, load_benchmark(config.economics.benchmark_path))
    assert spreads["spread_bps"].between(100.0, 400.0).all(), spreads["spread_bps"].to_dict()
    # Spreads compress at the top of the boom and blow out in the crisis.
    assert spreads.loc[2006, "spread_bps"] < spreads.loc[2008, "spread_bps"]


@requires_benchmark
def test_the_discount_rate_never_depends_on_the_borrower() -> None:
    """The guard against the rejected design creeping back.

    Discounting a loan at its own note rate minus a spread charges a risky borrower for
    their risk twice: once in the discount factor, once in the modelled hazard. So the
    rate must be a function of the vintage alone -- one value per cohort, no spread of
    values within it.
    """
    config = load_config()
    rates = discount_rate_by_vintage(config)
    assert isinstance(rates, pd.Series)
    assert rates.index.name is None or rates.index.name == "vintage"
    # One rate per vintage: nothing loan-level can enter.
    assert len(rates) == len(config.data.vintages)
    assert not rates.isna().any()


@requires_benchmark
def test_the_configured_spread_keeps_every_vintage_inside_the_bracket() -> None:
    """The economic constraint, enforced.

    A discount rate above the note rate means the lender required more than the borrower
    pays, before a single default is modelled. `discount_spread_bps: 200` did exactly
    that for 2000, 2006 and 2007, because the measured primary spread bottoms out at
    166bp for the 2006 cohort — so the spread must sit below the MINIMUM observed
    primary spread, not the mean.
    """
    import duckdb

    from cde.economics.rates import bracket_check

    config = load_config()
    if not config.database.is_file():
        pytest.skip("no database; run `cde ingest`")
    con = duckdb.connect(str(config.database), read_only=True)
    note = (
        con.execute(
            "SELECT origination_vintage AS vintage, avg(original_interest_rate)/100.0 AS note "
            "FROM origination GROUP BY 1 ORDER BY 1"
        )
        .df()
        .set_index("vintage")["note"]
    )
    con.close()

    checked = bracket_check(config, note, load_benchmark(config.economics.benchmark_path))
    outside = checked[~checked["inside_bracket"]]
    assert outside.empty, (
        f"discount_spread_bps={config.economics.discount_spread_bps:.0f} puts these "
        f"vintages outside [Treasury, note rate]: {outside.index.tolist()}; "
        f"headroom {checked['headroom_bps'].round(0).to_dict()}"
    )
