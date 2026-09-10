"""The risk-free leg of the discount rate, from the Treasury curve.

## Why the discount rate is not in the loan data

The Freddie Mac files carry the *borrower's* note rate. That is not the discount rate.
The discount rate is the *lender's* required return — cost of funds plus a return on the
capital tied up — and it is a market and policy quantity that appears nowhere in a
loan-level file.

## Why it must not be derived from the note rate either

An earlier design discounted each loan at its own note rate minus a spread. That is
wrong: the note rate already contains a credit-risk premium specific to that borrower,
so discounting a risky borrower's cash flows at a higher rate charges them for their
risk **twice** — once in the discount factor and again in the modelled hazard. The
discount rate must depend only on *when* a loan was written, never on *who* took it out.

## Why a single flat rate is worse than useless

Vintage mean note rates in this sample run from 5.51% (2003) to 8.13% (2000).
Discounting every cohort at one flat 6% makes the 2000 book look wonderful and the 2003
book look negative before a single default has been modelled. That is an artefact of the
assumption, and it would contaminate the cross-vintage comparison the whole out-of-time
ablation rests on.

## What is done instead

    r(vintage) = Treasury yield at origination + funding & required-return spread

Borrower-independent, vintage-specific through the Treasury leg, with only the spread
left as an assumption. And that assumption is *bracketed* rather than free, because

    note rate = Treasury + funding + required return + expected loss + servicing + margin
                           └─────────── the discount rate ──────────┘

so the spread must sit well below the observed note-rate-over-Treasury spread, whose
remainder is expected loss (modelled separately, so it must not be counted here) and
lender margin. :func:`implied_primary_spread` measures that upper bound per vintage.

Resolution is the vintage year, not the origination month. Month-level would be a little
more accurate — the curve moved more than 200bp inside 2008 — but the vintage is the unit
the ablation compares, and a per-loan rate lookup would add a join to every NPV
evaluation. The refinement is available if a specific analysis needs it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cde.config import Config

#: Marks a non-trading day in a FRED CSV. Not an empty field, so it must be coerced
#: explicitly or the column parses as text and every downstream mean is silently wrong.
FRED_MISSING = "."


def to_monthly(annual_rate: float) -> float:
    """Convert an annual rate to the monthly rate that compounds to it.

    ``(1 + r)**(1/12) - 1``, matching :attr:`EconomicsConfig.monthly_discount_rate`, so
    the two routes to a monthly rate cannot disagree.
    """
    return float((1.0 + annual_rate) ** (1.0 / 12.0) - 1.0)


def load_benchmark(path: str | Path, expect_series: str | None = None) -> pd.DataFrame:
    """Read a FRED yield CSV and aggregate daily observations to monthly means.

    Returns columns ``month`` (period start date), ``vintage`` and ``yield_pct``.

    ``expect_series`` checks the CSV's own column header against the series the config
    asked for. Pass it. Without the check this function reads positionally, which makes
    ``economics.benchmark_series`` purely documentary — the config asserts that the
    benchmark must track the horizon, and nothing enforced it. A future horizon change
    would silently keep discounting at the wrong duration, and the only symptom would be
    slightly wrong NPVs.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"no benchmark rates at {source} — run scripts/fetch_benchmark_rates.sh"
        )
    frame = pd.read_csv(source)
    if frame.shape[1] != 2:
        raise ValueError(f"{source}: expected 2 columns from FRED, found {frame.shape[1]}")
    if expect_series is not None:
        header = str(frame.columns[1]).strip()
        if header != expect_series:
            raise ValueError(
                f"{source} holds series {header!r} but economics.benchmark_series asks for "
                f"{expect_series!r}. The benchmark must match the horizon's duration — "
                f"re-run `scripts/fetch_benchmark_rates.sh {expect_series}`."
            )
    frame.columns = ["date", "yield_pct"]
    frame["date"] = pd.to_datetime(frame["date"])
    # errors="coerce" turns FRED's "." holiday marker into NaN. Without it the column
    # stays object-dtype and mean() either raises or silently drops everything.
    frame["yield_pct"] = pd.to_numeric(frame["yield_pct"], errors="coerce")
    frame = frame.dropna(subset=["yield_pct"])
    if frame.empty:
        raise ValueError(f"{source}: no numeric observations after clearing {FRED_MISSING!r}")
    frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        frame.groupby("month", as_index=False)["yield_pct"]
        .mean()
        .assign(vintage=lambda d: d["month"].dt.year)
    )
    return monthly


def benchmark_by_vintage(benchmark: pd.DataFrame, vintages: tuple[int, ...]) -> pd.Series:
    """Mean benchmark yield per vintage year, as a decimal fraction."""
    yearly = benchmark.groupby("vintage")["yield_pct"].mean() / 100.0
    missing = [v for v in vintages if v not in yearly.index]
    if missing:
        raise ValueError(
            f"benchmark series does not cover vintage(s) {missing}; it spans "
            f"{int(yearly.index.min())}-{int(yearly.index.max())}"
        )
    return yearly.reindex(list(vintages)).rename("benchmark_rate")


def discount_rate_by_vintage(config: Config, benchmark: pd.DataFrame | None = None) -> pd.Series:
    """Annual discount rate per vintage, as a decimal fraction.

    Under ``fixed`` mode every vintage gets ``economics.annual_discount_rate``, retained
    so the flat assumption can still be run as a sensitivity case and compared.
    """
    vintages = config.data.vintages
    economics = config.economics
    if economics.discount_rate_mode == "fixed":
        return pd.Series(
            economics.annual_discount_rate, index=list(vintages), name="discount_rate"
        )
    loaded = (
        benchmark
        if benchmark is not None
        else load_benchmark(economics.benchmark_path, economics.benchmark_series)
    )
    base = benchmark_by_vintage(loaded, vintages)
    return (base + economics.discount_spread_bps / 10_000.0).rename("discount_rate")


def bracket_check(config: Config, note_rates: pd.Series, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Verify the discount rate sits inside [Treasury, note rate] for every vintage.

    The economic constraint that makes this whole approach defensible. Since

        note rate = Treasury + funding + required return + expected loss + servicing + margin

    and the discount rate is only the funding-and-required-return part, it must land
    strictly between the two observables. A discount rate *above* the note rate would
    mean the lender demanded more than the borrower pays, before any loss is even
    modelled — arithmetically possible to configure, economically nonsense.

    This is not hypothetical: ``discount_spread_bps: 200`` put three of eight vintages
    outside the bracket, because the measured primary spread is only 166bp for 2006 and
    195-198bp for 2000 and 2007. The configured spread has to stay materially below the
    *minimum* observed primary spread, not the mean.
    """
    spreads = implied_primary_spread(note_rates, benchmark)
    rates = discount_rate_by_vintage(config, benchmark)
    out = spreads.join(rates)
    out["inside_bracket"] = (out["discount_rate"] > out["benchmark_rate"]) & (
        out["discount_rate"] < out["note_rate"]
    )
    out["headroom_bps"] = (out["note_rate"] - out["discount_rate"]) * 10_000.0
    return out


def implied_primary_spread(note_rates: pd.Series, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Note rate minus benchmark yield, per vintage, in basis points.

    The validation quantity, and the upper bound on a defensible discount spread. It is
    the *primary* spread — everything the borrower pays above the risk-free curve — so it
    contains expected loss and lender margin as well as funding cost. A discount spread
    approaching it would be double-counting credit risk already in the hazard model.

    A wrong FRED series or an off-by-one join shows up here immediately: the measured
    values run 165-324bp, tightest for the 2006 cohort at the peak of the credit boom and
    widest for 2008 in the crisis.
    """
    vintages = tuple(int(v) for v in note_rates.index)
    base = benchmark_by_vintage(benchmark, vintages)
    out = pd.DataFrame(
        {
            "note_rate": note_rates.astype(float),
            "benchmark_rate": base,
        }
    )
    out["spread_bps"] = (out["note_rate"] - out["benchmark_rate"]) * 10_000.0
    return out
