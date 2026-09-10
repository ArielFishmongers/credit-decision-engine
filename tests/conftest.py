"""Shared fixtures.

Everything here is built on Freddie Mac's public 1,000-row *format example*
(``release-47-sample-files.zip``), not on the 50,000-loan vintage sample. The format
example is enough to prove the pipeline parses, types, joins and reshapes correctly.
It is nowhere near enough for a statistic: it carries 12 loans with performance history.
Any test that asserts on a rate or a curve shape belongs against real vintage data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from cde.data.schema import ORIGINATION_NAMES, PERFORMANCE_NAMES

REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "freddie-mac"
SAMPLES = REFERENCE / "Sample Files"

#: Two vintages, because DataConfig requires train vintages strictly before test ones.
FIXTURE_VINTAGES = (2000, 2001)

requires_reference = pytest.mark.skipif(
    not SAMPLES.is_dir(),
    reason="reference docs absent; run scripts/fetch_reference_docs.sh",
)

CONFIG_TEMPLATE = """
data:
  release: 47
  source: sample
  train_vintages: [2000]
  test_vintages: [2001]
  raw_dir: {raw}
  interim_dir: {interim}
horizon:
  max_loan_age_months: 60
default_definition:
  mode: delinquency
  zero_balance_default_codes: ["02", "03", "09"]
  zero_balance_informative_exit_codes: ["15", "16", "96"]
  delinquency_threshold_months: 3
person_period:
  exclude_off_clock_loans: true
model:
  loan_age_spline_knots: 4
  spline_columns: [classic_fico]
  linear_columns: [original_interest_rate]
  categorical_columns: [occupancy_status, channel]
  min_category_share: 0.005
  l2_inverse_strength: 100.0
  tolerance: 1.0e-8
  max_iterations: 2000
economics:
  discount_rate_mode: fixed
  benchmark_series: DGS5
  benchmark_path: data/benchmark/treasury_dgs5.csv
  discount_spread_bps: 200
  annual_discount_rate: 0.06
  lgd: 0.30
  loss_disposition_probability: 0.60
  recovery_lag_months: 27
  servicing_cost_annual_bps: 35
calibration:
  bins: 20
  binning: quantile
  horizons: [12, 24, 36, 60]
  holdout_share: 0.2
  holdout_seed: 20260908
  methods: [platt, isotonic]
  isotonic_min_tail_rows: 1000
decision:
  binary_horizon_months: 36
  sweep_points: 200
  choose_thresholds_on_holdout: true
"""


def write_config(tmp_path: Path, raw: Path, interim: Path, **overrides: str) -> Path:
    """Write a config YAML into ``tmp_path``, optionally patching whole lines.

    Overrides are keyed by the YAML key and replace that key's line, which is enough for
    tests that need to break one field and check the error.
    """
    text = CONFIG_TEMPLATE.format(raw=raw.as_posix(), interim=interim.as_posix())
    for key, value in overrides.items():
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{key}:"):
                indent = line[: len(line) - len(line.lstrip())]
                lines[i] = f"{indent}{key}: {value}"
                break
        else:  # pragma: no cover - a typo in a test, not a code path
            raise KeyError(f"{key} not in the config template")
        text = "\n".join(lines)
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A raw directory holding the format example named as vintage sample files."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for vintage in FIXTURE_VINTAGES:
        # Loan identifiers are PYYQnXXXXXXX, so characters 1-4 encode the origination
        # quarter. Rewriting them per vintage keeps the primary keys unique, which the
        # real data gets for free -- a loan lives in exactly one vintage file.
        stamp = f"{str(vintage)[-2:]}Q1"
        for source, name in (
            ("origination_sample_file.txt", f"sample_orig_{vintage}.txt"),
            ("performance_sample_file.txt", f"sample_perf_{vintage}.txt"),
        ):
            text = (SAMPLES / source).read_text()
            (raw / name).write_text(re.sub(r"\bF\w\wQ\w(\d{7})\b", rf"F{stamp}\1", text))
    (tmp_path / "interim").mkdir()
    return tmp_path


@pytest.fixture
def config_path(project: Path) -> Path:
    return write_config(project, project / "raw", project / "interim")


# --------------------------------------------------------------------------------------
# Synthetic fixture.
#
# The format example cannot test the person-period construction at all: its origination
# and performance files share no loan identifiers (1,000 loans in one, a disjoint 12 in
# the other), so any join between them is empty. It is a *format* illustration.
#
# So the person-period logic is tested against loans built here, one per case, with the
# expected exit age and reason known by construction rather than eyeballed from output.
# --------------------------------------------------------------------------------------

ORIGINAL_TERM = 360
ORIGINAL_UPB = 200_000.0
ORIGINAL_RATE = 6.0


def _origination_row(
    loan: str,
    vintage: int,
    term: int = ORIGINAL_TERM,
    *,
    fico: int = 720,
    rate: float = ORIGINAL_RATE,
) -> str:
    """A valid 31-field origination record. Only the fields stage 1 reads are set."""
    values = dict.fromkeys(ORIGINATION_NAMES, "")
    values.update(
        {
            "loan_identifier": loan,
            "classic_fico": str(fico),
            "first_payment_date": f"{vintage}03",
            "maturity_date": f"{vintage + term // 12}02",
            "first_time_homebuyer_indicator": "N",
            "number_of_units": "1",
            # Varied across loans so the categorical path is exercised rather than
            # dropped for having a single level.
            "occupancy_status": "P" if int(loan[-1]) % 2 == 0 else "S",
            "original_combined_loan_to_value": "80",
            "original_debt_to_income_ratio": "35",
            "original_upb": f"{ORIGINAL_UPB:.2f}",
            "original_loan_to_value": "80",
            "original_interest_rate": f"{rate:.3f}",
            "channel": "R",
            "amortization_type": "FRM",
            "property_state": "CA",
            "property_type": "SF",
            "postal_code": "900",
            "loan_purpose": "P",
            "original_loan_term": str(term),
            "number_of_borrowers": "2",
            "mortgage_insurance_percentage": "0",
            "prepayment_penalty_indicator": "N",
            "interest_only_indicator": "N",
            "vantagescore_4_0": "9999",
            "property_valuation_method": "1",
            "harp_indicator": "N",
            "super_conforming_flag": "N",
            "seller_name": "TEST SELLER",
        }
    )
    return "|".join(values[name] for name in ORIGINATION_NAMES)


def _performance_row(
    loan: str,
    period: str,
    loan_age: int,
    status: str,
    *,
    remaining: int,
    zero_balance: str = "",
    actual_loss: float | None = None,
) -> str:
    """A valid 35-field performance record."""
    values = dict.fromkeys(PERFORMANCE_NAMES, "")
    values.update(
        {
            "loan_identifier": loan,
            "period": period,
            "current_actual_upb": f"{ORIGINAL_UPB:.2f}",
            "current_loan_delinquency_status": status,
            "loan_age": str(loan_age),
            "remaining_months_to_legal_maturity": str(remaining),
            "current_interest_rate": f"{ORIGINAL_RATE:.3f}",
            "current_interest_bearing_upb": f"{ORIGINAL_UPB:.2f}",
            "current_non_interest_bearing_upb": "0.00",
            "estimated_loan_to_value": "75",
            "servicer_name": "OTHER",
            "mortgage_insurance_cancellation_indicator": "7",
        }
    )
    if zero_balance:
        values["zero_balance_code"] = zero_balance
        values["zero_balance_effective_date"] = period
    if actual_loss is not None:
        # A realised loss, so the severity logic in cde.economics.lgd is exercised
        # rather than skipped. Positive means a loss, per the user guide's convention.
        values["actual_loss"] = f"{actual_loss:.2f}"
        values["zero_balance_removal_upb"] = f"{ORIGINAL_UPB:.2f}"
    return "|".join(values[name] for name in PERFORMANCE_NAMES)


def _months(start_year: int, count: int, first_age: int = 0) -> list[tuple[str, int]]:
    """(period, loan_age) pairs, ``count`` consecutive months from January ``start_year``."""
    out = []
    for i in range(count):
        year, month = divmod(i, 12)
        out.append((f"{start_year + year}{month + 1:02d}", first_age + i))
    return out


@dataclass(frozen=True)
class SyntheticLoan:
    """A loan whose person-period outcome is known by construction."""

    loan: str
    expected_exit_age: int
    expected_reason: str
    note: str
    off_clock: bool = False


def _build_synthetic(vintage: int) -> tuple[list[str], list[str], list[SyntheticLoan]]:
    origination: list[str] = []
    performance: list[str] = []
    expected: list[SyntheticLoan] = []
    suffix = str(vintage)[-2:]

    def emit(
        index: int,
        months: int,
        expected_exit_age: int,
        expected_reason: str,
        note: str,
        *,
        status_at: dict[int, str] | None = None,
        zero_balance_at: tuple[int, str] | None = None,
        loss_at_zero_balance: float | None = None,
        first_age: int = 0,
        term: int = ORIGINAL_TERM,
        entry_remaining: int | None = None,
        off_clock: bool = False,
    ) -> None:
        loan = f"F{suffix}Q1{index:07d}"
        # A spread of scores, so classic_fico is not constant and its spline basis
        # can actually be fitted.
        # FICO and rate both varied, so the spline and linear paths are exercised
        # rather than dropped for having a single value.
        origination.append(
            _origination_row(
                loan, vintage, term=term, fico=600 + 20 * index, rate=5.0 + 0.25 * index
            )
        )
        # On the origination clock, loan_age + remaining == original term in every month.
        # A left-truncated loan entering at age 6 therefore has 354 months remaining, not
        # 360 -- get that wrong and the off-clock check correctly rejects the loan.
        base_remaining = term - first_age if entry_remaining is None else entry_remaining
        for period, age in _months(vintage, months, first_age=first_age):
            status = (status_at or {}).get(age, "00")
            zero_balance = ""
            loss = None
            if zero_balance_at is not None and age == zero_balance_at[0]:
                zero_balance = zero_balance_at[1]
                loss = loss_at_zero_balance
            performance.append(
                _performance_row(
                    loan,
                    period,
                    age,
                    status,
                    remaining=base_remaining - (age - first_age),
                    zero_balance=zero_balance,
                    actual_loss=loss,
                )
            )
        expected.append(SyntheticLoan(loan, expected_exit_age, expected_reason, note, off_clock))

    # Survives the whole horizon: 71 observed months, censored administratively at 60.
    emit(1, 71, 60, "administrative_censor", "never delinquent, observed past the horizon")

    # Walks 01 -> 02 -> 03. First passage to 90+ DPD is age 12, not 10.
    emit(
        2, 31, 12, "default", "first passage to 90+ DPD",
        status_at={10: "01", 11: "02", 12: "03", 13: "04"},
    )

    # Voluntary payoff. Censoring in v1, and the largest exit in a mortgage book.
    emit(3, 21, 20, "prepaid", "voluntary payoff", zero_balance_at=(20, "01"))

    # Charge-off with the delinquency column never showing 90+ DPD: still a default.
    emit(
        4, 26, 25, "default", "credit event without a prior 90+ DPD record",
        zero_balance_at=(25, "03"), loss_at_zero_balance=0.40 * ORIGINAL_UPB,
    )

    # Performance simply stops before the horizon: censored by the data cutoff.
    emit(5, 41, 40, "data_censor", "observation ends before the horizon")

    # Reperforming loan securitisation: leaves the panel for a credit-correlated reason.
    emit(6, 16, 15, "informative_exit", "informative exit (code 16)", zero_balance_at=(15, "16"))

    # "RA" is REO acquisition, a default state that try_cast cannot see.
    emit(7, 36, 30, "default", "RA (REO acquisition) counts as default", status_at={30: "RA"})

    # 90+ DPD, but after the horizon: censored at 60, no event.
    emit(8, 71, 60, "administrative_censor", "90+ DPD beyond the horizon", status_at={65: "03"})

    # Enters at age 6: left truncation from Freddie Mac acquiring a seasoned loan.
    emit(9, 45, 50, "data_censor", "left truncated, enters at age 6", first_age=6)

    # Clock reset before acquisition: age 0 but 14 months delinquent on a 60-month
    # remaining term. Both off-clock signals fire.
    emit(
        10, 21, 0, "default", "off clock: modified before acquisition",
        status_at=dict.fromkeys(range(21), "14"), entry_remaining=60, off_clock=True,
    )

    # --- clock reset DURING observation ------------------------------------------
    #
    # The case the rest of this fixture missed, and which only showed up on real
    # data: a loan modified mid-life, so loan_age restarts and the same ages occur
    # twice. Sequencing the exit by loan_age value rather than by calendar order
    # duplicates rows, fires the event indicator twice, and can manufacture an
    # in-horizon default out of a much later delinquency.
    #
    # Built by hand rather than through emit(), which assumes a monotonic clock.

    # L11: current for 30 months, then modified -- clock back to 0 on a 480-month
    # term -- and 90+ DPD at post-modification age 20. That second age 20 is inside
    # the horizon but belongs to a different clock, so it must NOT be an event.
    # Correct answer: censored at age 29, the last month before the axis broke.
    loan = f"F{suffix}Q10000011"
    origination.append(_origination_row(loan, vintage, fico=700, rate=6.5))
    rows = [(period, age, "00", ORIGINAL_TERM - age) for period, age in _months(vintage, 30)]
    for i, (period, _) in enumerate(_months(vintage + 3, 25)):
        status = "03" if i >= 20 else "00"
        rows.append((period, i, status, 480 - i))
    for period, age, status, remaining in rows:
        performance.append(_performance_row(loan, period, age, status, remaining=remaining))
    expected.append(
        SyntheticLoan(loan, 29, "clock_reset_censor", "clock reset during observation")
    )

    # L12: 90+ DPD at age 12, and only later modified at age 40. The event fires
    # first, so the reset is irrelevant and must not change the answer. This is the
    # common case on real data -- modification follows delinquency.
    loan = f"F{suffix}Q10000012"
    origination.append(_origination_row(loan, vintage, fico=660, rate=7.25))
    rows = [
        (period, age, "03" if age >= 12 else "00", ORIGINAL_TERM - age)
        for period, age in _months(vintage, 40)
    ]
    for i, (period, _) in enumerate(_months(vintage + 4, 12)):
        rows.append((period, i, "00", 480 - i))
    for period, age, status, remaining in rows:
        performance.append(_performance_row(loan, period, age, status, remaining=remaining))
    expected.append(
        SyntheticLoan(loan, 12, "default", "event precedes the reset, so the reset is moot")
    )

    # L13: a missing reporting month at age 15, so loan_age jumps 14 -> 16. The risk
    # set cannot be contiguous across it, so the loan censors at 14.
    loan = f"F{suffix}Q10000013"
    origination.append(_origination_row(loan, vintage, fico=780, rate=5.5))
    for period, age in _months(vintage, 30):
        if age == 15:
            continue
        performance.append(
            _performance_row(loan, period, age, "00", remaining=ORIGINAL_TERM - age)
        )
    expected.append(SyntheticLoan(loan, 14, "month_gap_censor", "missing reporting month"))

    return origination, performance, expected


@pytest.fixture
def synthetic(tmp_path: Path) -> tuple[Path, list[SyntheticLoan]]:
    """A raw directory of hand-built loans, plus their expected outcomes."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (tmp_path / "interim").mkdir()
    expected: list[SyntheticLoan] = []
    for vintage in FIXTURE_VINTAGES:
        origination, performance, cases = _build_synthetic(vintage)
        (raw / f"sample_orig_{vintage}.txt").write_text("\n".join(origination) + "\n")
        (raw / f"sample_perf_{vintage}.txt").write_text("\n".join(performance) + "\n")
        if vintage == FIXTURE_VINTAGES[0]:
            expected = cases
    return write_config(tmp_path, raw, tmp_path / "interim"), expected
