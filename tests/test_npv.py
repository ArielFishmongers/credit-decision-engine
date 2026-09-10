"""The NPV identity, tested as invariants rather than against remembered numbers.

There is no reference implementation to check against and no published answer for this
book, so "it returned 12,431" proves nothing. What can be established is a set of
properties the function must have, several of which are the project's own claims in
executable form:

* it returns **exactly** zero for a riskless loan discounted at its own note rate
* it falls when severity, default hazard, discount rate or recovery delay rise
* it falls when the *same* amount of default is moved earlier -- claim 1's precondition
* the recovery split reduces to PROJECT_PLAN.md section 5.4 when the lag is zero

Every one of these would be satisfied by a wrong implementation that happened to be
monotone, which is why the zero-NPV identity carries most of the weight: it pins the
timing convention, the balance formula and the sign of every leg simultaneously.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cde.config import Config, load_config
from cde.economics.npv import (
    PATH_COLUMNS,
    TERM_COLUMNS,
    loan_npv,
    monthly_payment,
    recovery_multiple,
    scheduled_balance,
    to_monthly_discount,
)

PRINCIPAL = 200_000.0
NOTE_RATE_PCT = 6.0
TERM = 360
HORIZON = 120


@pytest.fixture
def config() -> Config:
    return load_config()


def note_rate_as_effective_annual(note_rate_pct: float = NOTE_RATE_PCT) -> float:
    """The annual discount rate whose *monthly* value equals the note rate over twelve.

    The two conventions in :mod:`cde.economics.npv` are deliberately different -- the
    payment uses ``r/12``, the discount compounds geometrically -- so discounting "at
    the note rate" means passing the effective annual rate that reproduces ``r/12``
    monthly. Getting this wrong is why the zero-NPV test is worth having: it fails by
    about 0.3% of principal if the conventions are confused.
    """
    monthly = note_rate_pct / 100.0 / 12.0
    return float((1.0 + monthly) ** 12 - 1.0)


def build(
    *,
    hazard_default: np.ndarray | float = 0.0,
    hazard_prepayment: np.ndarray | float = 0.0,
    horizon: int = HORIZON,
    discount_rate: float | None = None,
    note_rate_pct: float = NOTE_RATE_PCT,
    term: int = TERM,
    principal: float = PRINCIPAL,
    loan: str = "L1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One loan's terms and a projected path, with survival built from the hazards."""
    ages = np.arange(horizon, dtype=float)
    h_d = np.broadcast_to(np.asarray(hazard_default, dtype=float), ages.shape).copy()
    h_p = np.broadcast_to(np.asarray(hazard_prepayment, dtype=float), ages.shape).copy()
    # survival_entering: 1 for the first month, then the running product of survival.
    alive = np.concatenate([[1.0], np.cumprod(1.0 - h_d - h_p)[:-1]])
    paths = pd.DataFrame(
        {
            "loan_identifier": loan,
            "loan_age": ages.astype(int),
            "hazard_default": h_d,
            "hazard_prepayment": h_p,
            "survival_entering": alive,
        }
    )
    terms = pd.DataFrame(
        {
            "loan_identifier": [loan],
            "original_upb": [principal],
            "original_interest_rate": [note_rate_pct],
            "original_loan_term": [term],
            "discount_rate": [
                note_rate_as_effective_annual(note_rate_pct)
                if discount_rate is None
                else discount_rate
            ],
        }
    )
    return terms[list(TERM_COLUMNS)], paths[list(PATH_COLUMNS)]


def npv_of(config: Config, **kwargs: object) -> float:
    build_keys = {
        "hazard_default", "hazard_prepayment", "horizon", "discount_rate",
        "note_rate_pct", "term", "principal", "loan",
    }
    terms, paths = build(**{k: v for k, v in kwargs.items() if k in build_keys})  # type: ignore[arg-type]
    overrides = {k: v for k, v in kwargs.items() if k not in build_keys}
    return loan_npv(config, terms, paths, **overrides).per_loan  # type: ignore[arg-type]


# --- the amortisation identity -------------------------------------------------------


def test_scheduled_balance_starts_at_par_and_amortises_to_zero() -> None:
    assert scheduled_balance(PRINCIPAL, 0.06, TERM, 0) == pytest.approx(PRINCIPAL)
    assert scheduled_balance(PRINCIPAL, 0.06, TERM, TERM) == pytest.approx(0.0)
    # Never negative past maturity: a negative balance would read as an inflow on
    # default and quietly add value to the worst loans in the book.
    assert scheduled_balance(PRINCIPAL, 0.06, TERM, TERM + 60) == pytest.approx(0.0)


def test_zero_rate_balance_is_linear() -> None:
    """The i = 0 branch exists because the closed form divides by i."""
    assert scheduled_balance(PRINCIPAL, 0.0, 100, 25) == pytest.approx(0.75 * PRINCIPAL)
    assert monthly_payment(PRINCIPAL, 0.0, 100) == pytest.approx(PRINCIPAL / 100)


def test_payment_amortises_the_balance_exactly() -> None:
    """Balance recursion and closed form must agree, or B(t) and M describe two loans."""
    i = NOTE_RATE_PCT / 100.0 / 12.0
    payment = float(monthly_payment(PRINCIPAL, NOTE_RATE_PCT / 100.0, TERM))
    balance = PRINCIPAL
    for _ in range(TERM):
        balance = balance * (1.0 + i) - payment
    assert balance == pytest.approx(0.0, abs=1e-6)


# --- the sharpest check available ----------------------------------------------------


def test_riskless_loan_at_its_own_note_rate_is_worth_exactly_par(config: Config) -> None:
    """NPV == 0. The single most informative test in this file.

    With no default, no prepayment and no servicing cost, a loan discounted at its own
    note rate must be worth exactly what was lent. It pins the timing convention (cash
    at the END of month t, so the exponent is t+1), the balance formula, the terminal
    value and the sign of every leg at once -- a one-month shift alone breaks it by
    about 0.5% of principal.
    """
    npv = npv_of(config, servicing_cost_annual_bps=0.0)
    assert npv == pytest.approx(0.0, abs=1e-6)


def test_the_identity_holds_at_any_horizon_and_any_rate(config: Config) -> None:
    """Because B(H) is *defined* as the present value of what is left to pay."""
    for horizon in (1, 12, 60, 120, 360):
        for rate in (0.0, 3.5, 6.0, 11.25):
            npv = npv_of(
                config, horizon=horizon, note_rate_pct=rate, servicing_cost_annual_bps=0.0
            )
            assert npv == pytest.approx(0.0, abs=1e-6), f"horizon {horizon}, rate {rate}"


def test_the_terminal_value_carries_most_of_the_book(config: Config) -> None:
    """Sanity on magnitude, and on the scale of the error that omitting it would be.

    84% of principal is still outstanding at month 120 on a 30-year loan, which is
    **46% of principal in present value** once discounted back ten years. Drop the
    terminal value and every NPV lands near -46% of principal, so stage 3 would be
    comparing truncation artefacts rather than credit.

    The two numbers are different and it is worth being careful which is which: the
    outstanding balance is 84% of par, its discounted value is 46%.
    """
    terms, paths = build()
    result = loan_npv(config, terms, paths, servicing_cost_annual_bps=0.0)
    row = result.frame.iloc[0]
    assert row["terminal_leg"] / row["principal"] == pytest.approx(0.46, abs=0.02)
    # The loan is worth exactly par, so what is left without the terminal leg is the
    # size of the omission.
    without_terminal = row["npv"] - row["terminal_leg"]
    assert without_terminal / row["principal"] == pytest.approx(-0.46, abs=0.02)


# --- monotonicity, one economic assumption at a time ---------------------------------


def test_npv_falls_as_severity_rises(config: Config) -> None:
    hazard = 0.001
    values = [
        npv_of(config, hazard_default=hazard, lgd=lgd, loss_disposition_probability=0.9)
        for lgd in (0.0, 0.1, 0.3, 0.6, 0.9)
    ]
    assert values == sorted(values, reverse=True), values


def test_npv_falls_as_the_default_hazard_rises(config: Config) -> None:
    values = [npv_of(config, hazard_default=h) for h in (0.0, 0.0005, 0.001, 0.005, 0.01)]
    assert values == sorted(values, reverse=True), values


def test_npv_falls_as_the_discount_rate_rises(config: Config) -> None:
    """The loan's net cash flows are positive, so discounting them harder is worth less."""
    values = [
        npv_of(config, hazard_default=0.001, discount_rate=r)
        for r in (0.02, 0.04, 0.06, 0.08, 0.12)
    ]
    assert values == sorted(values, reverse=True), values


def test_npv_falls_as_the_recovery_lag_lengthens(config: Config) -> None:
    """Correction 2. Cash that arrives later is worth less; the identity ignored this."""
    values = [
        npv_of(config, hazard_default=0.002, recovery_lag_months=lag)
        for lag in (0, 6, 18, 27, 60)
    ]
    assert values == sorted(values, reverse=True), values


def test_servicing_cost_reduces_value(config: Config) -> None:
    """Correction 1. Carried unused in config until stage 3 needed it."""
    free = npv_of(config, servicing_cost_annual_bps=0.0)
    costly = npv_of(config, servicing_cost_annual_bps=35.0)
    assert costly < free
    # 35bp a year on a balance near par over ten years, discounted: a few percent of
    # principal. Bracketed loosely -- the point is the order of magnitude, not the value.
    assert -0.05 < (costly - free) / PRINCIPAL < -0.01


# --- claim 1's precondition ----------------------------------------------------------


def test_the_same_default_probability_earlier_is_worth_less(config: Config) -> None:
    """**Claim 1 in executable form**, or rather its precondition.

    Two loans with *identical cumulative default probability* over the window, differing
    only in when that default happens: one defaults in month 0, the other in the final
    month. A binary within-36-months target cannot tell them apart. The NPV must, and
    the early one must be worth less -- it loses the whole payment stream and recovers a
    balance that has barely amortised.

    This does not prove claim 1, which only the stage 5 ablation settles. It proves the
    NPV is *capable* of expressing it, without which the ablation would measure nothing.
    """
    probability = 0.02
    early = np.zeros(HORIZON)
    early[0] = probability
    late = np.zeros(HORIZON)
    late[-1] = probability

    early_terms, early_paths = build(hazard_default=early)
    late_terms, late_paths = build(hazard_default=late)
    # Same total default probability, by construction: nothing else leaves, so survival
    # is 1 up to the single month in which the hazard is non-zero.
    assert early_paths["hazard_default"].sum() == pytest.approx(
        late_paths["hazard_default"].sum()
    )

    early_npv = loan_npv(config, early_terms, early_paths).per_loan
    late_npv = loan_npv(config, late_terms, late_paths).per_loan
    assert early_npv < late_npv
    # Worth quantifying, because "different" would also be satisfied by a trivial gap.
    assert (late_npv - early_npv) / PRINCIPAL > 0.001


# --- the recovery split --------------------------------------------------------------


def test_recovery_multiple_reduces_to_the_plan_identity_at_zero_lag() -> None:
    """With no lag the split must collapse to ``1 - LGD``, exactly.

    This is what makes the split a refinement of PROJECT_PLAN.md section 5.4 rather than
    a different model, and it holds for any pair of factors.
    """
    for lgd, q in ((0.1330, 0.3697), (0.5, 0.5), (0.0, 0.9), (0.2407, 0.4765)):
        multiple = recovery_multiple(
            0.005, lgd=lgd, loss_disposition_probability=q, recovery_lag_months=0
        )
        assert float(multiple) == pytest.approx(1.0 - lgd)


def test_lagging_the_whole_recovery_leg_would_overshoot(config: Config) -> None:
    """The bug this design avoids, measured rather than described.

    The plan for this stage said to discount the recovery leg by the recovery lag. Done
    to the leg as a whole that also delays the ~63% of default events that CURE and
    never reach a disposition at all, so it charges a delay on cash that was never
    delayed. The wrong version is not obviously wrong from its output -- it is simply a
    slightly lower NPV -- so it is pinned here.
    """
    economics = config.economics
    monthly = float(to_monthly_discount(0.056))
    lag_discount = (1.0 + monthly) ** -economics.recovery_lag_months

    correct = float(
        recovery_multiple(
            monthly,
            lgd=economics.lgd,
            loss_disposition_probability=economics.loss_disposition_probability,
            recovery_lag_months=economics.recovery_lag_months,
        )
    )
    naive_whole_leg = (1.0 - economics.lgd) * lag_discount
    unlagged = 1.0 - economics.lgd

    assert naive_whole_leg < correct < unlagged
    cost_correct = (unlagged - correct) / unlagged
    cost_naive = (unlagged - naive_whole_leg) / unlagged
    # The naive version overstates the correction by roughly three and a half times.
    assert cost_naive / cost_correct > 3.0
    assert cost_correct < 0.05


# --- the guards ----------------------------------------------------------------------


def test_survival_inconsistent_with_the_hazards_is_refused(config: Config) -> None:
    """A frame filtered after survival was computed produces plausible numbers."""
    terms, paths = build(hazard_default=0.001)
    broken = paths.drop(index=paths.index[5]).reset_index(drop=True)
    with pytest.raises(ValueError, match="does not follow from the hazards"):
        loan_npv(config, terms, broken)


def test_a_path_not_starting_at_the_decision_point_is_refused(config: Config) -> None:
    terms, paths = build(hazard_default=0.001)
    shifted = paths.iloc[3:].reset_index(drop=True)
    with pytest.raises(ValueError, match="survival_entering == 1"):
        loan_npv(config, terms, shifted)


def test_paths_without_matching_terms_are_refused(config: Config) -> None:
    terms, paths = build()
    with pytest.raises(ValueError, match="no matching loan in terms"):
        loan_npv(config, terms.assign(loan_identifier="OTHER"), paths)


def test_severity_above_the_disposition_probability_is_refused(config: Config) -> None:
    """Implies severity given disposition above 1: a default recovering above par."""
    terms, paths = build(hazard_default=0.001)
    with pytest.raises(ValueError, match="exceeds loss_disposition_probability"):
        loan_npv(config, terms, paths, lgd=0.5, loss_disposition_probability=0.4)


def test_legs_sum_to_the_reported_npv(config: Config) -> None:
    """Guards against a leg being computed, reported and then left out of the total."""
    terms, paths = build(hazard_default=0.001, hazard_prepayment=0.01)
    row = loan_npv(config, terms, paths).frame.iloc[0]
    rebuilt = (
        row["interest_leg"]
        + row["prepayment_leg"]
        + row["default_leg"]
        + row["terminal_leg"]
        + row["servicing_leg"]
        - row["principal"]
    )
    assert rebuilt == pytest.approx(row["npv"])
