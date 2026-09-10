"""NPV of an instalment loan as a discounted cash-flow stream under competing hazards.

Where the project's claims stop being assertions and start being arithmetic. Claim 1
(timing matters as much as incidence) and claim 2 (level matters more than ranking) are
both *properties of this function*: timing enters through the survival factor and the
discount exponent, level enters through ``h_default`` directly and linearly.

## The identity, and the three things PROJECT_PLAN.md section 5.4 leaves out

Section 5.4 gives, with monthly index ``t``, scheduled payment ``M``, outstanding
balance ``B(t)``, hazards ``h_d`` and ``h_p``, survival ``S(t) = PROD_{u<t} (1 - h_d(u)
- h_p(u))``, discount rate ``r`` and severity ``LGD``:

    NPV = -P + SUM_t  S(t) * [ (1 - h_d - h_p)*M + h_p*B(t) + h_d*B(t)*(1 - LGD) ]
                     / (1 + r)^t

That is the right shape and it is incomplete in three ways, each of which changes the
answer by more than the effects stage 3 is trying to measure.

**1. Servicing cost.** Someone has to collect the payments, and it costs about 35bp a
year on the outstanding balance. Carried in config since the start and unused until now.
Charged on every loan alive entering the month, whatever happens during it.

**2. The recovery does not arrive at the default month.** The identity books
``B(t)*(1 - LGD)`` in the month of default. But default here is *first passage to 90+
days past due*, and the cash comes at disposition -- a measured median 18 and mean 28
months later. Worse, the leg cannot simply be lagged as a whole, because it is two
different things added together. Since ``LGD`` is the *effective* severity, already the
product of ``P(loss disposition | 90+ DPD)`` and severity given a disposition, the leg
splits **exactly**:

    B(t) * (1 - LGD_effective)
      == B(t) * (1 - q)                    <- cured: no disposition, so no lag applies
       + B(t) * q * (1 - LGD_disposition)  <- recovered cash, arriving LATE

with ``q = economics.loss_disposition_probability``. Only the second term is lagged. On
this book ``q = 0.37``, so lagging the whole leg would discount the 63% cure portion by
27 months as well -- costing 11.8% of the leg against the 3.2% actually owed, a
correction that overshoots the error it fixes by three and a half times. See
:func:`recovery_multiple`.

**3. A terminal value, without which nothing else matters.** The panel stops at the
120-month administrative censor and these are 30-year mortgages: **83.7% of principal is
still outstanding at the horizon.** Summing only the observed months would make every
NPV enormously negative and every comparison between them a comparison of truncation
artefacts rather than of credit. So the balance surviving to the horizon is booked
there, at par.

Booking it at par is an assumption, and its direction is known: a loan paying above the
discount rate is worth *more* than par, so this **understates** NPV, by more for the
better loans. The alternative -- projecting hazards past 120 months -- would be
inventing data the model has no support for, which is worse. Stated in
PROJECT_PLAN.md section 8.

## Two rate conventions, deliberately not unified

``annual_note_rate / 12`` prices the loan's own cash flows, because that is the
convention the mortgage payment was computed under and using anything else would make a
loan fail to amortise to zero. The *discount* rate compounds geometrically,
``(1+r)^(1/12) - 1``, because it is a required return. Keeping them apart is what makes
:func:`loan_npv` return exactly zero when a zero-hazard loan is discounted at its own
note rate -- the sharpest single sanity check available here, and the one that catches
sign errors, off-by-one months and balance-formula slips in one shot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cde.config import Config

#: Per-loan columns :func:`loan_npv` needs. ``discount_rate`` is annual and comes from
#: :func:`cde.economics.rates.discount_rate_by_vintage` -- keyed on the vintage, never
#: on the loan, or a risky borrower is charged for their risk twice.
TERM_COLUMNS = (
    "loan_identifier",
    "original_upb",
    "original_interest_rate",
    "original_loan_term",
    "discount_rate",
)

#: Per-row columns, exactly the shape :meth:`cde.models.hazard.HazardModels.survival`
#: returns, so its output can be passed straight in.
PATH_COLUMNS = (
    "loan_identifier",
    "loan_age",
    "hazard_default",
    "hazard_prepayment",
    "survival_entering",
)


def to_monthly_discount(annual_rate: float | np.ndarray) -> np.ndarray:
    """Geometric monthly discount rate: ``(1 + r)^(1/12) - 1``.

    Matches :attr:`cde.config.EconomicsConfig.monthly_discount_rate` and
    :func:`cde.economics.rates.to_monthly`, so the three routes cannot disagree.
    """
    return np.asarray(np.power(1.0 + np.asarray(annual_rate, dtype=float), 1.0 / 12.0) - 1.0)


def monthly_payment(
    principal: float | np.ndarray,
    annual_note_rate: float | np.ndarray,
    term_months: float | np.ndarray,
) -> np.ndarray:
    """Level monthly payment fully amortising ``principal`` over ``term_months``.

        M = P * i / (1 - (1 + i)^-n),   i = annual_note_rate / 12

    ``i`` is the note rate divided by twelve, not compounded geometrically: that is the
    convention the contractual payment is set under, and using the geometric rate here
    would leave a residual balance at maturity. The zero-rate case is ``P / n``, handled
    explicitly because the formula divides by ``i``.
    """
    p = np.asarray(principal, dtype=float)
    i = np.asarray(annual_note_rate, dtype=float) / 12.0
    n = np.asarray(term_months, dtype=float)
    if np.any(n <= 0):
        raise ValueError("term_months must be positive")
    with np.errstate(divide="ignore", invalid="ignore"):
        amortising = p * i / (1.0 - np.power(1.0 + i, -n))
    return np.asarray(np.where(np.isclose(i, 0.0), p / n, amortising), dtype=float)


def scheduled_balance(
    principal: float | np.ndarray,
    annual_note_rate: float | np.ndarray,
    term_months: float | np.ndarray,
    age: float | np.ndarray,
) -> np.ndarray:
    """Outstanding balance at the *start* of month ``age`` on the contractual schedule.

        B(t) = P * [ (1+i)^n - (1+i)^t ] / [ (1+i)^n - 1 ]

    so ``B(0) = P`` and ``B(n) = 0`` exactly.

    **The scheduled balance, deliberately, not the observed ``current_actual_upb``.** A
    decision engine values a loan at origination and cannot see that this particular
    borrower will make a curtailment in month 40. Using the realised balance would leak
    the future into a decision made before it, and would also mean the same applicant
    was valued differently depending on how they later behaved. The cost of the choice
    is real -- partial prepayments make actual balances run below scheduled ones -- and
    it is the honest cost of only using information available at the decision point.
    """
    p = np.asarray(principal, dtype=float)
    i = np.asarray(annual_note_rate, dtype=float) / 12.0
    n = np.asarray(term_months, dtype=float)
    # Past maturity the balance is zero, not negative. Clipped rather than left to the
    # formula, which would keep going and hand back a negative balance that reads as a
    # cash inflow on default.
    t = np.clip(np.asarray(age, dtype=float), 0.0, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        growth_n = np.power(1.0 + i, n)
        amortising = p * (growth_n - np.power(1.0 + i, t)) / (growth_n - 1.0)
    linear = p * (1.0 - t / n)
    return np.asarray(np.where(np.isclose(i, 0.0), linear, amortising), dtype=float)


def recovery_multiple(
    monthly_discount_rate: float | np.ndarray,
    *,
    lgd: float,
    loss_disposition_probability: float,
    recovery_lag_months: int,
) -> np.ndarray:
    """Fraction of ``B(t)`` recovered on default, in present value at the default month.

        (1 - q)  +  q * (1 - LGD/q) / (1 + i)^lag

    where ``q`` is the probability of a loss disposition and ``LGD`` the *effective*
    severity per default event, so ``LGD/q`` is severity given a disposition.

    The first term is the cure portion: no disposition happens, so no cash is delayed
    and the identity's original treatment stands. The second is the recovered cash,
    discounted the extra ``recovery_lag_months`` it takes to arrive.

    At ``recovery_lag_months = 0`` this collapses to ``1 - LGD`` exactly, recovering
    PROJECT_PLAN.md section 5.4 unchanged -- which is the invariant that proves the
    split is a refinement of the identity rather than a different model.
    """
    q = float(loss_disposition_probability)
    severity_given_disposition = float(lgd) / q
    lag_discount = np.power(
        1.0 + np.asarray(monthly_discount_rate, dtype=float), -float(recovery_lag_months)
    )
    return np.asarray((1.0 - q) + q * (1.0 - severity_given_disposition) * lag_discount)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{what} is missing column(s) {missing}; needs {list(columns)}")


def _check_survival_recursion(
    loans: np.ndarray, ages: np.ndarray, survival: np.ndarray, leaving: np.ndarray
) -> None:
    """Refuse a path whose survival does not match its own hazards.

    ``survival_entering`` is taken as given rather than recomputed, so that
    :meth:`cde.models.hazard.HazardModels.survival` stays the single implementation of
    the recursion. The price of that is that a caller could pass an inconsistent pair --
    hazards from one model and survival from another, or a frame filtered after survival
    was computed -- and get a perfectly plausible NPV. So the consistency is checked
    instead of trusted:

        S(t+1) == S(t) * (1 - h_d(t) - h_p(t))   within a loan
        S(first) == 1

    Rows are assumed sorted by (loan, ascending age); :func:`loan_npv` sorts before
    calling. Gaps inside a loan's span fail this check, which is the correct outcome --
    a missing month means a month of unaccounted exposure.
    """
    if loans.size == 0:
        return
    first_of_loan = np.empty(loans.size, dtype=bool)
    first_of_loan[0] = True
    first_of_loan[1:] = loans[1:] != loans[:-1]
    if not np.allclose(survival[first_of_loan], 1.0):
        bad = int((~np.isclose(survival[first_of_loan], 1.0)).sum())
        raise ValueError(
            f"{bad} loan(s) do not start with survival_entering == 1. A projected path "
            f"must begin at the decision point, before any month of exposure."
        )
    same_loan = ~first_of_loan[1:]
    expected = survival[:-1] * (1.0 - leaving[:-1])
    drift = np.abs(survival[1:] - expected)[same_loan]
    if drift.size and drift.max() > 1e-9:
        raise ValueError(
            f"survival_entering does not follow from the hazards in the same frame "
            f"(max drift {drift.max():.3e}). The hazards and the survival were computed "
            f"from different frames, or rows were filtered after survival was computed."
        )
    if ages.size > 1:
        backwards = same_loan & (np.diff(ages) <= 0)
        if backwards.any():
            raise ValueError("loan_age is not strictly increasing within a loan")


@dataclass(frozen=True)
class NpvResult:
    """Per-loan NPV, broken into the legs it is a sum of.

    The decomposition is not decoration: an NPV that moves in the right direction for
    the wrong reason is the failure mode here, and the only way to see that is to watch
    which leg moved.
    """

    frame: pd.DataFrame
    horizon: int
    lgd: float
    recovery_lag_months: int

    @property
    def total(self) -> float:
        return float(self.frame["npv"].sum())

    @property
    def per_loan(self) -> float:
        return float(self.frame["npv"].mean())

    def summary(self) -> str:
        f = self.frame
        principal = float(f["principal"].sum())
        lines = [
            f"NPV over {len(f):,} loans, {self.horizon} month horizon"
            f"   (LGD {self.lgd:.4f}, recovery lag {self.recovery_lag_months}m)",
            f"  {'principal advanced':<24} {-principal:>16,.0f}",
        ]
        for column, label in (
            ("interest_leg", "scheduled payments"),
            ("prepayment_leg", "prepayment at par"),
            ("default_leg", "recovery on default"),
            ("terminal_leg", "balance at the horizon"),
            ("servicing_leg", "servicing cost"),
        ):
            value = float(f[column].sum())
            lines.append(f"  {label:<24} {value:>16,.0f}   {value / principal:>7.2%} of principal")
        lines += [
            f"  {'':<24} {'-' * 16}",
            f"  {'NPV':<24} {self.total:>16,.0f}   {self.total / principal:>7.2%} of principal",
            f"  {'NPV per loan':<24} {self.per_loan:>16,.0f}",
        ]
        return "\n".join(lines)


def loan_npv(
    config: Config,
    terms: pd.DataFrame,
    paths: pd.DataFrame,
    *,
    lgd: float | None = None,
    loss_disposition_probability: float | None = None,
    recovery_lag_months: int | None = None,
    servicing_cost_annual_bps: float | None = None,
) -> NpvResult:
    """Present value of each loan's projected cash flows, one row out per loan.

    ``terms`` carries :data:`TERM_COLUMNS`, one row per loan. ``paths`` carries
    :data:`PATH_COLUMNS`, one row per loan-month, and is exactly what
    :meth:`cde.models.hazard.HazardModels.survival` returns -- so a recalibrated variant
    is produced by transforming the two hazard columns and passing the frame back in.

    The keyword arguments override the corresponding config values, for the sensitivity
    sweeps stage 4 needs. They are arguments rather than a mutated config because a
    sweep must not be able to leave the config changed for whatever runs next.

    **Timing convention, stated because everything depends on it.** Month ``t`` runs
    from ``loan_age == t`` to ``t + 1``. ``B(t)`` is the balance entering it,
    ``survival_entering`` is the probability of being alive to enter it, and every cash
    flow during it -- payment, prepayment, recovery, servicing -- is discounted as
    arriving at its *end*, so the exponent is ``t + 1``. The balance surviving the final
    projected month is booked at par at that same point.
    """
    _require_columns(terms, TERM_COLUMNS, "terms")
    _require_columns(paths, PATH_COLUMNS, "paths")
    economics = config.economics
    severity = economics.lgd if lgd is None else float(lgd)
    q = (
        economics.loss_disposition_probability
        if loss_disposition_probability is None
        else float(loss_disposition_probability)
    )
    lag = (
        economics.recovery_lag_months
        if recovery_lag_months is None
        else int(recovery_lag_months)
    )
    servicing_bps = (
        economics.servicing_cost_annual_bps
        if servicing_cost_annual_bps is None
        else float(servicing_cost_annual_bps)
    )
    if severity > q:
        raise ValueError(
            f"lgd {severity} exceeds loss_disposition_probability {q}, which implies "
            f"severity given disposition above 1 — a defaulted loan recovering more "
            f"than it owed"
        )
    monthly_servicing = servicing_bps / 10_000.0 / 12.0

    if terms["loan_identifier"].duplicated().any():
        raise ValueError("terms holds more than one row for some loan")
    # Sorted here rather than required of the caller: the output is one row per loan, so
    # row order is not observable in the result and there is no reason to make it the
    # caller's problem. The survival recursion check below does depend on the order.
    joined = (
        paths[list(PATH_COLUMNS)]
        .merge(terms[list(TERM_COLUMNS)], on="loan_identifier", how="left", validate="m:1")
        .sort_values(["loan_identifier", "loan_age"], kind="stable")
        .reset_index(drop=True)
    )
    orphans = joined["original_upb"].isna()
    if orphans.any():
        raise ValueError(
            f"{int(orphans.sum())} path row(s) have no matching loan in terms; the "
            f"projection and the loan characteristics came from different populations"
        )

    loan_ids = joined["loan_identifier"].to_numpy()
    age = joined["loan_age"].to_numpy(dtype=float)
    hazard_default = joined["hazard_default"].to_numpy(dtype=float)
    hazard_prepayment = joined["hazard_prepayment"].to_numpy(dtype=float)
    survival = joined["survival_entering"].to_numpy(dtype=float)
    leaving = hazard_default + hazard_prepayment
    _check_survival_recursion(loan_ids, age, survival, leaving)

    principal = joined["original_upb"].to_numpy(dtype=float)
    note_rate = joined["original_interest_rate"].to_numpy(dtype=float) / 100.0
    term = joined["original_loan_term"].to_numpy(dtype=float)
    monthly_discount = to_monthly_discount(joined["discount_rate"].to_numpy(dtype=float))

    balance = scheduled_balance(principal, note_rate, term, age)
    payment = monthly_payment(principal, note_rate, term)
    recovery = recovery_multiple(
        monthly_discount,
        lgd=severity,
        loss_disposition_probability=q,
        recovery_lag_months=lag,
    )
    # Cash arrives at the end of month t, so the exponent is t + 1.
    discount = np.power(1.0 + monthly_discount, -(age + 1.0))
    exposed = survival * discount

    legs = pd.DataFrame(
        {
            "loan_identifier": loan_ids,
            "interest_leg": exposed * (1.0 - leaving) * payment,
            "prepayment_leg": exposed * hazard_prepayment * balance,
            "default_leg": exposed * hazard_default * balance * recovery,
            "servicing_leg": -exposed * monthly_servicing * balance,
        }
    )
    per_loan = legs.groupby("loan_identifier", sort=True).sum()

    # Terminal value. At the end of each loan's last projected month it is still alive
    # with probability S(T) * (1 - h_d(T) - h_p(T)) and owes B(T+1); that balance is
    # booked at par. Without this the NPV is dominated by the 84% of principal the
    # horizon truncates rather than by anything to do with credit.
    last = joined.groupby("loan_identifier", sort=True).tail(1).index.to_numpy()
    terminal = pd.Series(
        survival[last]
        * (1.0 - leaving[last])
        * scheduled_balance(principal[last], note_rate[last], term[last], age[last] + 1.0)
        * discount[last],
        index=pd.Index(loan_ids[last], name="loan_identifier"),
        name="terminal_leg",
    )
    out = per_loan.join(terminal, how="left")
    out["principal"] = (
        terms.set_index("loan_identifier")["original_upb"].reindex(out.index).astype(float)
    )
    out["npv"] = (
        out["interest_leg"]
        + out["prepayment_leg"]
        + out["default_leg"]
        + out["terminal_leg"]
        + out["servicing_leg"]
        - out["principal"]
    )
    out["npv_per_dollar"] = out["npv"] / out["principal"]
    return NpvResult(
        frame=out.reset_index(),
        horizon=config.horizon.max_loan_age_months,
        lgd=severity,
        recovery_lag_months=lag,
    )
