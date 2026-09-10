"""Typed access to ``configs/default.yaml``.

Every economic and modelling assumption lives in YAML, never in code, so that each one
can be found, changed and sensitivity-tested. This module is the only thing that reads
it, and it validates on load.

Validation is not ceremony here. The failure mode this guards against is silent: a
``mode: delinqency`` typo would fall through to whatever the code's ``else`` branch does,
produce a plausible-looking hazard curve fitted to the wrong event, and never raise. That
is exactly the "metric theatre" the project rules forbid, so it fails at load instead.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import yaml

from cde.data.schema import (
    RELEASE,
    ZERO_BALANCE_CODES,
    ZERO_BALANCE_PREPAID,
)

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")

DEFAULT_MODES = ("delinquency", "termination")
SOURCES = ("sample", "standard")
DISCOUNT_RATE_MODES = ("treasury_plus_spread", "fixed")
BINNING_STRATEGIES = ("quantile", "uniform")
CALIBRATION_METHODS = ("platt", "isotonic")

if TYPE_CHECKING:  # stub-only module; `fields()` needs a dataclass-typed argument
    from _typeshed import DataclassInstance

_T = TypeVar("_T", bound="DataclassInstance")


class ConfigError(ValueError):
    """Raised when the config is internally inconsistent or contradicts the schema."""


@dataclass(frozen=True)
class DataConfig:
    """Which release, which sample, which origination vintages, and where they live."""

    release: int
    source: str
    train_vintages: tuple[int, ...]
    test_vintages: tuple[int, ...]
    raw_dir: Path
    interim_dir: Path

    @property
    def vintages(self) -> tuple[int, ...]:
        """Every vintage to ingest, train and test together, ascending."""
        return tuple(sorted(set(self.train_vintages) | set(self.test_vintages)))

    def __post_init__(self) -> None:
        if self.release != RELEASE:
            raise ConfigError(
                f"config declares release {self.release} but cde.data.schema is generated "
                f"for release {RELEASE}. The column layout differs between releases, and the "
                f"files have no header row, so a mismatch mislabels every field silently."
            )
        if self.source not in SOURCES:
            raise ConfigError(f"data.source must be one of {SOURCES}, got {self.source!r}")
        if not self.train_vintages:
            raise ConfigError("data.train_vintages is empty")
        if not self.test_vintages:
            raise ConfigError("data.test_vintages is empty")
        overlap = set(self.train_vintages) & set(self.test_vintages)
        if overlap:
            raise ConfigError(
                f"vintages {sorted(overlap)} appear in both train and test. Out-of-time "
                f"validation requires disjoint origination cohorts — see PROJECT_PLAN.md §5.5."
            )
        if max(self.train_vintages) >= min(self.test_vintages):
            raise ConfigError(
                f"train vintages must all precede test vintages: train ends "
                f"{max(self.train_vintages)}, test starts {min(self.test_vintages)}. "
                f"Anything else is a random split wearing an out-of-time costume."
            )


@dataclass(frozen=True)
class HorizonConfig:
    """The administrative censor. See PROJECT_PLAN.md §4.3."""

    max_loan_age_months: int

    def __post_init__(self) -> None:
        if self.max_loan_age_months < 1:
            raise ConfigError(
                f"horizon.max_loan_age_months must be >= 1, got {self.max_loan_age_months}"
            )


@dataclass(frozen=True)
class DefaultDefinition:
    """What counts as default, and in which month. See PROJECT_PLAN.md §10 question 5."""

    mode: str
    zero_balance_default_codes: tuple[str, ...]
    zero_balance_informative_exit_codes: tuple[str, ...]
    delinquency_threshold_months: int

    @property
    def delinquency_threshold_code(self) -> str:
        """The ``CURRENT LOAN DELINQUENCY STATUS`` code at the threshold.

        The column is a zero-padded count of missed payments, so three months is ``"03"``
        (90-119 days past due). Returned as text because the column is text: it also
        takes ``"RA"`` and ``"XX"``, which is why comparisons against it need care.
        """
        return f"{self.delinquency_threshold_months:02d}"

    def __post_init__(self) -> None:
        if self.mode not in DEFAULT_MODES:
            raise ConfigError(
                f"default_definition.mode must be one of {DEFAULT_MODES}, got {self.mode!r}"
            )
        if not 1 <= self.delinquency_threshold_months <= 99:
            raise ConfigError(
                "default_definition.delinquency_threshold_months must be in 1..99 "
                f"(the column is capped at 99), got {self.delinquency_threshold_months}"
            )
        defaults = set(self.zero_balance_default_codes)
        informative = set(self.zero_balance_informative_exit_codes)
        unknown = (defaults | informative) - set(ZERO_BALANCE_CODES)
        if unknown:
            raise ConfigError(
                f"unknown zero balance codes {sorted(unknown)}; release {RELEASE} defines "
                f"{sorted(ZERO_BALANCE_CODES)}"
            )
        if not defaults:
            raise ConfigError("default_definition.zero_balance_default_codes is empty")
        if defaults & informative:
            raise ConfigError(
                f"codes {sorted(defaults & informative)} are listed as both a credit event "
                f"and an informative exit; each code must be classified exactly once or the "
                f"same loan is counted as defaulted and censored"
            )
        if ZERO_BALANCE_PREPAID in defaults:
            raise ConfigError(
                f"code {ZERO_BALANCE_PREPAID} is voluntary payoff, not default. Treating "
                f"prepayment as a credit event would invert the sign of the largest exit "
                f"in the book."
            )


@dataclass(frozen=True)
class PersonPeriodConfig:
    """How the person-period table treats loans whose time axis is untrustworthy."""

    exclude_off_clock_loans: bool


@dataclass(frozen=True)
class ModelConfig:
    """Stage 2 -- the hazard model's specification. See configs/default.yaml."""

    loan_age_spline_knots: int
    spline_columns: tuple[str, ...]
    linear_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    min_category_share: float
    l2_inverse_strength: float
    tolerance: float
    max_iterations: int

    def __post_init__(self) -> None:
        if self.loan_age_spline_knots < 3:
            raise ConfigError(
                "model.loan_age_spline_knots must be >= 3 for a cubic spline basis, got "
                f"{self.loan_age_spline_knots}"
            )
        if not 0.0 <= self.min_category_share < 1.0:
            raise ConfigError(
                f"model.min_category_share must be in [0, 1), got {self.min_category_share}"
            )
        if not 0.0 < self.tolerance <= 1e-3:
            raise ConfigError(
                "model.tolerance must be in (0, 1e-3]; sklearn's 1e-4 default leaves the "
                "fitted probabilities off the observed event count on rare-event data, "
                f"got {self.tolerance}"
            )
        if self.l2_inverse_strength <= 0.0:
            raise ConfigError(
                "model.l2_inverse_strength is sklearn's C and must be positive, got "
                f"{self.l2_inverse_strength}"
            )
        overlap = set(self.spline_columns) & set(self.linear_columns)
        if overlap:
            raise ConfigError(f"columns are both splined and linear: {sorted(overlap)}")
        clash = (set(self.spline_columns) | set(self.linear_columns)) & set(
            self.categorical_columns
        )
        if clash:
            raise ConfigError(f"columns are both numeric and categorical: {sorted(clash)}")


@dataclass(frozen=True)
class EconomicsConfig:
    """Discount rate, loss given default, servicing cost. All provisional — §10 Q1, Q2."""

    discount_rate_mode: str
    benchmark_series: str
    benchmark_path: Path
    discount_spread_bps: float
    annual_discount_rate: float
    lgd: float
    loss_disposition_probability: float
    recovery_lag_months: int
    servicing_cost_annual_bps: float

    @property
    def monthly_discount_rate(self) -> float:
        """Monthly rate consistent with the annual one, compounded.

        ``(1 + r)**(1/12) - 1``, so twelve months of compounding reproduces the annual
        rate exactly. The alternative convention, ``r / 12``, is what mortgage notes use
        for *interest* accrual, and at 6% the two differ by 1.3 basis points a month —
        small, but it accumulates over a 120-month horizon and it is the kind of
        inconsistency worth being deliberate about rather than accidental.

        Flagged as a finance convention the author should verify independently before
        quoting a headline NPV: the geometric form is right for a required *return*, but
        practitioners discounting a loan's contractual cash flows often use r/12 to stay
        consistent with how the payment itself was computed.
        """
        return float((1.0 + self.annual_discount_rate) ** (1.0 / 12.0) - 1.0)

    @property
    def monthly_servicing_cost_rate(self) -> float:
        """Servicing cost as a monthly fraction of outstanding balance."""
        return self.servicing_cost_annual_bps / 10_000.0 / 12.0

    @property
    def lgd_given_disposition(self) -> float:
        """Severity conditional on actually reaching a loss disposition.

        Derived, never configured, so it cannot drift from the two numbers it is a
        ratio of. :attr:`lgd` is the *effective* severity per default event, which is
        already the product of the probability of a loss disposition and the severity
        given one -- so dividing recovers the second factor.

        The NPV needs both factors separately rather than only their product, because
        the recovery leg splits into a cure part that never reaches a disposition and
        a recovered part whose cash arrives ``recovery_lag_months`` later. See
        :func:`cde.economics.npv.loan_npv`.
        """
        return self.lgd / self.loss_disposition_probability

    @property
    def cure_share(self) -> float:
        """Share of default events that never produce a loss disposition."""
        return 1.0 - self.loss_disposition_probability

    def __post_init__(self) -> None:
        if self.discount_rate_mode not in DISCOUNT_RATE_MODES:
            raise ConfigError(
                f"economics.discount_rate_mode must be one of {DISCOUNT_RATE_MODES}, got "
                f"{self.discount_rate_mode!r}"
            )
        if not -500.0 <= self.discount_spread_bps <= 2000.0:
            raise ConfigError(
                "economics.discount_spread_bps is a spread over Treasuries in basis "
                f"points and must be in [-500, 2000], got {self.discount_spread_bps}"
            )
        if self.annual_discount_rate <= -1.0:
            raise ConfigError(
                f"economics.annual_discount_rate must exceed -1, got {self.annual_discount_rate}"
            )
        if not 0.0 <= self.lgd <= 1.0:
            raise ConfigError(
                f"economics.lgd is a fraction of exposure and must be in [0, 1], got {self.lgd}"
            )
        if self.servicing_cost_annual_bps < 0.0:
            raise ConfigError(
                f"economics.servicing_cost_annual_bps must be >= 0, got "
                f"{self.servicing_cost_annual_bps}"
            )
        if not 0.0 < self.loss_disposition_probability <= 1.0:
            raise ConfigError(
                "economics.loss_disposition_probability is P(loss disposition | default) "
                f"and must be in (0, 1], got {self.loss_disposition_probability}"
            )
        # The two configured severity numbers imply a third, and an incoherent pair
        # would produce a recovery above par -- a defaulted loan handing back more
        # than it owed. Caught here rather than as a negative loss in the NPV.
        if self.lgd > self.loss_disposition_probability:
            raise ConfigError(
                f"economics.lgd ({self.lgd}) exceeds economics."
                f"loss_disposition_probability ({self.loss_disposition_probability}), which "
                f"implies severity given disposition of "
                f"{self.lgd / self.loss_disposition_probability:.4f} -- above 1. `lgd` is the "
                f"EFFECTIVE severity per default event, already multiplied by the probability "
                f"of a loss disposition, so it cannot exceed that probability."
            )
        if self.recovery_lag_months < 0:
            raise ConfigError(
                f"economics.recovery_lag_months must be >= 0, got {self.recovery_lag_months}"
            )


@dataclass(frozen=True)
class CalibrationConfig:
    """Stage 3 -- how miscalibration is measured and corrected.

    Nothing here belongs in :func:`cde.models.store.fingerprint`. These settings change
    how a fitted hazard is *assessed*, never what it is, so a change to any of them must
    not invalidate a saved model -- pinned by
    ``tests/test_store.py::test_stage_four_assumptions_do_not_invalidate_a_fit``.
    """

    bins: int
    binning: str
    horizons: tuple[int, ...]
    holdout_share: float
    holdout_seed: int
    methods: tuple[str, ...]
    isotonic_min_tail_rows: int

    def __post_init__(self) -> None:
        if self.bins < 2:
            raise ConfigError(f"calibration.bins must be >= 2, got {self.bins}")
        if self.binning not in BINNING_STRATEGIES:
            raise ConfigError(
                f"calibration.binning must be one of {BINNING_STRATEGIES}, got "
                f"{self.binning!r}"
            )
        if not self.horizons:
            raise ConfigError("calibration.horizons is empty")
        if any(h < 1 for h in self.horizons):
            raise ConfigError(f"calibration.horizons must all be >= 1, got {list(self.horizons)}")
        if list(self.horizons) != sorted(set(self.horizons)):
            raise ConfigError(
                f"calibration.horizons must be ascending and unique, got {list(self.horizons)}"
            )
        if not 0.0 < self.holdout_share < 1.0:
            raise ConfigError(
                "calibration.holdout_share is a fraction of training loans and must be in "
                f"(0, 1), got {self.holdout_share}"
            )
        if not self.methods:
            raise ConfigError("calibration.methods is empty")
        unknown = set(self.methods) - set(CALIBRATION_METHODS)
        if unknown:
            raise ConfigError(
                f"unknown calibration.methods {sorted(unknown)}; known methods are "
                f"{sorted(CALIBRATION_METHODS)}"
            )
        if self.isotonic_min_tail_rows < 0:
            raise ConfigError(
                "calibration.isotonic_min_tail_rows must be >= 0 (0 disables the tail "
                f"cap), got {self.isotonic_min_tail_rows}"
            )


@dataclass(frozen=True)
class DecisionConfig:
    """Stage 4 -- how a score becomes an approve/decline decision.

    Like :class:`CalibrationConfig`, nothing here belongs in
    :func:`cde.models.store.fingerprint`: these settings change how a fitted hazard is
    *used*, never what it is.
    """

    binary_horizon_months: int
    sweep_points: int
    choose_thresholds_on_holdout: bool

    def __post_init__(self) -> None:
        if self.binary_horizon_months < 1:
            raise ConfigError(
                "decision.binary_horizon_months must be >= 1, got "
                f"{self.binary_horizon_months}"
            )
        if self.sweep_points < 10:
            raise ConfigError(
                f"decision.sweep_points must be >= 10 to trace a curve, got "
                f"{self.sweep_points}"
            )


@dataclass(frozen=True)
class Config:
    """The whole configuration, validated."""

    data: DataConfig
    horizon: HorizonConfig
    default_definition: DefaultDefinition
    person_period: PersonPeriodConfig
    model: ModelConfig
    economics: EconomicsConfig
    calibration: CalibrationConfig
    decision: DecisionConfig
    path: Path

    @property
    def database(self) -> Path:
        """Where the DuckDB database lives."""
        return self.data.interim_dir / "cde.duckdb"

    def __post_init__(self) -> None:
        """Checks that span two sections, so neither can make them alone."""
        horizon = self.horizon.max_loan_age_months
        beyond = [h for h in self.calibration.horizons if h > horizon]
        if beyond:
            raise ConfigError(
                f"calibration.horizons {beyond} exceed horizon.max_loan_age_months "
                f"({horizon}). No loan is observed past the administrative censor, so "
                f"cumulative incidence at those horizons would be measured against an "
                f"empty risk set and read as zero rather than as missing."
            )
        if self.decision.binary_horizon_months > horizon:
            raise ConfigError(
                f"decision.binary_horizon_months ({self.decision.binary_horizon_months}) "
                f"exceeds horizon.max_loan_age_months ({horizon}). The binary target "
                f"would be defined over months no loan is observed for, so every loan "
                f"would read as a non-default."
            )


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        section = raw[name]
    except KeyError:
        raise ConfigError(f"config is missing the {name!r} section") from None
    if not isinstance(section, dict):
        kind = type(section).__name__
        raise ConfigError(f"config section {name!r} must be a mapping, got {kind}")
    return section


def _build(cls: type[_T], section: dict[str, Any], name: str, **extra: Any) -> _T:
    """Instantiate a config dataclass, reporting unknown and missing keys by name.

    A silently-ignored key is the whole problem with hand-rolled config loading: you
    rename an assumption in YAML, the code keeps using its default, and the number that
    comes out is wrong in a way no test catches.
    """
    declared = {f.name for f in fields(cls)} - set(extra)
    unknown = set(section) - declared
    if unknown:
        raise ConfigError(
            f"unknown keys in {name!r}: {sorted(unknown)}; expected {sorted(declared)}"
        )
    missing = declared - set(section)
    if missing:
        raise ConfigError(f"{name!r} is missing keys: {sorted(missing)}")
    return cls(**section, **extra)


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the configuration.

    Raises :class:`ConfigError` on anything inconsistent, rather than proceeding with a
    plausible default.
    """
    resolved = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not resolved.is_file():
        raise ConfigError(f"no config at {resolved}")
    raw = yaml.safe_load(resolved.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{resolved} does not contain a YAML mapping")

    unknown = set(raw) - {
        "data",
        "horizon",
        "default_definition",
        "person_period",
        "model",
        "economics",
        "calibration",
        "decision",
    }
    if unknown:
        raise ConfigError(f"unknown top-level config sections: {sorted(unknown)}")

    data_section = dict(_section(raw, "data"))
    for key in ("train_vintages", "test_vintages"):
        if key in data_section:
            data_section[key] = tuple(data_section[key])
    for key in ("raw_dir", "interim_dir"):
        if key in data_section:
            data_section[key] = Path(data_section[key])

    model_section = dict(_section(raw, "model"))
    for key in ("spline_columns", "linear_columns", "categorical_columns"):
        if key in model_section:
            model_section[key] = tuple(model_section[key])

    economics_section = dict(_section(raw, "economics"))
    if "benchmark_path" in economics_section:
        economics_section["benchmark_path"] = Path(economics_section["benchmark_path"])

    calibration_section = dict(_section(raw, "calibration"))
    for key in ("horizons", "methods"):
        if key in calibration_section:
            calibration_section[key] = tuple(calibration_section[key])

    default_section = dict(_section(raw, "default_definition"))
    for key in ("zero_balance_default_codes", "zero_balance_informative_exit_codes"):
        if key in default_section:
            default_section[key] = tuple(str(code) for code in default_section[key])

    return Config(
        data=_build(DataConfig, data_section, "data"),
        horizon=_build(HorizonConfig, _section(raw, "horizon"), "horizon"),
        default_definition=_build(DefaultDefinition, default_section, "default_definition"),
        person_period=_build(PersonPeriodConfig, _section(raw, "person_period"), "person_period"),
        model=_build(ModelConfig, model_section, "model"),
        economics=_build(EconomicsConfig, economics_section, "economics"),
        calibration=_build(CalibrationConfig, calibration_section, "calibration"),
        decision=_build(DecisionConfig, _section(raw, "decision"), "decision"),
        path=resolved,
    )
