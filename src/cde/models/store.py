"""Save and reload fitted hazard models.

Stages 3, 4 and 5 all score with the *same* fitted models, repeatedly. Refitting
each time costs two and a half minutes and, worse, invites drift: a model refitted
under slightly different settings partway through an analysis would silently make
earlier and later numbers incomparable.

What gets saved is not just the two regressions. The :class:`DesignSpec` goes with
them, and it has to: it holds the spline knot positions, the imputation medians, the
standardisation mean and deviation for every linear column, and the retained
category levels. Those were all learned from the *training* rows, and scoring the
test set means reusing them verbatim. Recomputing any of them on the test set would
leak information across the out-of-time boundary, which is the one thing stage 5
must never do.

Alongside them goes a model card recording what was fitted and under what
assumptions, plus a fingerprint of the settings that determine the fit. Loading with
a config whose fingerprint differs raises, rather than quietly scoring with a model
built for a different definition of default.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from cde.config import Config
from cde.models.hazard import CauseModel, HazardModels

#: Bumped when the saved layout changes in a way old files cannot satisfy.
#: 2 added `holdout_share` to the model card -- see :func:`default_path`.
FORMAT_VERSION = 2


class StaleModelError(RuntimeError):
    """Raised when a saved model was fitted under different settings than the config."""


@dataclass(frozen=True)
class ModelCard:
    """What was fitted, on what, under which assumptions."""

    format_version: int
    fitted_at: str
    release: int
    train_vintages: tuple[int, ...]
    horizon_months: int
    default_mode: str
    delinquency_threshold_months: int
    feature_count: int
    fingerprint: str
    #: Fraction of training loans HELD OUT of this fit, so the card says what the model
    #: actually saw. 0.0 is stage 2's model, fitted on everything.
    #:
    #: Deliberately not in :func:`fingerprint`, and the reasoning is worth stating
    #: because the opposite looks more careful. If it were fingerprinted, the config's
    #: single `calibration.holdout_share` would match exactly one of the two saved
    #: models, and stage 2's full-sample fit would start reporting itself as stale under
    #: the very config that produced it. The two fits are kept apart by filename and
    #: told apart by this field, which a consumer can check -- `cde calibrate` does.
    holdout_share: float = 0.0
    causes: dict[str, dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"fitted {self.fitted_at}  release {self.release}  "
            f"vintages {list(self.train_vintages)}",
            f"  default definition : {self.default_mode} "
            f"(threshold {self.delinquency_threshold_months} months)",
            f"  horizon            : {self.horizon_months} months on book",
            f"  features           : {self.feature_count}",
            "  fitted on          : "
            + (
                "all training loans"
                if self.holdout_share == 0.0
                else f"{1.0 - self.holdout_share:.0%} of training loans "
                f"({self.holdout_share:.0%} held out for calibration)"
            ),
        ]
        for cause, facts in self.causes.items():
            lines.append(
                f"  {cause:<19}: {int(facts['events']):,} events, "
                f"base rate {facts['base_rate']:.5%}, "
                f"marginal gap {facts['marginal_gap']:+.5%}"
            )
        return "\n".join(lines)


def fingerprint(config: Config) -> str:
    """A hash over every setting that changes what the fit means.

    Deliberately narrow. The discount rate and LGD are *not* included: they belong
    to stage 4 and can be varied for sensitivity testing without invalidating a
    fitted hazard. The default definition, horizon, training vintages and model
    specification are included, because changing any of them makes a saved model
    answer a different question.
    """
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "release": config.data.release,
        "source": config.data.source,
        "train_vintages": list(config.data.train_vintages),
        "horizon": config.horizon.max_loan_age_months,
        "default_definition": {
            "mode": config.default_definition.mode,
            "threshold": config.default_definition.delinquency_threshold_months,
            "credit_event_codes": list(config.default_definition.zero_balance_default_codes),
            "informative_codes": list(
                config.default_definition.zero_balance_informative_exit_codes
            ),
        },
        "person_period": {
            "exclude_off_clock": config.person_period.exclude_off_clock_loans,
        },
        "model": {
            "loan_age_spline_knots": config.model.loan_age_spline_knots,
            "spline_columns": list(config.model.spline_columns),
            "linear_columns": list(config.model.linear_columns),
            "categorical_columns": list(config.model.categorical_columns),
            "min_category_share": config.model.min_category_share,
            "l2_inverse_strength": config.model.l2_inverse_strength,
            "tolerance": config.model.tolerance,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def default_path(config: Config, *, variant: str | None = None) -> Path:
    """Where models live: beside the database, under the gitignored interim directory.

    Not committed to git. They are derived artefacts, regenerated by `cde hazard`,
    and a binary blob in version control would go stale without anything noticing.

    ``variant`` gives a second fit its own file. Stage 3 needs a model fitted on a
    subset of the training loans, so that a recalibrator has somewhere honest to be
    fitted, and that model must NOT overwrite stage 2's -- the two answer different
    questions and the write-up quotes both.
    """
    stem = "hazard" if variant is None else f"hazard_{variant}"
    return config.data.interim_dir / "models" / f"{stem}.joblib"


def card_for(
    config: Config, models: HazardModels, *, holdout_share: float = 0.0
) -> ModelCard:
    return ModelCard(
        format_version=FORMAT_VERSION,
        fitted_at=datetime.now(UTC).isoformat(timespec="seconds"),
        release=config.data.release,
        train_vintages=config.data.train_vintages,
        horizon_months=config.horizon.max_loan_age_months,
        default_mode=config.default_definition.mode,
        delinquency_threshold_months=config.default_definition.delinquency_threshold_months,
        feature_count=len(models.spec.feature_names),
        fingerprint=fingerprint(config),
        holdout_share=holdout_share,
        causes={
            cause: {
                "events": float(fitted.events),
                "rows": float(fitted.rows),
                "base_rate": fitted.base_rate,
                "marginal_gap": fitted.marginal_gap,
                "iterations": float(fitted.iterations),
            }
            for cause, fitted in models.causes.items()
        },
    )


def save(
    config: Config,
    models: HazardModels,
    path: Path | None = None,
    *,
    holdout_share: float = 0.0,
) -> Path:
    """Persist the fitted models, their design spec and a model card."""
    target = path if path is not None else default_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "card": asdict(card_for(config, models, holdout_share=holdout_share)),
            "spec": models.spec,
            "max_iterations": models.max_iterations,
            "causes": {
                cause: {
                    "model": fitted.model,
                    "events": fitted.events,
                    "rows": fitted.rows,
                    "predicted_events": fitted.predicted_events,
                    "iterations": fitted.iterations,
                }
                for cause, fitted in models.causes.items()
            },
        },
        target,
        compress=3,
    )
    return target


def load(
    config: Config, path: Path | None = None, *, require_match: bool = True
) -> tuple[HazardModels, ModelCard]:
    """Reload fitted models, refusing a file built under different settings.

    ``require_match=False`` loads anyway, for inspecting an old fit deliberately.
    It is not the default, because the failure it permits is silent: scoring with a
    model fitted to a different definition of default produces perfectly plausible
    numbers.
    """
    source = path if path is not None else default_path(config)
    if not source.is_file():
        raise FileNotFoundError(
            f"no saved model at {source} — run `cde hazard` to fit and save one"
        )
    payload = joblib.load(source)
    card = ModelCard(**payload["card"])

    if card.format_version != FORMAT_VERSION:
        raise StaleModelError(
            f"{source} was written in format {card.format_version}, this build reads "
            f"{FORMAT_VERSION}. Refit with `cde hazard`."
        )
    current = fingerprint(config)
    if require_match and card.fingerprint != current:
        raise StaleModelError(
            f"{source} was fitted under different settings (fingerprint "
            f"{card.fingerprint}, config now {current}). Something that changes what "
            f"the model means has moved — the default definition, the horizon, the "
            f"training vintages or the model specification. Refit with `cde hazard`."
        )

    models = HazardModels(
        causes={
            cause: CauseModel(
                cause=cause,
                model=stored["model"],
                spec=payload["spec"],
                events=stored["events"],
                rows=stored["rows"],
                predicted_events=stored["predicted_events"],
                iterations=stored["iterations"],
            )
            for cause, stored in payload["causes"].items()
        },
        spec=payload["spec"],
        max_iterations=payload["max_iterations"],
    )
    return models, card
