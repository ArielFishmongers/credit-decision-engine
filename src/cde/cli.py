"""Command-line entry point. One subcommand per pipeline stage.

Each stage is separately runnable and separately inspectable, because each has to be
independently evaluable — a stage whose output can only be seen by running the four
stages after it cannot be debugged or defended.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from cde.calibration.calibrate import Calibrator
from cde.config import Config, ConfigError, load_config
from cde.models.store import StaleModelError


def _add_schema(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    subparsers.add_parser("schema", help="print the Release 47 column layout")


def _add_config(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    subparsers.add_parser("config", help="print the resolved configuration and derived values")


def _add_ingest(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "ingest", help="stage 1: unpack the raw files and load them into DuckDB"
    )
    parser.add_argument(
        "--check-casts",
        action="store_true",
        help="report any value that was present in the file but became NULL when cast",
    )


def _add_person_period(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    subparsers.add_parser(
        "person-period",
        help="stage 1: build the person-period table (one row per loan per month at risk)",
    )


def _add_vintage(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "vintage",
        help="stage 1: vintage hazard/incidence curves, the summary table and the figures",
    )
    parser.add_argument(
        "--figures-dir",
        default="reports/figures",
        help="where to write the PNGs (default: reports/figures)",
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="print the table only, skip plotting"
    )


def _add_hazard(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "hazard",
        help="stage 2: fit the cause-specific discrete-time hazard models",
    )
    parser.add_argument(
        "--figures-dir", default="reports/figures", help="where to write the PNGs"
    )
    parser.add_argument(
        "--reports-dir", default="reports", help="where to write the coefficient/diagnostic CSVs"
    )
    parser.add_argument(
        "--knot-study",
        action="store_true",
        help="also refit across loan-age knot counts and score each basis by BIC",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="fit and report but do not persist the models"
    )
    parser.add_argument(
        "--holdout-share",
        type=float,
        default=None,
        help=(
            "hold out this share of training LOANS and fit on the rest, saving to "
            "hazard_holdout.joblib. Pass with no value to use calibration.holdout_share. "
            "Stage 3 needs this model: a recalibrator has to be fitted on rows the "
            "hazard never saw, and stage 2's full-sample model has none"
        ),
        nargs="?",
        const=-1.0,
    )


def _add_calibrate(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "calibrate",
        help="stage 3: measure calibration on three populations and price the difference",
    )
    parser.add_argument(
        "--figures-dir", default="reports/figures", help="where to write the PNGs"
    )
    parser.add_argument("--reports-dir", default="reports", help="where to write the CSVs")
    parser.add_argument(
        "--no-figures", action="store_true", help="print the tables only, skip plotting"
    )
    parser.add_argument(
        "--no-npv",
        action="store_true",
        help="skip the cash-flow projection, which is the slow part",
    )
    parser.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="compute NPV at the configured economics only, not over the grid",
    )


def _add_decide(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "decide",
        help="stage 4: realised profit by approval cutoff; the profit-maximising rule",
    )
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--no-calibrated-arm",
        action="store_true",
        help=(
            "skip the arm that re-runs the decision under a recalibrated hazard, which "
            "is the experiment linking claim 2 to claim 3 and the slow part"
        ),
    )


def _add_ablate(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "ablate",
        help="stage 5: the B0-B4 out-of-time ablation, scored on realised cash",
    )
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--no-figures", action="store_true")


def _add_lgd(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "lgd",
        help="measure loss given default and the cure rate from realised losses",
    )
    parser.add_argument(
        "--reports-dir", default="reports", help="where to write the per-vintage CSV"
    )


def _run_schema() -> int:
    from cde.data.schema import ORIGINATION_COLUMNS, PERFORMANCE_COLUMNS, RELEASE

    print(f"Freddie Mac SFLD Release {RELEASE}")
    for label, cols in (
        ("ORIGINATION", ORIGINATION_COLUMNS),
        ("PERFORMANCE", PERFORMANCE_COLUMNS),
    ):
        print(f"\n{label} ({len(cols)} columns)")
        for i, (name, official) in enumerate(cols, 1):
            print(f"  {i:2d}  {name:<55} {official}")
    return 0


def _run_config(config_path: str) -> int:
    config = load_config(config_path)
    economics = config.economics
    definition = config.default_definition
    print(f"config          {config.path}")
    print(f"database        {config.database}")
    print(f"release         {config.data.release}  ({config.data.source} dataset)")
    print(f"train vintages  {list(config.data.train_vintages)}")
    print(f"test vintages   {list(config.data.test_vintages)}")
    print(f"horizon         {config.horizon.max_loan_age_months} months on book")
    print(f"default event   {definition.mode}", end="")
    if definition.mode == "delinquency":
        print(f"  (first passage to status >= '{definition.delinquency_threshold_code}')")
    else:
        print(f"  (zero balance codes {list(definition.zero_balance_default_codes)})")
    print(f"credit events   {list(definition.zero_balance_default_codes)}")
    print(f"informative     {list(definition.zero_balance_informative_exit_codes)}  (censored)")
    # Read from the model's own cause list rather than a config flag. A
    # `competing_risks.model_prepayment: false` switch used to live here and gated
    # nothing, so this line reported "censored" on a pipeline that had fitted,
    # diagnosed and persisted a prepayment hazard.
    from cde.models.hazard import CAUSE_TARGETS

    print(f"causes modelled {', '.join(sorted(CAUSE_TARGETS))}")
    print(f"discount mode   {economics.discount_rate_mode}")
    if economics.discount_rate_mode == "fixed":
        print(
            f"discount        {economics.annual_discount_rate:.2%} a year "
            f"= {economics.monthly_discount_rate:.6f} a month (compounded)"
        )
    else:
        print(
            f"                {economics.benchmark_series} + "
            f"{economics.discount_spread_bps:.0f}bp, per vintage"
        )
        _print_discount_bracket(config)
    # Severity is printed as its two measured factors and not only as the product,
    # because the NPV consumes both: the recovery leg splits into a cure part that never
    # reaches a disposition and a recovered part whose cash arrives years later.
    print(f"lgd (effective) {economics.lgd:.2%}   per default event, measured on train")
    print(
        f"                = {economics.loss_disposition_probability:.2%} "
        f"P(loss disposition | 90+ DPD)"
        f"  x  {economics.lgd_given_disposition:.2%} severity given one"
    )
    print(
        f"                  {economics.cure_share:.2%} of default events cure, and their "
        f"balance is booked unlagged"
    )
    print(
        f"recovery lag    {economics.recovery_lag_months} months, applied to the "
        f"disposition half of the recovery leg only"
    )
    print(
        f"servicing       {economics.servicing_cost_annual_bps:g} bps a year "
        f"= {economics.monthly_servicing_cost_rate:.8f} a month"
    )
    calibration = config.calibration
    print(
        f"calibration     {calibration.bins} {calibration.binning} bins; horizons "
        f"{list(calibration.horizons)} months"
    )
    print(
        f"                {calibration.holdout_share:.0%} in-time holdout "
        f"(seed {calibration.holdout_seed}); methods "
        f"{', '.join(calibration.methods)}"
    )
    print(
        f"                isotonic tail cap: a fitted value needs "
        f"{calibration.isotonic_min_tail_rows:,} rows of support"
    )
    return 0


def _print_discount_bracket(config: Config) -> None:
    """Show the per-vintage discount rate and whether it is economically coherent.

    The discount rate must sit inside [Treasury yield, note rate], because the note rate
    is the discount rate plus expected loss plus margin. Printed here rather than left to
    a test, because a spread that breaches it produces NPVs that look entirely normal.
    """
    from cde.economics.rates import bracket_check, load_benchmark

    try:
        benchmark = load_benchmark(config.economics.benchmark_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"                benchmark unavailable: {error}")
        return
    if not config.database.is_file():
        print("                (run `cde ingest` to check the rate against note rates)")
        return

    from cde.data.ingest import connect

    with connect(config.database, read_only=True) as con:
        note = (
            con.execute(
                "SELECT origination_vintage AS vintage, "
                "avg(original_interest_rate)/100.0 AS note "
                "FROM origination GROUP BY 1 ORDER BY 1"
            )
            .df()
            .set_index("vintage")["note"]
        )
    if note.empty:
        return
    checked = bracket_check(config, note, benchmark)
    print(f"                {'vintage':>7} {'benchmark':>10} {'discount':>9} "
          f"{'note':>7} {'headroom':>9}")
    for vintage, row in checked.iterrows():
        flag = "" if row["inside_bracket"] else "  <-- OUTSIDE THE BRACKET"
        print(
            f"                {int(vintage):>7} {row['benchmark_rate']:>9.2%} "
            f"{row['discount_rate']:>8.2%} {row['note_rate']:>6.2%} "
            f"{row['headroom_bps']:>7.0f}bp{flag}"
        )


def _run_ingest(config_path: str, *, check_casts: bool) -> int:
    from cde.data.ingest import cast_losses, connect, ingest

    config = load_config(config_path)
    with connect(config.database) as con:
        print(ingest(config, con).summary())
        print(f"  database    : {config.database}")
        if check_casts:
            losses = cast_losses(con)
            if not losses:
                print("  casts       : no values lost")
            else:
                print("  casts       : VALUES LOST — the schema types these wrongly, or the")
                print("                data holds a code the user guide does not document:")
                for table, column, count in losses:
                    print(f"                {table}.{column}: {count:,}")
                return 1
    return 0


def _run_person_period(config_path: str) -> int:
    from cde.data.ingest import connect
    from cde.features.person_period import build

    config = load_config(config_path)
    if not config.database.is_file():
        print(f"cde: no database at {config.database} — run `cde ingest` first")
        return 2
    with connect(config.database) as con:
        print(build(config, con).summary())
    return 0


def _run_vintage(config_path: str, *, figures_dir: str, no_figures: bool) -> int:
    from cde.data.ingest import connect
    from cde.eval import figures as figs
    from cde.eval.vintage import build, summarise, train_test_split, write_curves

    config = load_config(config_path)
    if not config.database.is_file():
        print(f"cde: no database at {config.database} — run `cde ingest` first")
        return 2

    horizon = config.horizon.max_loan_age_months
    with connect(config.database) as con:
        curves = train_test_split(config, build(config, con))
        summary = summarise(curves, horizon)

    print(summary.summary())
    frame = summary.frame
    # S(t) + F_d(t) + F_p(t) = 1 is an identity of the estimators, so checking it
    # would test floating-point arithmetic rather than the data. These two do say
    # something: one is a theorem that must hold, the other a prediction that need
    # not have.
    print(
        f"\n1 - KM >= Aalen-Johansen for every cohort: "
        f"{'yes' if (frame['km_default'] >= frame['cif_default']).all() else 'NO -- INVESTIGATE'}"
        f"   (a theorem; violation means the arithmetic is wrong)"
    )
    print(
        f"naive counting estimator differs from Aalen-Johansen by at most "
        f"{(frame['naive_default'] - frame['cif_default']).abs().max():.5f}"
    )
    print(
        f"  -- predicted, not guaranteed: with no loan censored before month {horizon}, "
        f"the counting estimator is unbiased. It breaks on an immature book."
    )

    out = Path(figures_dir)
    written = [write_curves(curves, out.parent / "vintage_curves.csv")]
    if not no_figures:
        written += [
            figs.default_at_horizon(frame, horizon, out / "vintage_default_at_horizon.png"),
            figs.default_curves(curves, out / "vintage_default_curves.png"),
            figs.estimator_comparison(
                curves, frame, (2000, 2002, 2003), out / "estimator_comparison.png"
            ),
            figs.hazard_curves(curves, out / "hazard_seasoning.png"),
            figs.exit_mix(curves, (2003, 2007), out / "exit_mix.png"),
        ]
        with connect(config.database) as con:
            calendar = con.execute(
                # The final calendar month of a cohort's window holds only loans
                # reaching the horizon, and an administratively censored loan cannot
                # record an event -- so that month's hazard is mechanically depressed
                # and plots as a vertical plunge to zero. Dropped, not smoothed over.
                "SELECT vintage, mean_loan_age, calendar_month, hazard_default FROM ("
                "  SELECT *, max(calendar_month) OVER (PARTITION BY vintage) AS last_month"
                "  FROM vintage_calendar_hazards"
                ") WHERE calendar_month < last_month ORDER BY vintage, calendar_month"
            ).df()
        written.append(
            figs.age_period_cohort(curves, calendar, out / "age_period_cohort.png")
        )
    print()
    for path in written:
        print(f"  wrote {path}")
    return 0


def _run_hazard(
    config_path: str,
    *,
    figures_dir: str,
    reports_dir: str,
    knot_study: bool = False,
    save_models: bool = True,
    holdout_share: float | None = None,
) -> int:
    from cde.data.ingest import connect
    from cde.eval import figures as figs
    from cde.models.hazard import (
        calibration_by_age,
        fit,
        load_person_period,
        required_columns,
    )

    config = load_config(config_path)
    if not config.database.is_file():
        print(f"cde: no database at {config.database} — run `cde ingest` first")
        return 2

    # -1.0 is argparse's sentinel for "--holdout-share given with no value".
    if holdout_share is not None and holdout_share < 0.0:
        holdout_share = config.calibration.holdout_share
    if holdout_share is not None and not 0.0 < holdout_share < 1.0:
        print(f"cde: --holdout-share must be in (0, 1), got {holdout_share}")
        return 2

    with connect(config.database) as con:
        frame = load_person_period(
            config, con, vintages=config.data.train_vintages, columns=required_columns(config)
        )
    print(
        f"training rows {len(frame):,}  loans {frame['loan_identifier'].nunique():,}  "
        f"vintages {list(config.data.train_vintages)}"
    )
    if holdout_share is not None:
        from cde.eval.calibration import holdout_mask

        held = holdout_mask(
            frame["loan_identifier"].to_numpy(),
            share=holdout_share,
            seed=config.calibration.holdout_seed,
        )
        frame = frame[~held].reset_index(drop=True)
        print(
            f"  holding out {holdout_share:.0%} of loans (seed "
            f"{config.calibration.holdout_seed}); fitting on {len(frame):,} rows / "
            f"{frame['loan_identifier'].nunique():,} loans"
        )
        print("  the held-out loans are where stage 3 fits its recalibrator: the only")
        print("  sample that is both unseen by this fit and available at decision time")

    models = fit(config, frame)
    print()
    print(models.summary())

    if save_models:
        from cde.models.store import default_path, save

        variant = None if holdout_share is None else "holdout"
        saved = save(
            config,
            models,
            default_path(config, variant=variant),
            holdout_share=0.0 if holdout_share is None else holdout_share,
        )
        print(f"\nsaved fitted models to {saved}")
        print("  stages 3-5 reload these rather than refitting, so their numbers")
        print("  are comparable; the design spec travels with them because its")
        print("  knots, medians and standardisation were learned on train only")
        assert saved == default_path(config, variant=variant)

    calibration = calibration_by_age(models, frame)
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    # Suffixed when a holdout was taken, so the two fits' diagnostics coexist. Stage 2
    # quotes the full-sample numbers and stage 3 the holdout ones; silently overwriting
    # would leave whichever ran last claiming to be both.
    tag = "" if holdout_share is None else "_holdout"
    written = [reports / f"hazard_calibration_by_age{tag}.csv"]
    calibration.to_csv(written[0], index=False)
    for cause, fitted in models.causes.items():
        path = reports / f"hazard_coefficients_{cause}{tag}.csv"
        fitted.coefficients().to_csv(path, index=False)
        written.append(path)

    print("\nmarginal event counts, predicted vs actual:")
    for cause in models.causes:
        predicted = (calibration[f"predicted_{cause}"] * calibration["at_risk"]).sum()
        actual = (calibration[f"observed_{cause}"] * calibration["at_risk"]).sum()
        print(f"  {cause:<12} predicted {predicted:>12,.0f}   actual {actual:>12,.0f}")

    # Residuals broken out by cohort. This is what showed the late-age misfit to be
    # a calendar effect rather than a spline-resolution one.
    from cde.models.selection import cohort_bias, late_age_threshold, residuals_by_vintage

    horizon = config.horizon.max_loan_age_months
    late = late_age_threshold(horizon)
    residuals = residuals_by_vintage(frame, models.causes["default"].hazard(frame))
    bias = cohort_bias(residuals, late)
    print("\nmean standardised residual by cohort (default hazard):")
    print("  a cohort systematically one-signed is mispredicted as a cohort,")
    print("  which no amount of loan-age flexibility can fix")
    for row in bias.itertuples():
        print(
            f"    {int(row.vintage)}   all ages {row.mean_z_all_ages:+6.2f}"
            f"   ages {late}-{horizon} {row.mean_z_late_ages:+6.2f}"
        )
    for name, table in (("hazard_residuals_by_vintage", residuals), ("hazard_cohort_bias", bias)):
        path = reports / f"{name}{tag}.csv"
        table.to_csv(path, index=False)
        written.append(path)

    if knot_study:
        from cde.models.selection import knot_study as run_knot_study

        print("\nloan-age knot study (BIC minimised = preferred basis):")
        table = run_knot_study(config, frame)
        print(table.to_string(index=False))
        path = reports / f"hazard_knot_study{tag}.csv"
        table.to_csv(path, index=False)
        written.append(path)

    written.append(
        figs.hazard_fit_check(
            calibration,
            Path(figures_dir) / f"hazard_fit_check{tag}.png",
            train_label=figs.vintage_label(config.data.train_vintages),
        )
    )
    print()
    for path in written:
        print(f"  wrote {path}")
    return 0


def _run_calibrate(
    config_path: str,
    *,
    figures_dir: str,
    reports_dir: str,
    no_figures: bool,
    no_npv: bool,
    no_sensitivity: bool,
) -> int:
    from cde.data.ingest import connect
    from cde.eval import calibration as cal
    from cde.eval import figures as figs
    from cde.models.store import default_path, load

    config = load_config(config_path)
    if not config.database.is_file():
        print(f"cde: no database at {config.database} — run `cde ingest` first")
        return 2

    path = default_path(config, variant="holdout")
    if not path.is_file():
        print(f"cde: no holdout model at {path}")
        print("     run `cde hazard --holdout-share` first.")
        print()
        print("     Stage 3 deliberately does NOT use stage 2's full-sample model. A")
        print("     recalibrator has to be fitted on loans the hazard never saw, and")
        print("     the full-sample fit has none: every training loan is in-sample, so")
        print("     a recalibrator fitted there would come back as the identity and")
        print("     stage 3 would report that calibration is worth nothing.")
        return 2
    models, card = load(config, path)
    if card.holdout_share <= 0.0:
        print(f"cde: {path} was fitted on ALL training loans (holdout_share 0).")
        print("     Refit with `cde hazard --holdout-share`.")
        return 2
    print(card.summary())
    print()

    with connect(config.database, read_only=True) as con:
        report = cal.build(
            config,
            models,
            con,
            holdout_share=card.holdout_share,
            with_npv=not no_npv,
            with_sensitivity=not no_sensitivity,
        )
    print(report.summary())

    reports = Path(reports_dir)
    written = [
        cal.write_scores(report, reports / "calibration_scores.csv"),
        cal.write_reliability(report, reports / "calibration_reliability.csv"),
    ]
    if not report.horizons.empty:
        written.append(
            cal.write_frame(report.horizons, reports / "calibration_by_horizon.csv")
        )
    if not report.npv.empty:
        written.append(cal.write_frame(report.npv, reports / "calibration_npv_impact.csv"))

    if not no_figures:
        out = Path(figures_dir)
        written.append(
            figs.reliability_curves(
                cal.reliability_table(report), out / "reliability_curves.png"
            )
        )
        # Vintage labels come from config, never from a literal in the caption: a
        # caption is baked into the PNG, so a stale one is invisible to every lint and
        # test here and survives until somebody reads the chart.
        test_vintages = config.data.test_vintages
        if not report.horizons.empty:
            written.append(
                figs.calibration_by_horizon(
                    report.horizons,
                    out / "calibration_by_horizon.png",
                    test_label=figs.vintage_label(test_vintages),
                )
            )
        if not report.npv.empty:
            written.append(
                figs.npv_impact(
                    report.npv,
                    out / "npv_impact.png",
                    train_label=figs.vintage_label(config.data.train_vintages),
                    sequential_fitted_on=(
                        f"the {test_vintages[0]} cohort" if len(test_vintages) > 1 else ""
                    ),
                    sequential_applied_to=figs.vintage_label(test_vintages[1:]),
                )
            )
    print()
    for item in written:
        print(f"  wrote {item}")
    failed = [message for ok, message in report.checks() if not ok]
    if failed:
        print()
        print("cde: a calibration check failed — the numbers above are not trustworthy:")
        for message in failed:
            print(f"  {message}")
        return 1
    return 0


def _run_decide(
    config_path: str,
    *,
    figures_dir: str,
    reports_dir: str,
    no_figures: bool,
    no_calibrated_arm: bool,
) -> int:
    from cde.calibration.calibrate import build_calibrator
    from cde.data.ingest import connect
    from cde.economics.decision import build_frame, evaluate
    from cde.eval import figures as figs
    from cde.eval.calibration import (
        eligible_for_npv,
        holdout_mask,
        load_loans,
        score_rows,
    )
    from cde.models.store import default_path, load

    config = load_config(config_path)
    if not config.database.is_file():
        print(f"cde: no database at {config.database} — run `cde ingest` first")
        return 2
    path = default_path(config, variant="holdout")
    if not path.is_file():
        print(f"cde: no holdout model at {path} — run `cde hazard --holdout-share` first")
        return 2
    models, card = load(config, path)
    print(card.summary())
    print()

    reports, out = Path(reports_dir), Path(figures_dir)
    with connect(config.database, read_only=True) as con:
        print("scoring and valuing the out-of-time book", flush=True)
        test = build_frame(config, models, con, config.data.test_vintages)

        chooser = None
        if config.decision.choose_thresholds_on_holdout:
            # Statistical thresholds are picked on the in-time holdout, never on the
            # book whose profit is then reported. Choosing a threshold by maximising
            # accuracy on the test set and reporting test profit at it is choosing the answer
            # and then quoting it.
            print("scoring the in-time holdout, where the thresholds are chosen",
                  flush=True)
            train_loans = eligible_for_npv(
                load_loans(config, con, vintages=config.data.train_vintages)
            )
            held = holdout_mask(
                train_loans["loan_identifier"].to_numpy(),
                share=card.holdout_share,
                seed=config.calibration.holdout_seed,
            )
            chooser = build_frame(
                config,
                models,
                con,
                config.data.train_vintages,
                loans=train_loans[held].reset_index(drop=True),
            )

        arms = {"raw": evaluate(config, test, chooser_frame=chooser)}

        if not no_calibrated_arm and len(config.data.test_vintages) > 1:
            # Does better calibration move the economic cutoff toward the oracle? This
            # is the only place claims 2 and 3 meet: stage 3 showed the level is wrong,
            # stage 4 shows the cutoff is an economic quantity, and if the two connect
            # then correcting the level should place the cutoff better and earn more.
            first = config.data.test_vintages[0]
            print(f"fitting a sequential calibrator on {first} for the calibrated arm",
                  flush=True)
            scored = score_rows(config, models, con, (first,), progress=False)
            calibrator = build_calibrator(
                "platt",
                tolerance=config.model.tolerance,
                max_iterations=config.model.max_iterations,
                isotonic_min_tail_rows=config.calibration.isotonic_min_tail_rows,
            ).fit(scored.hazard, scored.observed)
            print(f"  {calibrator.describe()}", flush=True)
            later = config.data.test_vintages[1:]
            print(f"re-deciding {list(later)} under the recalibrated hazard", flush=True)
            arms["recalibrated"] = evaluate(
                config,
                build_frame(config, models, con, later, calibrator=calibrator),
                chooser_frame=chooser,
            )
            arms["raw (same loans)"] = evaluate(
                config, build_frame(config, models, con, later), chooser_frame=chooser
            )

    written = []
    for name, result in arms.items():
        print()
        print(f"=== {name} ===")
        print(result.summary())
        slug = name.replace(" ", "_").replace("(", "").replace(")", "")
        written.append(_write(result.rules, reports / f"decision_rules_{slug}.csv"))
        written.append(_write(result.curve, reports / f"decision_curve_{slug}.csv"))

    baseline = arms["raw"]
    print()
    print("realised NPV per applicant is CASH, not the model's expectation. Scoring a")
    print("decision by the forecast that made it rewards overconfidence — see")
    print("src/cde/economics/realised.py.")

    if not no_figures:
        written.append(
            figs.decision_curve(baseline.curve, baseline.rules, out / "decision_curve.png")
        )
    print()
    for item in written:
        print(f"  wrote {item}")
    return 0


def _write(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _run_ablate(
    config_path: str, *, figures_dir: str, reports_dir: str, no_figures: bool
) -> int:
    from cde.calibration.calibrate import build_calibrator
    from cde.data.ingest import connect
    from cde.eval import figures as figs
    from cde.eval import harness
    from cde.eval.calibration import holdout_mask, score_rows
    from cde.models.binary import TARGET, loan_level_frame
    from cde.models.binary import fit as fit_binary
    from cde.models.store import default_path, load

    config = load_config(config_path)
    if not config.database.is_file():
        print(f"cde: no database at {config.database} — run `cde ingest` first")
        return 2
    path = default_path(config, variant="holdout")
    if not path.is_file():
        print(f"cde: no holdout model at {path} — run `cde hazard --holdout-share` first")
        return 2
    models, card = load(config, path)
    if card.holdout_share <= 0.0:
        print(f"cde: {path} was fitted on all training loans; refit with --holdout-share")
        return 2
    if len(config.data.test_vintages) < 2:
        print("cde: the ablation needs at least two test vintages — one to fit the")
        print("     sequential calibrator on, the rest to evaluate every arm on")
        return 2

    print(card.summary())
    print()
    train_vintages = config.data.train_vintages
    first_test, *later = config.data.test_vintages
    evaluated = tuple(later)

    with connect(config.database, read_only=True) as con:
        # --- the binary comparison model, on the SAME 80% of training loans ----------
        loan_level = loan_level_frame(config, con, train_vintages)
        held = holdout_mask(
            loan_level["loan_identifier"].to_numpy(),
            share=card.holdout_share,
            seed=config.calibration.holdout_seed,
        )
        binary = fit_binary(config, loan_level[~held].reset_index(drop=True))
        print(binary.summary())

        # --- B1's threshold, chosen on the in-time holdout ---------------------------
        holdout = loan_level[held].reset_index(drop=True)
        threshold, accuracy = harness.accuracy_maximising_threshold(
            binary.probability(holdout),
            holdout[TARGET].to_numpy(dtype=bool),
            holdout["resolved"].to_numpy(dtype=bool),
        )
        approves = float((binary.probability(holdout) < threshold).mean())
        print()
        print(
            f"B1's accuracy-maximising threshold, chosen on the in-time holdout: "
            f"{threshold:.6f}"
        )
        print(
            f"  accuracy {accuracy:.4f} there, approving {approves:.1%} of that holdout"
        )
        if approves > 0.999:
            print("  <-- it declines nobody. Accuracy is (TN+TP)/n and true negatives")
            print("      dominate at this base rate, so it falls monotonically as")
            print("      approval falls. B1 is therefore the do-nothing rule, which is")
            print("      the finding rather than a defect.")

        # --- the two recalibrators ---------------------------------------------------
        def _platt(hazard: np.ndarray, observed: np.ndarray, label: str) -> Calibrator:
            fitted = build_calibrator(
                "platt",
                tolerance=config.model.tolerance,
                max_iterations=config.model.max_iterations,
                isotonic_min_tail_rows=config.calibration.isotonic_min_tail_rows,
            ).fit(hazard, observed)
            print(f"  {label}: {fitted.describe()}", flush=True)
            return fitted

        print()
        print("fitting the recalibrators")
        # B4a: the honest one. Fitted on the held-out training loans, which is the only
        # sample both unseen by the hazard and available at the decision point.
        in_time_scored = score_rows(
            config, models, con, train_vintages,
            holdout_share=card.holdout_share, progress=False,
        )
        in_time = _platt(
            in_time_scored.hazard[in_time_scored.held],
            in_time_scored.observed[in_time_scored.held],
            "B4a, in-time holdout",
        )
        # B4b: the upper bound. Fitted on realised outcomes from the first test cohort,
        # which takes 120 months to observe, so no lender had it at the decision point.
        first_scored = score_rows(config, models, con, (first_test,), progress=False)
        sequential = _platt(
            first_scored.hazard, first_scored.observed, f"B4b, fitted on {first_test}"
        )

        print()
        print(f"running the ablation on {list(evaluated)}", flush=True)
        result = harness.build(
            config,
            models,
            binary,
            con,
            vintages=evaluated,
            accuracy_threshold=threshold,
            in_time=in_time,
            sequential=sequential,
            # Train-only, straight off the model card, so B2's prepayment assumption
            # cannot smuggle in anything the decision point would not have.
            prepayment_rate=float(card.causes["prepayment"]["base_rate"]),
        )

    print()
    print(result.summary())

    reports = Path(reports_dir)
    written = [_write(result.arms, reports / "ablation.csv")]
    written.append(_write(binary.coefficients(), reports / "binary_coefficients.csv"))

    if result.detail is not None:
        cuts = harness.failure_analysis(result.detail, arm="B3")
        print()
        print("FAILURE ANALYSIS (B3): expected minus realised NPV per loan, so a")
        print("positive gap means the model expected more value than arrived.")
        print()
        print("  by cohort — which vintages break it")
        print(f"    {'vintage':>8} {'loans':>8} {'expected':>10} {'realised':>10} "
              f"{'gap':>10} {'36m default':>12}")
        for row in cuts["by_vintage"].itertuples():
            print(
                f"    {int(row.origination_vintage):>8} {row.loans:>8,} "
                f"{row.expected_npv:>10,.0f} {row.realised_npv:>10,.0f} "
                f"{row.gap:>10,.0f} {row.default_rate:>11.2%}"
            )
        print()
        print("  by the model's own risk decile — a flat gap is a LEVEL error that")
        print("  recalibration fixes; a gap that varies with risk is a SHAPE error")
        print("  that a two-parameter monotone map cannot")
        print(f"    {'decile':>6} {'loans':>8} {'mean score':>11} {'expected':>10} "
              f"{'realised':>10} {'gap':>10} {'36m default':>12}")
        for row in cuts["by_risk"].itertuples():
            print(
                f"    {int(row.decile):>6} {row.loans:>8,} {row.mean_score:>11.5f} "
                f"{row.expected_npv:>10,.0f} {row.realised_npv:>10,.0f} "
                f"{row.gap:>10,.0f} {row.default_rate:>11.2%}"
            )
        for name, table in cuts.items():
            written.append(_write(table, reports / f"ablation_failure_{name}.csv"))
    if not no_figures:
        written.append(
            figs.ablation(result.arms, Path(figures_dir) / "ablation.png")
        )
    print()
    for item in written:
        print(f"  wrote {item}")
    return 0


def _run_lgd(config_path: str, *, reports_dir: str) -> int:
    from cde.data.ingest import connect
    from cde.economics.lgd import (
        by_vintage,
        cure_opportunity_cost,
        degradation,
        estimate_splits,
    )

    config = load_config(config_path)
    if not config.database.is_file():
        print(f"cde: no database at {config.database} — run `cde ingest` first")
        return 2

    with connect(config.database, read_only=True) as con:
        splits = estimate_splits(config, con)
        table = by_vintage(config, con)
        cure_cost = cure_opportunity_cost(config, con, config.data.train_vintages)

    for estimate in splits.values():
        print(estimate.summary())
        print()
    print(degradation(splits))

    train = splits["train"]
    print()
    print(f"configured economics.lgd        {config.economics.lgd:.4f}")
    print(f"train-measured effective LGD    {train.effective_lgd:.4f}")
    drift = abs(config.economics.lgd - train.effective_lgd)
    if drift > 5e-4:
        print(f"  <-- DRIFT of {drift:.4f}; update configs/default.yaml")
    else:
        print("  config matches the measurement")
    print()
    print(cure_cost.summary())
    print()
    print("Only the train figure may be used as an input. The test figure is a result:")
    print("a lender deciding in 2005 could not know what 2006-2008 losses would be.")

    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "lgd_by_vintage.csv"
    table.to_csv(path, index=False)
    print(f"\n  wrote {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cde", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    subparsers = parser.add_subparsers(dest="stage", required=True)
    _add_schema(subparsers)
    _add_config(subparsers)
    _add_ingest(subparsers)
    _add_person_period(subparsers)
    _add_vintage(subparsers)
    _add_hazard(subparsers)
    _add_calibrate(subparsers)
    _add_decide(subparsers)
    _add_ablate(subparsers)
    _add_lgd(subparsers)
    args = parser.parse_args(argv)

    try:
        if args.stage == "schema":
            return _run_schema()
        if args.stage == "config":
            return _run_config(args.config)
        if args.stage == "ingest":
            return _run_ingest(args.config, check_casts=args.check_casts)
        if args.stage == "person-period":
            return _run_person_period(args.config)
        if args.stage == "vintage":
            return _run_vintage(
                args.config, figures_dir=args.figures_dir, no_figures=args.no_figures
            )
        if args.stage == "lgd":
            return _run_lgd(args.config, reports_dir=args.reports_dir)
        if args.stage == "hazard":
            return _run_hazard(
                args.config,
                figures_dir=args.figures_dir,
                reports_dir=args.reports_dir,
                knot_study=args.knot_study,
                save_models=not args.no_save,
                holdout_share=args.holdout_share,
            )
        if args.stage == "ablate":
            return _run_ablate(
                args.config,
                figures_dir=args.figures_dir,
                reports_dir=args.reports_dir,
                no_figures=args.no_figures,
            )
        if args.stage == "decide":
            return _run_decide(
                args.config,
                figures_dir=args.figures_dir,
                reports_dir=args.reports_dir,
                no_figures=args.no_figures,
                no_calibrated_arm=args.no_calibrated_arm,
            )
        if args.stage == "calibrate":
            return _run_calibrate(
                args.config,
                figures_dir=args.figures_dir,
                reports_dir=args.reports_dir,
                no_figures=args.no_figures,
                no_npv=args.no_npv,
                no_sensitivity=args.no_sensitivity,
            )
    except ConfigError as error:
        parser.exit(2, f"cde: config error: {error}\n")
    except StaleModelError as error:
        # Previously uncaught, so a settings change produced a traceback rather than the
        # one-line instruction that fixes it. The message already explains what moved.
        parser.exit(2, f"cde: {error}\n")
    except FileNotFoundError as error:
        parser.exit(2, f"cde: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
