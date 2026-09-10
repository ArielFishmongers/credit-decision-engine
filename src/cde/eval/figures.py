"""Stage 1 and 2 figures.

Seven charts, each answering one question. The numbers all come from
``vintage_hazards`` -- the charts and any quoted figure cite the same query.

Design notes, because the choices were deliberate rather than default:

* **Vintage is ordinal, not categorical.** 2000 < 2001 < ... , so the cohorts get
  a single-hue ramp stepped light-to-dark by year, not eight unrelated hues. Train
  and test are the two ramps (blue, orange), which is also the group distinction.
* **Never eight lines on one axis.** Adjacent steps of an eight-step ramp measure
  a normal-vision colour difference of about 10 -- below the readability floor, so
  full-colour readers cannot reliably tell two of the lines apart. Train and test
  are therefore separate panels, each with five or fewer lines from one validated
  ramp, plus a direct end-label per line so identity never rests on colour alone.
* **Palettes were validated, not eyeballed.** Every palette below passed the
  six-check validator on the light surface: the three-series categorical set
  all-pairs, and each ordinal ramp for monotone lightness, adjacent step gaps and
  light-end contrast. Where a colour sits below 3:1 against the surface the relief
  rule applies, which is why every series is directly labelled.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

# --- palette (see references/palette.md; all values validated on #fcfcfb) -----------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: Ordinal ramp, blue, five steps. Validated: monotone, gaps >= 0.06, light end 2.06:1.
TRAIN_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
#: Ordinal ramp, orange, three steps. Validated: monotone, gaps >= 0.06, light end 2.31:1.
TEST_RAMP = ["#ee9165", "#e05f28", "#95390c"]
#: Categorical slots 1-3. Validated all-pairs: worst CVD dE 9.2, normal-vision 24.0.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]
#: Categorical slots 1-4, for the stacked composition. Validated adjacent: worst 9.1.
STACK = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BACKDROP = "#d6d5cf"

LINE_WIDTH = 2.0


def vintage_label(vintages: Sequence[int]) -> str:
    """Render a vintage set as "2000–2004" when contiguous, "2000, 2002" when not.

    Exists because hard-coded vintages in figure captions have now gone stale twice in
    this project. A caption is baked into the PNG, so a wrong one is invisible to every
    lint and test in the repo and survives until somebody reads the chart -- and the
    control arm of PROJECT_PLAN.md section 6c produced a chart claiming its recalibrator
    was fitted on 2000-2004 when it was fitted on 2000-2002. Every caption that names a
    cohort now takes the label from config through this function.
    """
    ordered = sorted(int(v) for v in vintages)
    if not ordered:
        return ""
    if len(ordered) == 1:
        return str(ordered[0])
    contiguous = ordered == list(range(ordered[0], ordered[-1] + 1))
    if contiguous:
        return f"{ordered[0]}\u2013{ordered[-1]}"
    return ", ".join(str(v) for v in ordered)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_MUTED,
            "ytick.labelcolor": INK_MUTED,
            "legend.frameon": False,
            "lines.linewidth": LINE_WIDTH,
            "figure.dpi": 130,
        }
    )


def _despine(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _finish(fig: Figure, title: str, subtitle: str, path: Path) -> Path:
    """Place the title block and save.

    Positions are computed in inches from the top of the figure rather than as
    axes fractions, and the subtitle is wrapped to the figure width. The earlier
    version placed both at fixed fractions and then saved with
    ``bbox_inches="tight"``, which cropped the reserved band and printed the
    subtitle on top of the title.
    """
    height = fig.get_figheight()
    wrap_at = max(50, int(fig.get_figwidth() * 15))
    lines = textwrap.wrap(subtitle, wrap_at) if subtitle else []
    band = 0.40 + 0.17 * len(lines)
    fig.tight_layout(rect=(0, 0, 1, 1 - band / height))
    fig.text(
        0.006, 1 - 0.26 / height, title,
        ha="left", va="baseline", fontsize=13, fontweight="bold", color=INK,
    )
    for i, line in enumerate(lines):
        fig.text(
            0.006, 1 - (0.46 + 0.17 * i) / height, line,
            ha="left", va="baseline", fontsize=9, color=INK_SECONDARY,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _smooth(values: pd.Series, window: int = 3) -> pd.Series:
    """Centred rolling mean.

    A monthly hazard on 50,000 loans with a sub-1% annual default rate is a ratio
    of small counts, so the raw series is dominated by sampling noise and the
    shape is unreadable. Smoothing is disclosed in every axis label that uses it,
    and the unsmoothed numbers are in reports/vintage_curves.csv.
    """
    return values.rolling(window, center=True, min_periods=1).mean()


def _label_ends(
    ax: Axes, entries: list[tuple[float, float, str, str]], min_gap: float = 0.05
) -> None:
    """Direct end-labels, pushed apart so they cannot overprint each other.

    Without this, vintages whose curves converge produce a single illegible smear
    of overlapping year labels -- which is what "2001/2002/2003" looked like on
    the first render of the hazard chart.
    """
    low, high = ax.get_ylim()
    span = high - low or 1.0
    placed: list[float] = []
    for x, y, text, colour in sorted(entries, key=lambda e: e[1]):
        target = (y - low) / span
        if placed and target - placed[-1] < min_gap:
            target = placed[-1] + min_gap
        placed.append(target)
        ax.annotate(
            text,
            xy=(x, low + target * span),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=colour,
            fontweight="bold",
            annotation_clip=False,
        )


def _ramp(split: str) -> list[str]:
    return TRAIN_RAMP if split == "train" else TEST_RAMP


def _split_panels(
    curves: pd.DataFrame,
    column: str,
    title: str,
    subtitle: str,
    ylabel: str,
    path: Path,
    *,
    smooth: bool = False,
) -> Path:
    """Two panels, train and test, each with one validated ordinal ramp.

    The y-axis is shared. That squashes the train panel, whose hazards are an
    order of magnitude smaller -- and that squashing is the finding, so it stays.
    Rescaling each panel independently would hide the regime shift the chart
    exists to show.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    pending: list[tuple[Axes, list[tuple[float, float, str, str]]]] = []
    for ax, split in zip(axes, ("train", "test"), strict=True):
        subset = curves[curves["split"] == split]
        vintages = sorted(subset["vintage"].unique())
        ramp = _ramp(split)
        labels: list[tuple[float, float, str, str]] = []
        for i, vintage in enumerate(vintages):
            series = subset[subset["vintage"] == vintage].sort_values("loan_age")
            values = _smooth(series[column]) if smooth else series[column]
            colour = ramp[i % len(ramp)]
            ax.plot(series["loan_age"], values, color=colour, label=str(vintage))
            labels.append(
                (float(series["loan_age"].iloc[-1]), float(values.iloc[-1]), str(vintage), colour)
            )
        ax.set_title(
            f"{'Train' if split == 'train' else 'Test'} vintages "
            f"({vintages[0]}\u2013{vintages[-1]})"
        )
        ax.set_xlabel("months on book")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_xlim(0, curves["loan_age"].max() + 6)
        ax.legend(loc="upper left", fontsize=8, labelcolor=INK_SECONDARY, ncols=2)
        _despine(ax)
        pending.append((ax, labels))
    # Deferred until both panels are drawn: the axes share their y-limit, so the
    # span the minimum label gap is measured against is not final until now.
    for ax, labels in pending:
        _label_ends(ax, labels)
    axes[0].set_ylabel(ylabel)
    return _finish(fig, title, subtitle, path)


def default_curves(curves: pd.DataFrame, path: Path) -> Path:
    """Fig 1 -- cumulative default incidence by month on book, per cohort."""
    return _split_panels(
        curves,
        "cif_default",
        "Cumulative default incidence by month on book",
        "Aalen\u2013Johansen cumulative incidence; default is first passage to 90+ DPD. "
        "Freddie Mac single-family sample, 50,000 loans per vintage year. "
        "Both panels share the y-axis, so the gap between them is the regime shift.",
        "cumulative % defaulted",
        path,
    )


def hazard_curves(curves: pd.DataFrame, path: Path) -> Path:
    """Fig 4 -- the monthly default hazard: the seasoning curve."""
    return _split_panels(
        curves,
        "hazard_default",
        "Monthly default hazard by month on book",
        "P(first 90+ DPD during month t | still at risk entering month t), "
        "3-month centred rolling mean. The shape is strongly non-monotonic, which is "
        "why loan age must enter the model flexibly rather than as a linear term \u2014 "
        "but see the age\u2013period\u2013cohort figure before reading it as maturation.",
        "monthly hazard (3-month mean)",
        path,
        smooth=True,
    )


def default_at_horizon(summary: pd.DataFrame, horizon: int, path: Path) -> Path:
    """Fig 2 -- the headline: cumulative default at the horizon, per cohort."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.0))
    rows = summary.sort_values("vintage")
    colours = [
        TRAIN_RAMP[2] if split == "train" else TEST_RAMP[1] for split in rows["split"]
    ]
    positions = range(len(rows))
    ax.barh(list(positions), rows["cif_default"], color=colours, height=0.62)
    for y, (value, split) in enumerate(zip(rows["cif_default"], rows["split"], strict=True)):
        ax.annotate(
            f"{value:.1%}",
            xy=(value, y),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=TRAIN_RAMP[4] if split == "train" else TEST_RAMP[2],
        )
    ax.set_yticks(list(positions))
    ax.set_yticklabels([f"{v}  ({s})" for v, s in zip(rows["vintage"], rows["split"], strict=True)])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlabel(f"cumulative % defaulted by month {horizon}")
    ax.set_xlim(0, rows["cif_default"].max() * 1.18)
    ax.grid(axis="y", visible=False)
    _despine(ax)
    return _finish(
        fig,
        f"Default rate at {horizon} months on book, by origination cohort",
        "The regime shift the out-of-time split is built to test. Train cohorts in "
        "blue, test cohorts in orange. Aalen\u2013Johansen cumulative incidence, "
        "first passage to 90+ DPD.",
        path,
    )


def estimator_comparison(
    curves: pd.DataFrame, summary: pd.DataFrame, vintages: tuple[int, ...], path: Path
) -> Path:
    """Fig 3 -- three estimators, and why only one of them is right.

    The three panels are ordered by how much prepayment there is; the fourth
    shows the consequence directly: the amount by which Kaplan-Meier overstates
    scales with the size of the competing risk.

    The naive counting estimator is drawn *dashed on top of* Aalen-Johansen
    rather than as its own solid line, because on this data the two agree to
    within 0.04 percentage points and a solid line simply hid the one underneath
    it. The agreement is the point, so the chart has to show it as agreement
    rather than as a missing series.
    """
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    flat = axes.flatten()
    for ax, vintage in zip(flat[:3], vintages, strict=True):
        subset = curves[curves["vintage"] == vintage].sort_values("loan_age")
        prepaid = summary.loc[summary["vintage"] == vintage, "cif_prepay"].iloc[0]
        ratio = (
            summary.loc[summary["vintage"] == vintage, "km_default"].iloc[0]
            / summary.loc[summary["vintage"] == vintage, "cif_default"].iloc[0]
        )
        ax.plot(
            subset["loan_age"], subset["km_default"], color=CATEGORICAL[1],
            label="1 \u2212 Kaplan\u2013Meier (wrong)",
        )
        ax.plot(
            subset["loan_age"], subset["cif_default"], color=CATEGORICAL[0],
            linewidth=2.8, label="Aalen\u2013Johansen (correct)",
        )
        ax.plot(
            subset["loan_age"], subset["naive_default"], color=CATEGORICAL[2],
            linewidth=2.4, linestyle=(0, (2, 6)), label="naive count / cohort",
        )
        last = subset.iloc[-1]
        _label_ends(
            ax,
            [
                (float(last["loan_age"]), float(last["km_default"]),
                 f"{last['km_default']:.1%}", CATEGORICAL[1]),
                (float(last["loan_age"]), float(last["cif_default"]),
                 f"{last['cif_default']:.1%}", CATEGORICAL[0]),
            ],
            min_gap=0.07,
        )
        ax.set_title(f"{vintage} \u2014 {prepaid:.0%} prepaid \u2014 overstated {ratio:.1f}\u00d7")
        ax.set_xlabel("months on book")
        ax.set_ylabel("cumulative % defaulted")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_xlim(0, subset["loan_age"].max() + 10)
        ax.legend(loc="lower right", fontsize=8, labelcolor=INK_SECONDARY)
        _despine(ax)
    flat[0].annotate(
        "the green dashes are the naive estimator, sitting\n"
        "exactly on Aalen\u2013Johansen: they agree to 0.07pp,\n"
        "because every cohort here is fully mature at 120 months",
        xy=(0.03, 0.68), xycoords="axes fraction", fontsize=8, color=INK_SECONDARY,
    )

    ax = flat[3]
    ordered = summary.sort_values("cif_prepay").reset_index(drop=True)
    ratios = ordered["km_default"] / ordered["cif_default"]
    ax.scatter(
        ordered["cif_prepay"], ratios, s=70, color=CATEGORICAL[1],
        edgecolor=SURFACE, linewidth=2, zorder=3,
    )
    # Alternate the label side so neighbouring vintages cannot overprint: 2003
    # and 2004 sit within a percentage point of each other on this axis.
    for i, (x, y, vintage) in enumerate(
        zip(ordered["cif_prepay"], ratios, ordered["vintage"], strict=True)
    ):
        ax.annotate(
            str(vintage), xy=(x, y), xytext=(0, 10 if i % 2 == 0 else -16),
            textcoords="offset points", ha="center", fontsize=8, color=INK_SECONDARY,
        )
    ax.axhline(1.0, color=AXIS, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(
        "no overstatement", xy=(0.01, 1.0), xytext=(0, 6),
        textcoords="offset points", fontsize=8, color=INK_MUTED,
    )
    ax.set_title("The error scales with the competing risk")
    ax.set_xlabel("share of cohort prepaid by the horizon")
    ax.set_ylabel("1 \u2212 KM  \u00f7  Aalen\u2013Johansen")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0.8, ratios.max() * 1.15)
    # Data-driven rather than pinned: prepayment shares moved from 0.40-0.91 at a
    # 60-month horizon to 0.77-0.94 at 120, and a hard-coded window left most of
    # the axis empty.
    margin = 0.04
    ax.set_xlim(
        max(0.0, float(ordered['cif_prepay'].min()) - margin),
        min(1.0, float(ordered['cif_prepay'].max()) + margin),
    )
    _despine(ax)

    return _finish(
        fig,
        "Three cumulative-incidence estimators; only one is right",
        "Treating prepayment as censoring (1 \u2212 Kaplan\u2013Meier) answers a "
        "counterfactual: what default would look like if nobody could refinance. A "
        "repaid loan is a real, final outcome that cannot default, so it belongs in "
        "the denominator \u2014 which is what the cumulative incidence function does.",
        path,
    )


def exit_mix(curves: pd.DataFrame, vintages: tuple[int, int], path: Path) -> Path:
    """Fig 5 -- what actually happens to a cohort, month by month.

    The bands sum to one at every month, which is a direct check that the exit
    taxonomy is exhaustive and mutually exclusive. It is also the chart that
    decides whether prepayment can be treated as a nuisance.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    # Counting shares, not the product-limit quantities. Of N loans in the cohort,
    # how many have prepaid / defaulted / been censored out / remain, at month t.
    # Those four partition the cohort by construction. An earlier version derived
    # the fourth band as 1 - S - F_d - F_p, which is identically zero because that
    # is an algebraic identity of the estimators, so the band was invisible for a
    # structural reason rather than because censoring is rare.
    bands = (
        ("not_yet_exited_share", "not yet exited", STACK[0]),
        ("naive_prepay", "prepaid (competing risk)", STACK[1]),
        ("naive_default", "defaulted", STACK[2]),
        ("naive_censored", "censored out", STACK[3]),
    )
    for ax, vintage in zip(axes, vintages, strict=True):
        subset = curves[curves["vintage"] == vintage].sort_values("loan_age").copy()
        ax.stackplot(
            subset["loan_age"],
            *[subset[column] for column, _, _ in bands],
            labels=[label for _, label, _ in bands],
            colors=[colour for _, _, colour in bands],
            edgecolor=SURFACE,
            linewidth=2,
        )
        ax.set_title(f"{vintage} cohort")
        ax.set_xlabel("months on book")
        ax.set_xlim(0, subset["loan_age"].max())
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.grid(visible=False)
        _despine(ax)
    axes[0].set_ylabel("share of the cohort")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(
        handles[::-1],
        labels[::-1],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        labelcolor=INK_SECONDARY,
    )
    return _finish(
        fig,
        "Where a cohort goes: prepayment dominates, default is the thin band",
        "Counting shares of the origination cohort. Prepayment being the largest band "
        "by far \u2014 77% to 94% of a cohort within ten years \u2014 is why it cannot be "
        "treated as a censoring nuisance, and is what settles whether stage 2 needs "
        "competing risks.",
        path,
    )


def age_period_cohort(curves: pd.DataFrame, calendar: pd.DataFrame, path: Path) -> Path:
    """Fig 6 -- the same numbers on two x-axes.

    For a loan from vintage v at age a, the calendar period is p = v + a exactly.
    Age, cohort and period are perfectly collinear, so no data separates
    maturation from vintage quality from macroeconomics. What a picture *can*
    show: a shock that lines up vertically across cohorts sitting at different
    ages is a period effect, and no age-based story produces it.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for split in ("train", "test"):
        ramp = _ramp(split)
        vintages = sorted(curves.loc[curves["split"] == split, "vintage"].unique())
        for i, vintage in enumerate(vintages):
            colour = ramp[i % len(ramp)]
            by_age = curves[curves["vintage"] == vintage].sort_values("loan_age")
            axes[0].plot(
                by_age["loan_age"],
                _smooth(by_age["hazard_default"]),
                color=colour,
                label=str(vintage),
            )
            by_month = calendar[calendar["vintage"] == vintage].sort_values("calendar_month")
            axes[1].plot(
                by_month["calendar_month"],
                _smooth(by_month["hazard_default"]),
                color=colour,
                label=str(vintage),
            )
    axes[0].set_title("Aligned by loan age")
    axes[0].set_xlabel("months on book")
    axes[0].set_ylabel("monthly default hazard (3-month mean)")
    axes[0].set_xlim(0, curves["loan_age"].max())
    axes[1].set_title("Aligned by calendar date")
    axes[1].set_xlabel("calendar month")
    axes[0].legend(fontsize=8, labelcolor=INK_SECONDARY, ncols=2, loc="upper left")
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        _despine(ax)
    axes[1].annotate(
        "every cohort peaks here,\nat a different age",
        xy=(0.46, 0.86),
        xycoords="axes fraction",
        fontsize=8,
        color=INK_SECONDARY,
        ha="center",
    )
    return _finish(
        fig,
        "The age\u2013period\u2013cohort problem, in two pictures of the same data",
        "period = vintage + age, exactly, so the three effects are perfectly "
        "collinear and no decomposition is identified without a restriction imposed "
        "by assumption. What the right panel does establish: the cohorts peak "
        "together in calendar time while sitting at different ages, which no "
        "pure-maturation story can produce.",
        path,
    )


def hazard_fit_check(
    calibration: pd.DataFrame, path: Path, *, train_label: str = ""
) -> Path:
    """Stage 2 -- predicted against observed hazard, by month on book, per cause.

    The first thing to look at after fitting, and the one that decides whether the
    loan-age spline basis is rich enough. If the mean predicted hazard at each age
    does not track the observed rate at that age, the basis is too coarse and no
    amount of covariate work will fix it.

    In-sample by construction, so this is a specification check, not evidence of
    generalisation. A logistic fit reproduces the marginal event count almost
    exactly; what it need not reproduce, and what is actually being checked here, is
    the *shape* against a variable entered through a restricted basis.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    causes = (
        ("default", "Default hazard", axes[0]),
        ("prepayment", "Prepayment hazard", axes[1]),
    )
    for cause, title, ax in causes:
        observed, predicted = f"observed_{cause}", f"predicted_{cause}"
        if observed not in calibration.columns:
            continue
        ax.plot(
            calibration["loan_age"],
            calibration[observed],
            marker="o",
            markersize=4,
            linestyle="none",
            color=CATEGORICAL[0],
            label="observed",
        )
        ax.plot(
            calibration["loan_age"],
            calibration[predicted],
            color=CATEGORICAL[1],
            linewidth=2.4,
            label="model",
        )
        ax.set_title(title)
        ax.set_xlabel("months on book")
        ax.set_ylabel("monthly hazard")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_ylim(bottom=0)
        # "best" rather than "upper left": at a 120-month horizon the prepayment hazard
        # peaks early, and a pinned upper-left legend sat on top of that peak.
        ax.legend(loc="best", fontsize=8, labelcolor=INK_SECONDARY)
        _despine(ax)
    return _finish(
        fig,
        "Does the fitted hazard reproduce the shape it was given?",
        "Mean predicted hazard against the observed rate at each month on book"
        + (f", training vintages {train_label}" if train_label else "")
        + ". In-sample, so this checks the specification — whether the loan-age spline "
        "basis is rich enough — and not whether the model generalises.",
        path,
    )


# --- stage 3 -------------------------------------------------------------------------


def _diagonal(ax: Axes, low: float, high: float, *, label: bool = True) -> None:
    """Perfect calibration. The only reference line that matters on a reliability plot.

    Labelled on one panel only. Every panel carries the same line, so repeating the text
    three times buys nothing and costs a collision: placed above the line's upper end it
    sat level with the panel title and read as a second heading, and placed below it ran
    straight through the data in whichever panel happened to bend that way.
    """
    ax.plot(
        [low, high], [low, high],
        color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1,
    )
    if label:
        # Below and right of the line, in the region a model that under-predicts leaves
        # empty -- and under-prediction is what this book does out of time.
        ax.annotate(
            "perfectly calibrated",
            xy=(high, low), xytext=(-6, 10), textcoords="offset points",
            ha="right", va="bottom", fontsize=7.5, color=INK_MUTED, style="italic",
        )


def reliability_curves(table: pd.DataFrame, path: Path) -> Path:
    """Stage 3 -- observed against predicted, one panel per population.

    **Log axes on both sides, which is not the conventional choice and is the right one
    here.** The monthly default hazard spans nearly four orders of magnitude across the
    quantile bins, so a linear reliability plot compresses nineteen of the twenty bins
    into the bottom-left corner and shows only the riskiest one. On log axes the diagonal
    is still the diagonal and every bin is legible.

    The panels are ordered as a control, then the two real measurements: in-sample must
    sit on the diagonal by construction, so it calibrates the reader's eye for how much
    scatter is just sampling noise before they look at the two that carry the finding.
    """
    _style()
    populations = [
        p for p in ("train (in-sample)", "holdout (in-time)", "test (out-of-time)")
        if p in set(table["population"])
    ]
    fig, axes = plt.subplots(1, len(populations), figsize=(3.7 * len(populations), 4.3),
                             sharex=True, sharey=True)
    axes_list = list(np.atleast_1d(axes))
    variants = [v for v in ("raw", "platt", "isotonic") if v in set(table["variant"])]
    zero_bins = 0
    low, high = 1.0, 1e-12
    for column in ("mean_predicted", "observed_rate"):
        values = table[column][table[column] > 0]
        if not values.empty:
            low = min(low, float(values.min()))
            high = max(high, float(values.max()))
    # Room below the smallest real value for the zero-bin markers to be visible.
    low, high = low * 0.25, high * 2.0
    floor = low * 1.5

    for ax, population in zip(axes_list, populations, strict=True):
        panel = table[table["population"] == population]
        _diagonal(ax, low, high, label=population == populations[0])
        for i, variant in enumerate(variants):
            series = panel[panel["variant"] == variant].sort_values("mean_predicted")
            if series.empty:
                continue
            colour = CATEGORICAL[i % len(CATEGORICAL)]
            # A zero on EITHER axis is unplottable on a log scale, and both occur here:
            # a bin in which nothing defaulted, and -- less obviously -- a bin whose
            # isotonic-corrected prediction is exactly 0, which happens when the block
            # PAVA fitted held no events at all. Left to matplotlib the zero-x points
            # were silently clipped and the connecting line ran flat to the axis edge,
            # which read as a real feature of the data. They are drawn on the floor
            # instead, hollow, and counted in the caption.
            plottable = (series["observed_rate"] > 0) & (series["mean_predicted"] > 0)
            drawn, hidden = series[plottable], series[~plottable]
            zero_bins += len(hidden)
            # `raw` goes down as a wide band and the corrected variants as thin lines on
            # top of it. They very nearly coincide, which IS the finding -- but drawn at
            # equal width the first series plotted is simply invisible, and a reader
            # cannot tell a hidden line from a missing one.
            wide = variant == "raw"
            ax.plot(
                drawn["mean_predicted"], drawn["observed_rate"],
                marker="none" if wide else "o", markersize=3.2,
                linewidth=5.0 if wide else 1.7,
                color=colour, label=variant, zorder=2 if wide else 3,
                solid_capstyle="round",
            )
            if not hidden.empty:
                ax.plot(
                    hidden["mean_predicted"].clip(lower=floor),
                    hidden["observed_rate"].clip(lower=floor),
                    marker="o", markersize=4.5, linestyle="none",
                    markerfacecolor="none", markeredgewidth=1.3,
                    color=colour, zorder=4,
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_title(population)
        ax.set_xlabel("predicted monthly hazard")
        ax.legend(loc="upper left", fontsize=8, labelcolor=INK_SECONDARY)
        _despine(ax)
    axes_list[0].set_ylabel("observed monthly default rate")
    note = (
        f" {zero_bins} hollow marker(s) sit on the floor: bins holding an exact zero on "
        f"one axis, which a log scale cannot place — either no loan defaulted, or "
        f"isotonic corrected the prediction to exactly 0."
        if zero_bins
        else ""
    )
    return _finish(
        fig,
        "Does a predicted 2% mean 2%?",
        "Observed default rate against predicted monthly hazard, in 20 equal-count "
        "bins. Points below the diagonal mean the model predicted more default than "
        "happened; above, less. In-sample is a control and must sit on the diagonal — "
        "a converged logistic fit is calibrated on its own training rows by "
        "construction. Raw is the wide band; the corrected variants are the thin "
        "lines, and they lie almost exactly on top of it." + note,
        path,
    )


def calibration_by_horizon(
    horizons: pd.DataFrame, path: Path, *, test_label: str = ""
) -> Path:
    """Stage 3 -- predicted against observed CUMULATIVE incidence, per horizon.

    The monthly view and this one can disagree, which is the reason both exist. The
    model emits a monthly hazard; what a lender prices is the chance of default over
    years, and that is the monthly hazards compounded through the survival product. A
    small monthly bias compounds — in either direction — so a model can look adequate
    per month and be badly wrong over three years.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    frame = horizons.sort_values("horizon_months")
    width = 0.38
    positions = np.arange(len(frame), dtype=float)
    ax = axes[0]
    ax.bar(positions - width / 2, frame["predicted_rate"], width,
           color=CATEGORICAL[0], label="predicted")
    ax.bar(positions + width / 2, frame["observed_rate"], width,
           color=CATEGORICAL[1], label="observed")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{int(m)}m" for m in frame["horizon_months"]])
    ax.set_ylabel("cumulative % defaulted")
    ax.set_xlabel("months on book")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title("Cumulative default incidence")
    ax.legend(loc="upper left", fontsize=8, labelcolor=INK_SECONDARY)
    _despine(ax)

    ax = axes[1]
    ax.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.plot(frame["horizon_months"], frame["ratio"], marker="o", markersize=5,
            color=CATEGORICAL[2])
    for row in frame.itertuples():
        ax.annotate(
            f"{row.ratio:.2f}x",
            xy=(row.horizon_months, row.ratio), xytext=(0, 8),
            textcoords="offset points", ha="center", fontsize=8,
            color=INK_SECONDARY, fontweight="bold",
        )
    ax.set_ylabel("predicted / observed")
    ax.set_xlabel("months on book")
    ax.set_title("Ratio, and which way it moves with the horizon")
    ax.set_ylim(bottom=0.0)
    _despine(ax)
    return _finish(
        fig,
        "Cumulative incidence, where the monthly errors compound",
        "Out-of-time book"
        + (f" ({test_label})" if test_label else "")
        + ". A ratio below 1 means the model expected less default than happened. Loans "
        "censored before a horizon are excluded from it rather than counted as "
        "survivors, which would understate incidence by the share of the book that "
        "stopped reporting.",
        path,
    )


def npv_impact(
    npv: pd.DataFrame,
    path: Path,
    *,
    train_label: str = "",
    sequential_fitted_on: str = "",
    sequential_applied_to: str = "",
) -> Path:
    """Stage 3 -- the money chart, and the whole reason the stage exists.

    Left: NPV per loan under raw and recalibrated probabilities, at the configured
    economics. Right: the same difference across the severity grid, one line per
    discount spread — because a headline that survives only at one assumption is not a
    finding, and the point value of neither assumption is defensible on its own.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    # Prefer the scope carrying the sequential arm: in-time recalibration barely moves
    # these probabilities, so a panel showing only raw/platt/isotonic shows three almost
    # identical bars and reads as "calibration is worth nothing" -- which is a statement
    # about the recalibrator, not about calibration.
    if "scope" in npv.columns:
        preferred = npv[npv["scope"] == "sequential"]
        full = preferred if not preferred.empty else npv[npv["scope"] == "headline"]
        grid = npv[npv["scope"] == "sensitivity"]
        if grid.empty:
            grid = full
    else:
        full = grid = npv
    base_lgd = float(full["lgd"].iloc[0])
    base_spread = float(full["discount_spread_bps"].iloc[0])
    headline = full

    ax = axes[0]
    variants = list(headline["variant"])
    values = headline["npv_per_loan"].to_numpy(dtype=float)
    #: One colour per variant, reused by the right panel so the two agree.
    colours = [STACK[i % len(STACK)] for i in range(len(variants))]
    ax.bar(np.arange(len(variants)), values, 0.6, color=colours)
    ax.axhline(0.0, color=AXIS, linewidth=1.0)
    for i, value in enumerate(values):
        ax.annotate(
            f"{value:,.0f}",
            xy=(i, value), xytext=(0, 6 if value >= 0 else -14),
            textcoords="offset points", ha="center", fontsize=8,
            color=INK_SECONDARY, fontweight="bold",
        )
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels(variants)
    ax.set_ylabel("NPV per loan ($)")
    # The population belongs in the title. These bars are drawn on whichever scope
    # carries the most variants, which is NOT the same population as the headline table,
    # and a panel that does not say which loans it priced invites the two to be compared.
    ax.set_title(
        f"{int(full['loans'].iloc[0]):,} loans, LGD {base_lgd:.4f}, "
        f"spread {base_spread:.0f}bp"
    )
    _despine(ax)

    ax = axes[1]
    # Colour carries the variant, matching the bars on the left, and the discount spread
    # becomes a band rather than its own set of lines. An earlier version crossed variant
    # against spread as colour-by-spread and style-by-variant: six series, a six-entry
    # legend sitting over the data, and the primary comparison -- which variant -- was
    # the harder of the two to read.
    #
    # The band is also the better answer to the panel's own question. If it stays wholly
    # on one side of zero, the sign of the effect survives the assumption; the exact
    # line matters much less than that.
    grid = grid[grid["variant"] != "raw"]
    order = [v for v in variants if v in set(grid["variant"])]
    for variant in order:
        colour = colours[variants.index(variant)]
        series = grid[grid["variant"] == variant]
        band = series.groupby("lgd")["delta_vs_raw"].agg(["min", "max", "median"])
        band = band.sort_index()
        if len(band) > 1 and not (band["min"] == band["max"]).all():
            ax.fill_between(
                band.index, band["min"], band["max"],
                color=colour, alpha=0.18, linewidth=0,
            )
        ax.plot(
            band.index, band["median"],
            marker="o", markersize=3.5, color=colour, label=variant,
        )
    ax.axhline(0.0, color=AXIS, linewidth=1.0)
    ax.set_xlabel("loss given default (effective, per default event)")
    ax.set_ylabel("NPV per loan, recalibrated − raw ($)")
    spreads = sorted(grid["discount_spread_bps"].unique())
    ax.set_title(
        "Does the sign survive the assumptions?"
        if len(spreads) < 2
        else f"Does the sign survive? (band spans {spreads[0]:.0f}-{spreads[-1]:.0f}bp)"
    )
    ax.legend(loc="best", fontsize=8, labelcolor=INK_SECONDARY)
    _despine(ax)
    return _finish(
        fig,
        "What honest probabilities are worth, per loan",
        "Expected NPV projected from origination over the 120-month horizon under "
        "the fitted competing-risks hazards. Platt and isotonic were fitted on an "
        "in-time holdout of "
        + (f"{train_label} " if train_label else "training ")
        + "loans and never saw an out-of-time outcome — which is why they move "
        "nothing."
        + (
            f" The sequential arm was fitted on {sequential_fitted_on} and applied "
            f"to {sequential_applied_to}, so it DID see out-of-time outcomes: an "
            "upper bound on what recalibration could achieve, not an operating "
            "recommendation, since observing a cohort fully takes ten years."
            if "sequential" in set(headline["variant"]) and sequential_fitted_on
            else ""
        ),
        path,
    )


def decision_curve(curve: pd.DataFrame, rules: pd.DataFrame, path: Path) -> Path:
    """Stage 4 -- realised profit against approval rate, with each rule's cutoff marked.

    **The chart claim 3 exists for.** One score, one model, one book; the only thing that
    differs between the marked points is the criterion used to turn the score into a
    cutoff, and they are hundreds of dollars an applicant apart.

    The right panel carries accuracy on the same x-axis, and it is the more damning of
    the two: accuracy climbs monotonically to its maximum at 100% approval, so the
    accuracy-maximising rule is the do-nothing rule — and it gets there while reporting a
    headline in the nineties.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    ax.axhline(0.0, color=AXIS, linewidth=1.0)
    ax.plot(
        curve["approval_rate"], curve["npv_per_applicant"],
        color=CATEGORICAL[0], linewidth=2.4, zorder=2,
    )
    marks = [
        (row.rule, float(row.approval_rate), float(row.npv_per_applicant))
        for row in rules.itertuples()
    ]
    labels: list[tuple[float, float, str, str]] = []
    for i, (rule, rate, value) in enumerate(marks):
        colour = STACK[i % len(STACK)]
        ax.plot([rate], [value], marker="o", markersize=7, linestyle="none",
                color=colour, zorder=4)
        labels.append((rate, value, rule, colour))
    _label_ends(ax, labels, min_gap=0.075)
    ax.set_xlabel("share of applicants approved")
    ax.set_ylabel("realised NPV per applicant ($)")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlim(0, 1.28)
    ax.set_title("Realised profit, and where each rule puts the cutoff")
    _despine(ax)

    ax = axes[1]
    placed = rules.set_index("rule")["approval_rate"].to_dict()
    for i, (column, label, rule) in enumerate(
        (("accuracy", "accuracy", "accuracy-max"), ("f1", "F1", "F1-max"))
    ):
        colour = CATEGORICAL[i % len(CATEGORICAL)]
        ax.plot(curve["approval_rate"], curve[column], color=colour, linewidth=2.0,
                label=label)
        best = curve.iloc[int(curve[column].to_numpy().argmax())]
        ax.plot([best["approval_rate"]], [best[column]], marker="o", markersize=7,
                linestyle="none", color=colour, zorder=4)
        ax.annotate(
            f"peaks here on TEST, at {best['approval_rate']:.0%}",
            xy=(best["approval_rate"], best[column]), xytext=(-8, 9),
            textcoords="offset points", ha="right", fontsize=7.5,
            color=INK_SECONDARY, fontweight="bold",
        )
        # Where the rule ACTUALLY sits, having chosen its threshold on the in-time
        # holdout. The distance between the two marks is the leakage that picking the
        # threshold on the evaluation set would have quietly collected -- so it is drawn
        # rather than left as a discrepancy between the two panels.
        rate = placed.get(rule)
        if rate is not None and abs(rate - float(best["approval_rate"])) > 0.01:
            ax.axvline(rate, color=colour, linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
            ax.annotate(
                f"rule lands at {rate:.0%}\n(threshold from the holdout)",
                xy=(rate, 0.06 + 0.5 * i), xytext=(-6, 0), textcoords="offset points",
                ha="right", va="bottom", fontsize=7.5, color=colour,
            )
    ax.set_xlabel("share of applicants approved")
    ax.set_ylabel("statistical metric")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.12)
    ax.set_title("The statistical metrics, on the same axis")
    ax.legend(loc="upper left", fontsize=8, labelcolor=INK_SECONDARY)
    _despine(ax)

    return _finish(
        fig,
        "The cutoff is an economic quantity, not a statistical one",
        "Every rule ranks applicants by the same predicted default probability from the "
        "same fitted hazard, so only the CRITERION differs. Profit is realised cash, "
        "not the model's own expectation — scoring a decision by the forecast that made "
        "it rewards overconfidence. Statistical thresholds are chosen on the in-time "
        "holdout, never on this book — so a rule need not sit where its metric peaks "
        "here, and the distance between the two is the leakage that choosing on the test "
        "set would have collected. The oracle is hindsight and bounds what a perfectly "
        "placed cutoff on this ranking could have earned.",
        path,
    )


def ablation(arms: pd.DataFrame, path: Path) -> Path:
    """Stage 5 -- the B0-B4 ablation as a waterfall, plus each arm's capture of its own oracle.

    The left panel is the headline table drawn: each bar is realised NPV per applicant,
    and the step between bars is what that arm added. Reading the steps rather than the
    levels is the point — the levels all share the same terminal-value and censoring
    conventions, which cancel in a difference and do not cancel in a level.

    The right panel separates the two ways an arm can leave money on the table. Realised
    profit mixes how well a score *ranks* applicants with whether its cutoff is in the
    right place; capture is the share of what that arm's own ranking could have earned
    with a perfectly placed cutoff. A low capture and a good AUC calls for moving the
    cutoff; a high capture and a poor AUC calls for a better model. They are opposite
    fixes and the headline cannot tell them apart.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    keys = list(arms["arm"])
    values = arms["realised"].to_numpy(dtype=float)
    positions = np.arange(len(keys), dtype=float)

    ax = axes[0]
    # One colour for the arms a lender could actually run, and a second for the upper
    # bound that saw outcomes it could not have had. The arms are an ordinal progression
    # and the x labels carry their identity, so six categorical hues would say nothing --
    # and STACK only holds four, so B4a and B4b were silently reusing B0's and B1's.
    colours = [
        CATEGORICAL[1] if key.endswith("b") and key.startswith("B4") else CATEGORICAL[0]
        for key in keys
    ]
    ax.bar(positions, values, 0.62, color=colours, zorder=2)
    ax.axhline(0.0, color=AXIS, linewidth=1.0)
    # The step from the previous arm, which is what each arm actually contributed.
    for i in range(1, len(keys)):
        step = values[i] - values[i - 1]
        if abs(step) < 1e-9:
            continue
        ax.annotate(
            f"{step:+,.0f}",
            xy=(positions[i], max(values[i], values[i - 1])),
            xytext=(0, 22), textcoords="offset points", ha="center", fontsize=7.5,
            color=INK_MUTED, style="italic",
        )
    for i, value in enumerate(values):
        ax.annotate(
            f"{value:,.0f}",
            xy=(positions[i], value),
            xytext=(0, 7 if value >= 0 else -15), textcoords="offset points",
            ha="center", fontsize=8, color=INK_SECONDARY, fontweight="bold",
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(keys)
    ax.set_ylabel("realised NPV per applicant ($)")
    ax.set_title("Realised profit, and what each arm added")
    _despine(ax)

    ax = axes[1]
    ax.bar(positions - 0.19, arms["capture_cut"].to_numpy(dtype=float), 0.36,
           color=CATEGORICAL[0], label="capture of best cutoff", zorder=2)
    ax.bar(positions + 0.19, arms["auc"].to_numpy(dtype=float), 0.36,
           color=CATEGORICAL[1], label="AUC", zorder=2)
    ax.axhline(0.5, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    # Anchored left, where B0 has no capture bar, rather than right over B4b's pair.
    # Short, and anchored in the gap B0/B1 leave by having no capture bar. The longer
    # wording ran into B2's bars.
    ax.annotate("AUC 0.5", xy=(positions[0] - 0.42, 0.5), xytext=(0, 5),
                textcoords="offset points", ha="left", fontsize=7.5,
                color=INK_MUTED, style="italic")
    ax.set_xticks(positions)
    ax.set_xticklabels(keys)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel("share")
    ax.set_title("Is the ranking weak, or the cutoff misplaced?")
    ax.legend(loc="upper left", fontsize=8, labelcolor=INK_SECONDARY)
    _despine(ax)

    return _finish(
        fig,
        "Where the value actually comes from",
        "Out-of-time applicants, every arm on the same loans and the same economics, "
        "each scored on realised cash rather than on its own forecast. B1 is the typical "
        "public notebook; B2 adds only the decision layer, B3 the survival framing, B4 "
        "recalibration. B4a is what a lender could have run at the time; B4b (orange) "
        "saw the first test cohort's realised outcomes, which take ten years to observe, "
        "so it is an upper bound rather than a rule.",
        path,
    )
