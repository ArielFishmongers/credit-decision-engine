#!/usr/bin/env python
"""Compare the crisis split against the less-crisis-exposed control arm.

WHY THIS EXISTS. The headline stage 3 experiment trains on 2000-2004 and tests on
2006-2008, straddling the housing crisis. That makes "the model under-predicts out of
time" close to guaranteed by the split rather than discovered in it, and it leaves the
measured NPV effect uninterpretable on its own: it could be a regime effect specific to a
once-in-a-generation break, or ordinary out-of-time drift that the crisis merely
amplifies. Those two readings imply completely different write-ups.

So the same pipeline is run again on `configs/calm_control.yaml` (train 2000-2002, test
2003-2004) and this script puts the two arms side by side.

## Compare NPV in dollars and in per-dollar-of-principal, NEVER as a percentage

The percentage change in NPV is worthless across these two arms and it took running them
to see why. The 2003-2004 vintages carry the thinnest note-rate-over-discount-rate spread
in the book -- 50bp and 39bp of headroom against 110bp for 2000 -- so their projected NPV
per loan is near zero and can be negative: -$150 against +$3,396 for the crisis arm's test
book. A percentage change against a denominator 23 times smaller, and of the opposite
sign, is not a comparable quantity. In dollars the crisis effect is -$1,349 a loan against
-$94; per dollar of principal, -0.69 against -0.06 percentage points. Those are comparable
and they tell the same story; the percentages (-39.7% against -62.7%) invert it.

## A normalisation that was tried and does NOT work, recorded so it is not retried

The obvious way to compare two arms with different regime shifts is to divide the shift
out: a model that learned nothing about *why* loans default and merely reproduced its
training base rate would score a marginal ratio of r0 = train base rate / test base rate,
so the fraction of the shift captured looks like (log m - log r0) / (0 - log r0).

It breaks on the control arm, and the way it breaks is the finding. The control's test
book has a *lower* base rate than its training book (0.0691% against 0.0862%, a shift of
0.80x), so a model carrying its training level across would OVER-predict. Instead it
under-predicts, at 0.6449. The error is not in the same direction as the base-rate shift
at all, which sends the formula to 303% and reveals its premise to be false: this
miscalibration is not a level error inherited from the training period. It is driven by
the covariates. The 2003-2004 cohorts look unusually safe at origination -- the refinancing
wave brought high scores and low loan-to-value -- so the model predicts *below* even its
training rate, and they then default more than their origination profile implies because
they met the crash in mid-life. Origination covariates cannot represent a calendar effect,
which is the age-period-cohort problem of PROJECT_PLAN.md section 6a arriving again by
another route.

    python scripts/compare_split_arms.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ARMS = {
    "crisis": (Path("reports"), "2000-04 -> 2006-08"),
    "control": (Path("reports/control"), "2000-02 -> 2003-04"),
}


def _row(scores: pd.DataFrame, population: str, variant: str) -> pd.Series | None:
    hit = scores[(scores["population"] == population) & (scores["variant"] == variant)]
    return hit.iloc[0] if len(hit) else None


def _arm(directory: Path) -> dict[str, str]:
    scores = pd.read_csv(directory / "calibration_scores.csv")
    in_sample = _row(scores, "train (in-sample)", "raw")
    holdout = _row(scores, "holdout (in-time)", "raw")
    test = _row(scores, "test (out-of-time)", "raw")
    assert in_sample is not None and holdout is not None and test is not None

    out = {
        "train base rate": f"{in_sample['base_rate']:.5%}",
        "test base rate": f"{test['base_rate']:.5%}",
        "  shift, test / train": f"{test['base_rate'] / in_sample['base_rate']:.2f}x",
        "in-sample ratio (control)": f"{in_sample['marginal_ratio']:.4f}",
        "in-time holdout ratio": f"{holdout['marginal_ratio']:.4f}",
        "OUT-OF-TIME ratio": f"{test['marginal_ratio']:.4f}",
        "  shortfall from 1.0": f"{1.0 - test['marginal_ratio']:.1%}",
        "out-of-time ECE / base rate": f"{test['ece'] / test['base_rate']:.1%}",
    }

    platt = _row(scores, "test (out-of-time)", "platt")
    if platt is not None:
        gap = abs(test["marginal_ratio"] - 1.0)
        out["in-time platt closes"] = f"{(gap - abs(platt['marginal_ratio'] - 1.0)) / gap:.1%}"

    sequential = [s for _, s in scores.iterrows() if "sequential" in str(s["population"])]
    raw_later = next((s for s in sequential if s["variant"] == "raw"), None)
    fitted = next((s for s in sequential if s["variant"] != "raw"), None)
    if raw_later is not None and fitted is not None:
        gap = abs(raw_later["marginal_ratio"] - 1.0)
        out["sequential closes"] = f"{(gap - abs(fitted['marginal_ratio'] - 1.0)) / gap:.1%}"

    horizons = directory / "calibration_by_horizon.csv"
    if horizons.is_file():
        table = pd.read_csv(horizons).set_index("horizon_months")["ratio"]
        for months in (12, 24, 36, 60):
            if months in table.index:
                out[f"cumulative ratio, {months}m"] = f"{table.loc[months]:.3f}"

    npv_path = directory / "calibration_npv_impact.csv"
    if npv_path.is_file():
        npv = pd.read_csv(npv_path)
        seq = npv[npv["scope"] == "sequential"].set_index("variant")
        if "sequential" in seq.index:
            raw_npv, corrected = seq.loc["raw"], seq.loc["sequential"]
            out["NPV/loan, raw"] = f"${raw_npv['npv_per_loan']:,.0f}"
            out["NPV/loan, sequential"] = f"${corrected['npv_per_loan']:,.0f}"
            out["  effect, $ per loan"] = f"${corrected['delta_vs_raw']:,.0f}"
            move = (corrected["npv_per_dollar"] - raw_npv["npv_per_dollar"]) * 100.0
            out["  effect, pp of principal"] = f"{move:+.4f}pp"
        if "platt" in seq.index:
            move = (seq.loc["platt", "npv_per_dollar"] - seq.loc["raw", "npv_per_dollar"]) * 100.0
            out["  in-time platt, pp"] = f"{move:+.4f}pp"
    return out


def main() -> int:
    missing = [n for n, (d, _) in ARMS.items() if not (d / "calibration_scores.csv").is_file()]
    if missing:
        print("missing arm(s): " + ", ".join(missing))
        print("  cde hazard --holdout-share && cde calibrate")
        print("  cde --config configs/calm_control.yaml hazard --holdout-share \\")
        print("      --reports-dir reports/control --figures-dir reports/control/figures")
        print("  cde --config configs/calm_control.yaml calibrate \\")
        print("      --reports-dir reports/control --figures-dir reports/control/figures")
        return 2

    columns = {name: _arm(directory) for name, (directory, _) in ARMS.items()}
    width = 20
    print(f"{'':<30}" + "".join(f"{name:>{width}}" for name in columns))
    print(f"{'':<30}" + "".join(f"{label:>{width}}" for _, label in ARMS.values()))
    print(f"{'':<30}" + "".join(f"{'-' * 18:>{width}}" for _ in columns))
    for key in next(iter(columns.values())):
        print(
            f"{key:<30}"
            + "".join(f"{col.get(key, '—'):>{width}}" for col in columns.values())
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
