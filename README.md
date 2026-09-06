# Credit Decision Engine

Predicting *when* consumer borrowers default, calibrating those probabilities well enough to
price a loan with, and choosing the approval threshold that maximises expected profit rather
than accuracy.

**Status: in progress.** Stage 1 of 6. Nothing here is measured yet, and this README will report
results honestly — including where the model fails — when there are any.

## The argument

A lender approving a loan is not solving a classification problem. It is deciding whether the
expected discounted cash flow from an applicant exceeds the capital it ties up. The probability
of default is an *input* to that calculation, not the output of the exercise.

Most public credit-risk work stops at the input: fit a classifier to a binary "did this default"
target, report AUC, pick a threshold near where accuracy peaks. Three things are wrong with that,
and this project measures each.

1. **Timing matters as much as incidence.** A loan defaulting in month 2 destroys far more value
   than one defaulting in month 30, because in the second case 28 payments have been collected.
   A binary target discards the timing, and with it most of the economic content.
2. **Level matters more than ranking.** An NPV calculation consumes the actual probability, not
   the ordering. A model with excellent AUC that is systematically overconfident will misprice
   every loan in the book while looking excellent on the reported metric.
3. **The optimal cutoff is an economic quantity, not a statistical one.** It sits where marginal
   expected profit reaches zero — almost never where accuracy or F1 peaks.

## Approach

| Stage | What |
|---|---|
| 1 | Person-period dataset; vintage curves in SQL; the age–period–cohort identification problem |
| 2 | Discrete-time hazard model — logistic regression on person-period data, hazard by month on book |
| 3 | Calibration: reliability curves, Brier, ECE; Platt and isotonic recalibration |
| 4 | Loan NPV under a default hazard; expected profit by approval threshold; profit-maximising cutoff |
| 5 | Out-of-time validation and the B0–B4 ablation |
| 6 | Write-up, including what does not work |

The ablation is the point. Five models — approve-everyone, the conventional binary classifier at
an accuracy-maximising cutoff, the same at a profit-maximising cutoff, an uncalibrated hazard
model, and a calibrated one — reported as expected NPV per applicant on out-of-time data. That
decomposes how much of the value comes from the survival framing, how much from calibration, and
how much from the decision layer alone.

## Data

Freddie Mac Single-Family Loan-Level Dataset, Release 47 (July 2026): ~49.2M originations and
~2.9Bn monthly performance records, January 1999 – March 2026.

Chosen over Lending Club deliberately. The performance file **is** a person-period dataset — one
row per loan per month, with `LOAN AGE` as a column — so censoring, competing risks (voluntary
payoff versus default) and exposure at default are all observed rather than reconstructed. It
also carries realised loss components, so loss given default can be estimated rather than assumed.

The trade is domain distance: these are US mortgages, and the target application is unsecured
consumer lending. The technique transfers; the parameters do not. That mapping is stated
explicitly rather than glossed.

**The data is not in this repo and never will be** — it is licensed for use, not redistribution.
See `CLAUDE.md` for access, and `PROJECT_PLAN.md` §4 for the full data rationale.

## Getting started

```bash
pip install -e ".[dev]"
scripts/fetch_reference_docs.sh   # public Freddie Mac documentation
cde schema                        # the Release 47 column layout
pytest
```

The dataset itself requires a free Clarity Data Intelligence account (CRT portal → Data Download
→ SFLLD). Start with `sample_YYYY.zip`, not the full quarters.

## Known limitations

Written up front, as design constraints rather than retrofitted excuses.

- **Selection bias cannot be corrected here.** The population is loans that passed an
  originator's underwriting *and* Freddie Mac's purchase criteria. There are no rejected
  applications at all, which pushes estimates optimistic.
- **Secured, not unsecured.** Mortgage loss given default is collateral-driven; unsecured
  consumer LGD is near-total.
- **US, not UK** — different macro regime, different regulatory framework.
- **Censored at 60 months on book** by design, which discards long-horizon behaviour.

See `PROJECT_PLAN.md` for the full specification.
