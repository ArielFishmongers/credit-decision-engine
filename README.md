# Credit Decision Engine

This credit decision engine predicts *when* borrowers default with a
discrete-time competing-risks hazard model, calibrates those probabilities well enough to price a
loan with, and wraps them in an approval rule that maximises expected profit rather than accuracy. It is not a default classifier — the target is a decision, and the probability of default is an input to
it rather than the output of the exercise.

## Overview

A lender approving a loan is deciding whether the expected discounted cash flow from an applicant
exceeds the capital it ties up. Most public credit-risk work stops well short of that, instead fitting a classifier to a binary "did this default" target, then reporting AUC, and finally picking a threshold near where accuracy peaks. Three things are wrong with that, and this project is built to measure each one.

1. **Timing matters as much as incidence.** A binary target discards the timing, and with it most of the economic content. A loan defaulting in month 2 destroys far more value
   than one defaulting in month 30, because in the second case 28 payments have been collected.
2. **Level matters more than ranking.** An NPV calculation consumes the probability of default, not
   the ranking, whereas AUC only sees ranking, meaning an excellent AUC with systematic overconfidence still misprices every loan in the book.
3. **The optimal cutoff is an economic quantity, not a statistical one.** A threshold based on accuracy or F1 is built on the ratio of right and wrong predictions, not taking into account the possible losses or gains of each loan. The optimal cutoff should be calculated per loan to maximise expected profit.

## Data

The dataset used: Freddie Mac Single-Family Loan-Level Dataset, Release 47 (July 2026) — ~49.2M originations and
~2.9Bn monthly performance records, January 1999 to March 2026. This project uses the
50,000-loans-per-vintage random samples: **2000–2004 to train, 2006–2008 to test**, an
out-of-time split across the financial crisis, with 2005 omitted as the ambiguous transition
year. That is 400,000 loans and 19.20M person-period rows after a 120-month administrative
censor.

**The data is not in this repo and never will be.** It is licensed for use, not redistribution.
Access is free but requires a Clarity Data Intelligence account, CRT portal (not MBS):
<https://claritydownload.fmapps.freddiemac.com/CRT/#/sflld> → Data Download → SFLLD. Take
`sample_YYYY.zip` for each vintage into `data/raw/`. The one committed data file is the FRED
Treasury series behind the discount rate, which is US federal government work in the public
domain.

## Pipeline Structure

| Stage | What it does | Code |
|---|---|---|
| 1 | Parse the pipe-delimited files against an explicit Release 47 schema into DuckDB; build the person-period table (one row per loan per month at risk); compute vintage curves in SQL | `data/`, `features/` |
| 2 | Fit cause-specific discrete-time hazards — logistic regression on person-period rows, loan age through a spline basis, default and prepayment as competing causes | `models/` |
| 3 | Measure calibration (reliability, Brier, ECE) in-sample, on an in-time holdout and out of time; recalibrate with Platt and isotonic; price the difference | `calibration/` |
| 4 | Project loan NPV under the fitted hazards, rebuild realised cash from the observed balance path, and sweep the approval cutoff | `economics/` |
| 5 | The B0–B4 ablation on one out-of-time population, plus failure analysis by cohort and by risk decile | `eval/` |

## Layout

```
PROJECT_PLAN.md          the spec — framing, data decisions, stages, measured results
configs/
  default.yaml           every economic and modelling assumption
  calm_control.yaml      the same experiment without the crisis in the comparison
scripts/                 reference-doc and Treasury fetchers; the split-arm comparison
src/cde/
  config.py              typed, validated access to the YAML — the only reader of it
  data/                  Release 47 schema (generated), text loader, DuckDB ingest
  features/              person-period construction; sql/person_period.sql
  models/                cause-specific hazard models, the binary comparison model
  calibration/           Platt, isotonic, diagnostics
  economics/             NPV, realised cash, LGD, rates, the decision layer
  eval/                  per-stage evaluation harnesses and figures
  cli.py                 one subcommand per stage
tests/                   incl. a synthetic fixture with outcomes known by construction
reports/                 derived CSVs and figures (regenerable from the CLI)
docs/                    fetched Freddie Mac reference docs (gitignored) and project notes
notebooks/               exploration only — never the deliverable
```

Analytical SQL lives in `.sql` files rather than in Python strings, so it reads as SQL. Only the
type-cast layer is generated, from the schema's own column map.

## Results

All figures below are out-of-time. Section references are to `PROJECT_PLAN.md`.

### Calibration holds in time and breaks out of time (§6c)

The table reports predicted defaults ÷ observed defaults. A ratio below 1 means the model
predicts fewer defaults than actually occur; at 0.3717 out of time it predicts about a third of
them.

| Population | Rows | Default events | Predicted ÷ observed |
|---|---|---|---|
| train, in-sample (the fitted 80%) | 9,691,658 | 7,416 | 0.9996 |
| holdout, in-time (the unseen 20%) | 2,427,055 | 1,834 | 1.0212 |
| test, out-of-time (2006–2008) | 7,082,483 | 18,978 | **0.3717** |

The in-time holdout at 1.0212 says overfitting accounts for about two percentage points and
essentially nothing else, so the 0.3717 is a regime change rather than a generalisation failure.
The *ranking* survives it intact — observed default rate rises monotonically across all twenty
predicted-risk bins — while the *level* fails as an arch: calibrated at the safe end, worst in the
middle, recovering somewhat at the risky end. Grouping the out-of-time book into twenty
equal-count bins by predicted risk:

| Predicted-risk bin | Observed ÷ predicted |
|---|---|
| safest | 0.97× |
| middle | 4.10× |
| riskiest | 1.38× |

The middle bins are off by more than 4×, and the middle is exactly where the approve-or-decline
population sits — so the error is largest precisely where the cutoff has to be placed.

### The cutoff is worth more than the model (§6d)

Every rule below ranks applicants by the same predicted 36-month default probability from the
same fitted hazard, so only the criterion turning that score into a cutoff differs. Scored on
realised cash. The *oracle* is the best cutoff choosable knowing the result of each loan, not a number the lender can produce, but the ceiling that makes the other gaps interpretable.

| Rule | Approves | Realised NPV per applicant |
|---|---|---|
| approve all | 100.0% | −\$1,034 |
| accuracy-maximising | 100.0% | −\$1,034 |
| F1-maximising (weighs misses and false alarms equally) | 98.0% | −\$824 |
| expected NPV > 0 | 77.5% | **+\$393** |
| oracle (hindsight) | 51.5% | +\$784 |

The accuracy-maximising cutoff is the same as the approve all rule, and by arithmetic rather than
accident. At a 36-month default rate of 6.4% the true negatives dominate, so always approving will maximise accuracy due to this huge imbalance of defaulted loans.

Approving on positive
expected NPV instead earns **+\$1,427 per applicant** relative to the approve all and accuracy-maximising rules.

Calibration pays through where the cutoff lands, not through the portfolio total. Reading the probabilities at face value, the raw model approves 85% of the loans when about 63% was optimal, under-predicting default, so too many loans clear the "NPV > 0" bar. Calibrating the level shrinks those inflated NPVs, moving approval down to 71.5%. That single move is worth a further +\$277 per applicant.

Capture is defined as (what your rule earned) / (what the oracle earned). The raw rule captured 71.9% of the oracle's value, whereas post-calibration evaluation saw this rise to 95.5%.

### The ablation (§6e)

An **ablation** stacks models so each one differs from the one above it by a single design choice, then scores them all identically so that any change in profit can be attributed to the singular choice and nothing else. This is how the project decomposes where the value actually comes from.

Every arm runs on the same 87,854 out-of-time applicants (the 2007–2008 loans) under the same economics, and the binary comparison model is handed the identical training loans, covariates, encoding and optimiser settings as the hazard. 85 features against 92, the missing 7 being exactly the loan-age spline columns the binary model has no time axis to use.

Two columns are the most important: *Step* is what each arm adds over the one directly above it. *Its own forecast* is what that arm expected
to earn, shown to contrast with the value it realised.

| arm | approves | realised | step | its own forecast | AUC |
|---|---|---|---|---|---|
| B0 approve all | 100.0% | −\$79 | — | — | 0.500 |
| B1 binary, accuracy-max | 100.0% | −\$79 | \$0 | \$4,405 | 0.741 |
| B2 binary, profit-max | 89.2% | +\$460 | **+\$538** | \$4,541 | 0.741 |
| B3 hazard, profit-max | 84.5% | +\$915 | **+\$455** | \$3,640 | 0.806 |
| B4a + in-time recalibration | 84.7% | +\$896 | −\$18 | \$3,652 | 0.806 |
| B4b + sequential recalibration | 71.3% | +\$1,460 | +\$563 | \$2,660 | 0.803 |

Reading the table we see that B0 is the floor: approve everyone, no model, at −\$79. As seen above, B1 has no effect on the realised value and B2 increases the realised NPV due to maximising profit over accuracy. This verifies Claim 3 made in the overview section. B3 then changes only the model, swapping the binary target for the discrete-time hazard. This adds a further +\$455, additionally increasing AUC from 0.741 → 0.806. Knowing when a loan defaults improves the ranking as well as the pricing, which is more than Claim 1 strictly promised. B4a then recalibrates on held-back training data — the arm a lender could actually run — losing \$18, doing nothing as stage 3 predicted, because in-time the model is already right and the recalibrator has nothing to learn. B4b recalibrates instead on the first test cohort's realised outcomes and is worth +\$563, but that cohort takes 120 months to observe, so it is a ceiling on what recalibration could be worth, showing the possible maximised outcome.

The forecast (how much the model predicts to make) falls from \$4,541→\$2,660 while the realised results (how much the model actually made) rise from \$460→\$1,460. This is because as the model becomes better it becomes less overconfident, meaning that its forecast approaches the realised result which increases itself due to the model improving.

> (Approve-all reads −\$79 here against −\$1,034 in §6d because this table drops the 2006 cohort,
> which is spent fitting B4b's calibrator; every arm is scored on the same 2007–2008 set.)

### Where the model fails

We see that the model fails to predict the NPV of loans written in 2007 but succeeds in loans written in 2008: −\$3,274 realised against +\$2,854, on expectations
of \$733 and \$5,838. This is because a 2007 origination met the financial crash about eighteen months in, near the hazard's
peak with almost no principal amortised whereas a 2008 origination was written after prices had begun
falling, under tighter underwriting. The difference between the loans is not their origination data but the timing of them meeting the financial crisis.

By comparing risk deciles, we see that the error in prediction is a *shape* error rather than a *level* error. The expected vs realised
gap varies elevenfold from safest to riskiest loan, which is why a two-parameter monotone
recalibration cannot fix it. Additionally, the model's expected NPV is positive in every decile while
realised value crosses zero between the fifth and sixth, meaning a lender running it would see every segment profitable.

Underneath all of it is one structural limit. You cannot tell a loan's age apart from the
calendar date and the cohort it came from, because any two of the three fix the third exactly
(`period = cohort + age`). This is the age–period–cohort identification problem, and no amount of
data or spline flexibility escapes it. Origination covariates simply cannot represent a macro
effect that is defined by the calendar. The same limit is behind the 2004 cohort's residuals in
stage 2, the control arm under-predicting a book that looked *safer* at origination, and the
2007-versus-2008 gap above.

## Limitations

Design constraints rather than retrofitted analysis; `PROJECT_PLAN.md` §8 carries all thirteen
with their measurements.

- **The largest NPV magnitudes are from a crisis:** Measuring across the 2008 crisis gives an effect 11 times larger than over a calm period. The mechanism, however, generalises but the magnitude does not.
- **Selection bias cannot be corrected here:** The population is made up of only approved loans, loans that passed an
  originator's underwriting *and* Freddie Mac's purchase criteria. This pushes estimates optimistic. The Standard dataset also excludes the worst-rated mortgages — Alt-A, no-doc and
  option ARMs, so the crisis degradation measured here is a lower bound on what the market suffered.
- **Secured, not unsecured; US, not UK.** Mortgage LGD is collateral-driven, unsecured consumer
  LGD is near-total. The regulatory and macro regimes differ meaning that the techniques used in this model can be carried over to different types of loans, but the numbers cannot.
- **Default is first passage to 90+ DPD, which is not a loss event.** 63% of these loans recover and do not
  produce a loss disposition. The model handles this through an effective LGD, but the NPV cannot represent *timing*. This understates a loan's value by
  about 4.7% of its outstanding balance. A multi-state model is the correct treatment
  and the most valuable extension available.
- **The terminal balance is booked at par** at the 120-month horizon, where 16.3% of principal has been repaid, because projecting hazards past the model's support would be inventing data.
  It cancels in the differences the results rest on, so there is no absolute NPV valuation.
- **Censoring at 120 months** captures 92.5% of eventual loss dispositions and misses 7.5%,
  leaving realised NPV optimistic by roughly \$350 a loan.
- **Oracle rows are hindsight**, included only to bound what a ranking could have earned with a
  perfectly placed cutoff.

Out of scope, with reasons in §11: deployment or serving, deep learning, gradient boosting as the
headline model, fair-lending analysis, and risk-based pricing with adverse selection.
## Installation

```bash
pip install -e ".[dev]"                 # requires Python 3.11+
scripts/fetch_reference_docs.sh         # public Freddie Mac documentation
scripts/fetch_benchmark_rates.sh        # Treasury yields for the discount rate
pytest
```

## Usage

One command per stage, once the samples are in `data/raw/`:

```bash
cde schema            # the Release 47 column layout
cde config            # every resolved assumption, and the discount-rate bracket check
cde ingest            # parse the pipe-delimited files into DuckDB  [--check-casts]
cde person-period     # build one row per loan per month at risk
cde vintage           # cumulative default curves by cohort, in SQL, plus figures
cde hazard            # fit the cause-specific hazard models       [--knot-study]
cde hazard --holdout-share      # refit on 80% of training loans, holding the rest back
cde lgd               # loss given default, the cure rate, and what the cure blend costs
cde calibrate         # calibration on three populations, and the NPV impact
cde decide            # realised profit by approval cutoff, and the profit-maximising rule
cde ablate            # the B0-B4 ablation and the failure analysis
```

`cde calibrate` needs the holdout model rather than stage 2's, and refuses to guess: a
recalibrator fitted on the rows the hazard was fitted on comes back as the identity, because a
converged logistic regression is calibrated on its own training data by construction. Run
`cde hazard --holdout-share` first.

The control arm re-runs the experiment with the crisis taken out of the comparison:

```bash
cde --config configs/calm_control.yaml hazard --holdout-share \
    --reports-dir reports/control --figures-dir reports/control/figures
cde --config configs/calm_control.yaml calibrate \
    --reports-dir reports/control --figures-dir reports/control/figures
python scripts/compare_split_arms.py       # the two arms side by side
```

Outputs land in `reports/` as a CSV per measured quantity, with figures in `reports/figures/`.

