# Credit Decision Engine — Project Plan

**Status:** living spec, v0.6, 9 September 2026. **All six stages are complete and measured**;
§6a-§6e carry the results and stage 6 is the README. Decisions are recorded here as they are
taken, with the evidence that settled them — several early assumptions have been overturned by
measurement and are marked as such rather than quietly edited.

**Author:** Ariel Fishgang

---

## 1. Problem framing

A lender approving a consumer loan is not solving a classification problem. It is deciding
whether the expected discounted cash flow from an applicant exceeds the capital it ties up. The
probability of default is an *input* to that calculation, not the output of the exercise.

Almost every public credit-risk project stops at the input. It fits a classifier to a binary
"did this loan default" target, reports AUC, picks a threshold near where accuracy or F1 peaks,
and stops. Three things are wrong with that, and this project exists to demonstrate each:

**Claim 1 — timing matters as much as incidence.** A loan that defaults in month 2 destroys far
more value than one that defaults in month 30, because in the second case you have already
collected 28 payments. A binary target throws the timing away entirely, and with it most of the
economic content of the prediction.

**Claim 2 — level matters more than ranking.** A net present value calculation consumes the
actual probability, not the ordering. A model with excellent AUC that is systematically
overconfident will misprice every loan in the book while looking excellent on the metric that
gets reported.

**Claim 3 — the optimal cutoff is an economic quantity, not a statistical one.** It sits where
marginal expected profit reaches zero. That is almost never where accuracy, F1 or Youden's J
peaks, and the gap between them is money.

The deliverable is a system that predicts *when* borrowers default, calibrated well enough to
price with, wrapped in a decision rule that maximises expected profit.

**If a component does not serve one of the three claims, it does not belong in v1.**

## 2. Why this project

Portfolio artefact and interview preparation for UK Summer 2027 quantitative and data roles.
It was prompted by a posting that described the work as *"using data and modelling to ground
assumptions for our NPV model, such as predicting what default rates will be in the future, and
using that to inform our lending decisions"* — which is a fair statement of the problem this
project solves — but **it is deliberately not scoped to any one employer or product.**

The design objective is to **model correctly on the best data available**, not to mimic a
particular lender's book. Where a modelling choice could be made either for fidelity to some
target product or for correctness on the data in hand, correctness wins and the decision is
recorded here. The 120-month horizon in §4.3 is the clearest example: an earlier version
censored at 60 months to bracket unsecured consumer-loan tenor, which discarded half the
observable losses. That trade is no longer made.

The real success criterion is not that the code runs. It is that every modelling decision can
be defended out loud, with the measurement that justified it.

## 3. Prior art and where this differs

Consumer credit scoring is a mature field, and none of the individual techniques here are novel.
The contribution of this project is entirely in the *combination* and the rigour: survival
framing, calibration, an explicit economic decision layer, and honest out-of-time failure
analysis, in one pipeline, on public data.

The reading list is stage 0's deliverable and is not yet complete. Sources I expect to be
load-bearing, all to be verified before relying on them:

- Survival analysis generally — Kleinbaum & Klein, *Survival Analysis: A Self-Learning Text*
- Discrete-time hazard specifically — Singer & Willett, *Applied Longitudinal Data Analysis*
- Credit risk in practice, with code — Baesens, Rösch & Scheule, *Credit Risk Analytics*
- Credit scoring theory — Thomas, Edelman & Crook, *Credit Scoring and Its Applications*
- Calibration — Niculescu-Mizil & Caruana, "Predicting Good Probabilities With Supervised
  Learning", ICML 2005
- Reject inference — Crook & Banasik, on whether it actually improves scorecards
- Basel IRB definitions of PD / LGD / EAD — BIS primary documents

**Known risk:** these are recalled, not verified. Confirm each exists, in which edition, and
which chapters are relevant, before quoting any of it in a write-up.

## 4. Data

### 4.1 Primary dataset — Freddie Mac Single-Family Loan-Level Dataset

Chosen over Lending Club deliberately. The decisive property is that the performance file **is**
a person-period dataset: one row per loan per month, with `LOAN AGE` as a column. Lending Club
gives a loan-level snapshot from which time-to-default must be reconstructed via `issue_d` and
`last_pymnt_d`, which carries a systematic offset because charge-off occurs months after the
final payment. That would mean defending a workaround across three of five stages.

| Property | Provided by |
|---|---|
| Join key | `LOAN IDENTIFIER`, present in both files |
| Survival time index | `LOAN AGE` (months on book), with `PERIOD` as the calendar month |
| Event type | `ZERO BALANCE CODE`: `01` prepaid/matured, `03` short sale or charge-off, `09` REO disposition |
| Event date | `ZERO BALANCE EFFECTIVE DATE` |
| Censoring | absence of a zero-balance code by the performance cutoff |
| EAD | `CURRENT ACTUAL UPB`, observed monthly |
| LGD | `ACTUAL LOSS` and its components (`NET SALES PROCEEDS`, `MI RECOVERIES`, `NON MI RECOVERIES`, `TOTAL EXPENSES`, `DELINQUENT ACCRUED INTEREST`) |
| Origination features | `CLASSIC FICO`, `VANTAGESCORE 4.0` (new in Release 47), DTI, LTV/CLTV, original UPB, original rate, original term, channel, loan purpose, occupancy, property type/state, first-time-buyer indicator |

Release 47 (29 July 2026): 109 quarters, originations and performance through 31 March 2026,
approximately 49.2M origination records and 2.91Bn performance records in the Standard dataset.

**Access:** free for non-commercial, academic and research use, via registration on Freddie Mac's
Clarity Data Intelligence portal and acceptance of its Terms of Use. Registration is the licence
acceptance. **Consequence for this repo: the data is never committed.** Ship a download script
and a `.gitignore` entry.

| Resource | URL | Login? |
|---|---|---|
| SFLLD download (register / sign in here) | `https://claritydownload.fmapps.freddiemac.com/CRT/#/sflld` | yes |
| Clarity hub | `https://capitalmarkets.freddiemac.com/clarity` | yes |
| Dataset resource page | `https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset` | no |
| Release 47 column headers | `.../fmac-resources/research/docs/file_headers_july_2026.zip` | no |
| Release 47 file layout | `.../fmac-resources/research/pdf/file_layout_july_2026.xlsx` | no |
| User guide (July 2026) | `.../fmac-resources/research/pdf/general_user_guide_july_2026.pdf` | no |
| Licensing agreement | `.../fmac-resources/research/pdf/dataset_licensing_agreement.pdf` | no |
| Format example (1,000 rows) | `.../fmac-resources/research/docs/release-47-sample-files.zip` | no |
| Support / access requests | clarity@freddiemac.com | — |

Paths marked `...` are relative to `https://www.freddiemac.com`. The legacy portal at
`freddiemac.embs.com/FLoan` is retired and now serves only a redirect notice.

**Documentation and the format example need no account** — only the data itself is behind
Clarity. Schema work can therefore start before registration clears.

**Which licence applies.** `dataset_licensing_agreement.pdf` on the resource page is the
*fee-based commercial* agreement (a License Fee of $27,562.50, covering redistribution, sub-
licensing and derived products sold to end users). It does not apply here. Non-commercial and
internal use is licensed separately and royalty-free, and those are the Terms of Use accepted at
registration. This project is non-commercial academic work and sits under the royalty-free terms,
which is also why the data must never be redistributed from the repo.

**Files:** pipe-delimited `.txt`, no header row. Release 47 renamed the files as well as the
columns: standard quarters are `historical_data_YYYYQn.zip` containing `orig_YYYYQn.txt` and
`perf_YYYYQn.txt`; the vintage sample is `sample_YYYY.zip` containing `sample_orig_YYYY.txt` and
`sample_perf_YYYY.txt` (previously `sample_svcg_`). Release 47 layout is **31 origination columns
and 35 performance columns**, taken verbatim from `file_headers_july_2026.zip` — not from the
pre-July-2026 user guide, which uses superseded names (`CREDIT SCORE` is now `CLASSIC FICO`,
`LOAN SEQUENCE NUMBER` is now `LOAN IDENTIFIER`, `MONTHLY REPORTING PERIOD` is now `PERIOD`,
`NET SALE PROCEEDS` is now `NET SALES PROCEEDS`, `ACTUAL LOSS CALCULATION` is now `ACTUAL LOSS`).
Standard dataset only — the Non-Standard dataset has not been refreshed since Release 30.

### 4.2 Sample-first strategy

Freddie Mac publishes a simple random sample of 50,000 loans per full vintage year
(`sample_YYYY.zip`, containing `sample_orig_YYYY.txt` and `sample_perf_YYYY.txt`), downloadable from Clarity. **The entire pipeline
is built and validated on that sample.** Do not confuse it with `release-47-sample-files.zip` on
the public resource page, which is a 1,000-row format illustration only — useful for writing the
parser before an account exists, useless for modelling. It is a few million performance rows per vintage — trivial for DuckDB on a laptop —
and being a genuine random sample, hazard and calibration estimates from it are unbiased.

Scale to full quarters only if a specific analysis demands the tails. Note that the sample is
50,000 *per vintage year*, so each year is a different sampling fraction of a differently-sized
book; weight accordingly before making any cross-vintage population claim.

### 4.3 Design decision: observation horizon

Freddie Mac loans are 15- and 30-year mortgages; loan age runs to 360 months. A full 360-month
hazard is computationally heavy and most of its tail carries little information, so the panel is
administratively censored — but where.

**Decision: censor at 120 months on book.** Chosen from the data. Measured share of all eventual
loss dispositions falling inside the window:

| Horizon | Dispositions captured | Performance rows |
|---|---|---|
| 60m | 52.7% | 15.5M |
| 84m | 75.8% | 18.2M |
| **120m** | **92.5%** | **20.5M** |
| 180m | 98.9% | 22.1M |

**Why not 60 months, as this document originally specified.** It captured barely half the losses,
which made the default population incomplete — and incomplete by a *different amount per cohort*
(24% for the 2003 vintage against 70% for 2008, because foreclosure timelines moved enormously
over the period). A label whose completeness varies systematically with the cohort injects a
spurious cohort effect into the out-of-time comparison the whole ablation depends on. The
original rationale was to bracket unsecured consumer instalment tenor of 1-7 years; §2 no longer
makes that trade.

**Why not 180 months.** Diminishing returns — 6.4 further percentage points of coverage for
another 1.6M rows — and every vintage here is still fully mature at 120 months: 2008 + 120 is
2018 against a March 2026 cutoff, so no censoring is introduced.

The horizon also fixes the discount-rate benchmark. Ten years of cash flows are discounted on the
10-year Treasury (`DGS10`), not the 5-year; the two must move together or the term structure is
mispriced.

### 4.4 Secondary dataset — Lending Club, as a transfer test (stretch)

If time allows, refit the same pipeline on Lending Club data (2007–2018Q4) and report where it
breaks. This is the out-of-domain move, it directly answers the "but this is mortgages"
objection, and the rejected-loans file makes the selection-bias discussion concrete rather than
hypothetical.

**This is stretch scope, not v1.** LendingClub no longer exists as a brand — the corporation
became Happen, Inc. in June 2026 and the original download pages are gone — so archive a mirror
now while mirrors remain up, then set it aside.

## 5. Method

### Stage 0 — Theory *(before any code)*

Learn what is genuinely new: credit-risk vocabulary (PD/LGD/EAD), loan unit economics, survival
analysis with an emphasis on discrete-time hazard models, calibration versus discrimination,
vintage and age-period-cohort analysis, reject inference, cost-sensitive decision theory.

**Deliverable:** the ability to derive the NPV identity in §5.4 unaided. A written reading list
was specified here originally and has been **dropped** (decided 7 September 2026): the theory is
done, and maintaining a bibliography as a separate artefact bought nothing the write-up needs.
Sources are cited inline in stage 6 where they are actually load-bearing.

### Stage 1 — Data and vintage analysis

Download script against Clarity; parse the pipe-delimited files with an explicit Release 47
column schema; load to DuckDB; construct the person-period table (one row per loan per month at
risk, up to the 120-month administrative censor).

Then the SQL piece: **cumulative default rate by loan age, one curve per origination cohort**,
computed with window functions. This is the standard consumer-credit vintage chart and it is
naturally a SQL exercise. It also exists to demonstrate SQL publicly, since current SQL evidence
is Athena/Trino at work with nothing shareable.

Write up the age–period–cohort identification problem honestly: maturation, vintage quality and
calendar effects are structurally non-identifiable without assumptions, because age + cohort
determines period exactly. Saying so is a stronger answer than a chart pretending otherwise.

### Stage 2 — Discrete-time hazard model

Fit logistic regression on the person-period dataset, with loan age entering flexibly (splines
or dummies rather than a linear term) and origination covariates as predictors. Output: a hazard
curve h(t) by month on book, per applicant.

Censoring is handled structurally — a loan simply stops contributing rows after its last observed
month. This is the property the dataset was chosen for.

**Competing risks: resolved, both causes are modelled.** `ZERO BALANCE CODE` distinguishes
voluntary payoff (`01`) from the credit-event codes (`02`, `03`, `09`). Prepayment is not a
nuisance — a loan repaid early stops paying interest, which is a real NPV term — and stage 1
measured it at 40-91% of every cohort, an order of magnitude larger than default. So default and
prepayment are fitted as separate cause-specific hazards, sharing one design matrix. See §10 Q4
for the measurement that forced it.

### Stage 3 — Calibration

Reliability curves, Brier score and expected calibration error, at fixed horizons (12, 24, 36
months) and integrated over the observation window. Recalibrate with Platt scaling and isotonic
regression; compare.

**The test that matters is not the calibration metric itself but its downstream effect:** quantify
how much the NPV of a portfolio changes when computed with raw versus recalibrated probabilities.
That number is the evidence for Claim 2, and asserting Claim 2 without it is hand-waving.

### Stage 4 — Economics and the decision layer

Derive the NPV of an instalment loan from first principles. With monthly index t, scheduled
payment M, outstanding balance B(t), default hazard h_d(t), prepayment hazard h_p(t), survival
S(t) = Π_{u<t} (1 − h_d(u) − h_p(u)), discount rate r and loss given default LGD:

```
NPV = −P + Σ_t  S(t) · [ (1 − h_d(t) − h_p(t))·M
                       + h_p(t)·B(t)
                       + h_d(t)·B(t)·(1 − LGD) ]  /  (1 + r)^t
```

Deriving this is what makes Claims 1 and 2 obvious rather than asserted: timing enters through
S(t) and the discount factor, level enters through h_d directly.

Then the decision: expected profit as a function of the approval threshold τ, and argmax over τ.
Extension if time allows: risk-based pricing, setting the rate as a function of PD rather than
using a single cutoff — and the twist that makes it real, that raising the rate changes who
accepts, which is adverse selection, so price and default rate are not independent.

### Stage 5 — Validation and ablation

**Out-of-time, never a random split.** Train on originations before a cutoff vintage, test on
originations after it. Random splits leak future information and flatter the model.

The ablation is the experimental core, because it decomposes where the value actually comes from:

| Model | Target | Threshold rule | Isolates |
|---|---|---|---|
| B0 | — | approve all | floor |
| B1 | binary default within 36m | accuracy-maximising | the typical public notebook |
| B2 | binary default within 36m | profit-maximising | value of the decision layer alone |
| B3 | discrete-time hazard, uncalibrated | profit-maximising | value of the survival framing |
| B4 | discrete-time hazard, recalibrated | profit-maximising | value of calibration |

Reporting NPV per applicant across B0–B4 on the out-of-time set is the single most
informative table in the project, and it is what an interviewer will actually interrogate.

**Corrected 8 September 2026: that NPV must be REALISED, not expected.** This section
originally specified "expected NPV per applicant", which is circular and would have
inverted the result. Stage 3 measured the size of the problem: on the 2007–2008 book the
uncalibrated hazard assesses portfolio value at $3,396 a loan and the honestly
recalibrated one at $2,046, because a model that under-predicts default believes every
loan is profitable. Scoring each arm on its own expected NPV therefore ranks B3 above B4
by $1,349 — it rewards precisely the overconfidence stage 3 exists to expose, and the
better-calibrated model loses by construction.

So the rule is: **the decision comes from predictions, the score comes from cash.** Each
arm chooses its cutoff using only information available at the decision point, and every
arm is then scored on the same realised cash flows from `src/cde/economics/realised.py`.
Expected NPV remains the decision input and is reported alongside, because the gap between
expected and realised is itself a result.

**Metrics:** discrimination as AUC and Gini (= 2·AUC − 1, the form credit teams quote); calibration
as reliability curve, Brier and ECE; survival-specific time-dependent AUC and integrated Brier
score; economics as **realised** NPV per applicant and approval rate at the chosen cutoff.

### Stage 6 — Write-up

A README that can be read in ten minutes and followed end to end, with a limitations section
written honestly rather than defensively.

**Delivered 9 September 2026.** `README.md` is a conventional project README — what the project
is, the data, how the pipeline works, install, usage, layout, then the measured results and the
limitations. It is deliberately *not* structured as an interview document; the full argument,
the negative results and the four stage-2 bugs stay here in §6a-§6e, where they belong, and the
README quotes the headline measurements and points at those sections.

Two things landed in the README rather than here. The age-period-cohort write-up deferred from
stage 1 appears as the closing paragraph of the failure analysis, since the same identification
limit is what connects the 2004 residuals, the control arm's backwards result and the
2007-versus-2008 gap. And §2's framing question is answered in the results: the ranking
transfers, the level does not, and the level is what a forecast is.

The reading list of §3 stays dropped, and no source is quoted in the README — §3's citations are
recalled rather than verified, so quoting them would breach the project's own standard.

## 6. Definition of done, per stage

| Stage | Done when |
|---|---|
| 0 | **Done.** NPV identity derivable unaided; reading-list deliverable dropped |
| 1 | **Done.** Person-period table built (19.20M rows, 399,904 loans, 28,228 events) and documented; vintage curves computed in SQL; six figures rendered; results in §6a. APC write-up deferred to stage 6 and delivered there |
| 2 | **Done.** Cause-specific hazard curves by month on book; censoring handled structurally and documented; competing risks resolved by measurement and both causes fitted; knot basis chosen by BIC; results in §6b |
| 3 | **Done.** Reliability curves, ECE and Murphy-decomposed Brier on three populations; Platt and isotonic compared; downstream NPV impact quantified at −39.7% per loan under a strictly monotone correction, which is claim 2 with discrimination held fixed. Discount rate (§10 Q1) and LGD (§10 Q2) resolved as inputs; results in §6c |
| 4 | **Done.** NPV derived; realised NPV per loan built and reconciled to $0.0000 on an exact identity; realised profit as a function of the approval cutoff; the economic rule beats the accuracy-maximising one by $1,427 per applicant on the same score, and the accuracy-maximising cutoff is provably the do-nothing rule. Results in §6d. Risk-based pricing and adverse selection out of scope — §11 |
| 5 | **Done.** B0–B4 ablation on one out-of-time population, every arm scored on realised cash: B1 (the typical notebook) adds $0, the decision layer +$538, the survival framing +$455, honest recalibration −$18, recalibration with out-of-time information +$563. Failure analysis names 2007 as the cohort that breaks it and shows the error is a shape error, not a level one. Results in §6e |
| 6 | **Done.** `README.md` covers what the project is, the data, the pipeline, install, usage, layout, then results (calibration, the decision layer, the B0-B4 ablation, the failure analysis and the control arm) and the limitations that bound them — including the stress-result label on the headline magnitude |

## 6a. Stage 1 measured results

Re-measured at the 120-month horizon, 8 September 2026. Freddie Mac Release 47 vintage
samples, 50,000 loans per vintage year: 400,000 loans and 22.65M performance records
ingested, **19.20M person-period rows** after the 120-month censor. Default is first
passage to 90+ DPD.

| Vintage | Split | Default by 120m | Prepaid by 120m | Still performing | 1−KM overstates by |
|---|---|---|---|---|---|
| 2000 | train | 3.22% | 94.39% | 2.39% | 6.57x |
| 2001 | train | 2.89% | 90.84% | 6.27% | 4.30x |
| 2002 | train | 2.85% | 86.32% | 10.83% | 3.16x |
| 2003 | train | 3.37% | 77.12% | 19.51% | 2.02x |
| 2004 | train | 6.16% | 77.69% | 16.14% | 2.02x |
| 2006 | test | 13.64% | 78.58% | 7.77% | 2.15x |
| 2007 | test | 15.66% | 77.21% | 7.14% | 1.99x |
| 2008 | test | 8.80% | 84.97% | 6.23% | 2.46x |

**The out-of-time split has something to test, though less than an earlier draft claimed.**
Test cohorts default at between **1.4x and 5.5x** the rate of train cohorts (worst test
15.66% against best train 2.85%; best test 8.80% against worst train 6.16%). Per
person-period row the gap is **0.0763% train against 0.2680% test, a factor of 3.5.**

Worth stating in the write-up: this understates the market, because the Standard dataset
is fixed-rate, full-documentation and non-government-programme only, so Alt-A, no-doc and
option ARMs are excluded before we see them.

### The horizon change corrected a number of our own making

An earlier version of this section reported the same table at a 60-month horizon and
claimed test cohorts defaulted at **"four to ten times"** the train rate. That was partly
an artefact of the censoring choice, not a property of the data:

| | 60-month horizon | 120-month horizon |
|---|---|---|
| 2003 default rate | 1.35% | 3.37% (2.5x higher) |
| 2007 default rate | 13.30% | 15.66% (1.2x higher) |
| Train/test ratio | ~4-10x | **1.4-5.5x** |
| Severity degradation | 2.35x | **1.81x** |

Calm-cohort loans default *later in life*, so a 60-month window truncated the comparison
group far harder than the crisis cohorts. We were measuring our own censor and reporting
it as a regime shift. See §4.3 for why 120 months was chosen instead.

The longer window also revealed structure the short one could not see: the prepayment
hazard has a **second hump at months 90-110**, the default hazard **plateaus and turns
over near month 105**, and **the 2008 cohort crosses below 2006 at about month 55** — so
the 60-month view ranked those two cohorts the wrong way round.

**A prediction that was wrong, recorded rather than quietly dropped.** Before the first
run the expectation written down was low-single-digit cumulative default for 2006-07. The
measured figures are 13.64% and 15.66%. The reason is the definition: 90+ DPD catches
every borrower who ever fell three payments behind, including the ~63% who cure or are
modified, which is a far broader population than those who reach charge-off. It is the
right target for an approve/decline decision and it is not the same quantity as "loans
that produced a loss".

**Three data pathologies found and handled**, all documented in
`src/cde/features/sql/person_period.sql`:

1. **The loan-age clock resets on modification** — 10,091 loans, 8,832 of them inside the
   120-month horizon. Sequencing by `LOAN AGE` value rather than calendar order made such
   a loan traverse the same ages twice, duplicating rows, firing the event indicator more
   than once, and manufacturing in-horizon defaults out of much later delinquencies. Fixed
   by truncating in `PERIOD` order at the first discontinuity; 706 loans censor as
   `clock_reset_censor`.
2. **Loans modified before Freddie Mac acquired them** enter at age 0 already delinquent,
   with `MODIFICATION FLAG` blank so the reset is invisible — 47 loans. Detected by two
   independent signals and excluded, switchable in config.
3. **144 loans missing a reporting month**, which breaks the risk set's contiguity.
   Censored at the gap.

**A claim retracted.** An earlier draft reported that `S(t) + F_d(t) + F_p(t) = 1` to
machine precision as evidence that the exit taxonomy is exhaustive. It is not evidence of
anything: it is an algebraic identity of the estimators and holds however wrong the exit
classification is. Exhaustiveness is checked where it can be — on `loan_exit`, by
requiring every loan to carry exactly one enumerated reason, and by reconciling against
the origination cohort:

```
400,000 origination loans
   −    49  never observed inside the horizon
   −    47  excluded as off-clock
   = 399,904  loans in person_period        (reconciles)
```

The counting estimator and Aalen–Johansen agree to within **0.00068** across all eight
cohorts, because every cohort is fully mature at 120 months — 2008 + 120 months is 2018
against a March 2026 cutoff. That agreement validates the arithmetic; it is not a licence
to use the counting estimator on an immature book.

## 6b. Stage 2 measured results

Cause-specific discrete-time hazard models, fitted on the 2000-2004 training vintages at
the 120-month horizon.

| Cause | Events | Rows | Base rate | lbfgs iterations | Marginal gap |
|---|---|---|---|---|---|
| Default | 9,250 | 12,118,713 | 0.0763% | — | +0.02% |
| Prepayment | 212,542 | 12,118,713 | 1.7539% | 157 | −0.01% |

92 features: a 7-term loan-age spline basis, 21 predictor spline terms, 3 linear terms, 6
missing-value indicators and 55 one-hot columns. `VANTAGESCORE 4.0`,
`PROPERTY VALUATION METHOD` and `SPECIAL ELIGIBILITY PROGRAM` are entirely null on these
vintages and are dropped automatically; `AMORTIZATION TYPE`, `INTEREST ONLY INDICATOR`,
`SUPER CONFORMING FLAG` and `HARP INDICATOR` carry a single value each — they are the
Standard dataset's own filter criteria — and are dropped for zero variance.

### Six knots, chosen by BIC rather than by taste

| Knots | Parameters | Log-likelihood | BIC (rows) | BIC (events) |
|---|---|---|---|---|
| **6** | 93 | −67,604.44 | **136,725.74** | **136,058.20** |
| 9 | 105 | −67,581.69 | 136,875.96 | 136,122.28 |
| 12 | 117 | −67,553.82 | 137,015.94 | 136,176.13 |
| 16 | 131 | −67,521.01 | 137,178.67 | 136,238.36 |

Going from 6 to 16 knots buys 83.4 log-likelihood points for 38 extra parameters, which
does not pay under either denominator. **This was re-run at 120 months and the answer did
not change** — a prediction I got wrong, having expected a doubled age range to want a
richer basis. Six knots captures both prepayment humps.

Two BIC columns are reported because it is genuinely unclear what *n* should be on
person-period data: there are 12.1M rows but only 9,250 events, and the effective
information is nearer the event count. Here they agree, so the ambiguity does not matter.

### The cohort residuals, and the limit they reveal

Mean standardised residual of the default hazard, by cohort:

| Cohort | All ages | Ages 90–120 |
|---|---|---|
| 2000 | −0.36 | −0.72 |
| 2001 | −0.30 | −0.74 |
| 2002 | −0.54 | −0.92 |
| 2003 | −0.35 | −0.06 |
| **2004** | **+0.98** | **+1.65** |

Four cohorts sit slightly negative; 2004 is systematically under-predicted. The reason is
calendar arithmetic — 2004's late ages are 2008 onward, and no other training cohort was
that old during the crisis. **This is the age–period–cohort problem inside the model**, and
no amount of loan-age flexibility fixes it: more knots would chase a calendar effect with
an age variable, and vintage dummies would fit the past and be unknowable for a future
cohort. The right response is to name the limit.

At the 60-month horizon this residual read **+3.29**. The signal weakened partly because
2004's crisis exposure is a smaller share of a 120-month life, so the finding stands but is
less stark than first reported.

### It agrees with credit intuition

Checked through predictions rather than coefficients, since splines make individual
coefficients uninterpretable. Re-measured at 120 months; all four monotone across every
band, in the expected direction.

| Predictor | Low band | High band | Spread | Direction |
|---|---|---|---|---|
| `CLASSIC FICO` → default | 0.335% (<620) | 0.014% (780+) | **24.1x** | falls |
| Combined LTV → default | 0.028% (<=60) | 0.173% (91-100) | 6.3x | rises |
| DTI → default | 0.037% (<=20) | 0.115% (44-65) | 3.1x | rises |
| Note rate → **prepayment** | 1.096% (<5.5%) | 4.705% (8.5%+) | 4.3x | rises |

The ordering — credit score dominating leverage, which dominates debt burden — is what a
credit team would expect, because FICO summarises years of actual repayment behaviour
while DTI is arithmetic at one moment on self-reported income.

The last row is the one worth leading with, because nothing in the model was built to
produce it: the refinancing incentive falls straight out of the data. A higher note rate
means more to gain from refinancing, so more early repayment. It also matters for stage 4 —
prepayment is ~23x more common than default here and strongly rate-driven, so the NPV will
be more sensitive to the prepayment hazard than the default one, which is counterintuitive
for a "credit" model.

### Four bugs, all of which produced plausible output and no warning

1. **sklearn's default `tol=1e-4`** stops the optimiser early on rare events, leaving
   predicted events ~1% off actual. A converged logistic fit with an unpenalised intercept
   satisfies `sum(y − p) = 0` exactly, so that gap was impossible-if-converged.
2. **Unscaled features** — `ORIGINAL LOAN TERM` at 60–360 against everything else in
   [0,1]. That spread stalled lbfgs on its relative-function tolerance and it *reported
   success*. Standardising cut the gap to under 0.02%.
3. **`max_iterations` defaulting to 0**, which would have flagged every fit as hitting the
   iteration cap. A warning that is always on is worse than none.
4. **`survival()` depending on row order** — `cumprod`, `shift` and `cumsum` walk rows
   physically and never consult `loan_age`, so a shuffled frame gave wrong intermediate
   values while the final survival looked right, multiplication being commutative.

All four are now guarded: the marginal gap and iteration count print on every fit, and
tests assert the feature scale is bounded, that shuffling changes nothing, and that the
buggy version *is* plausible-looking.

## 6c. Stage 3 measured results

Measured 8 September 2026. **The hazard used throughout stage 3 is fitted on 80% of the
training loans**, not on all of them — a recalibrator has to be fitted on rows the hazard
never saw, and stage 2's full-sample model has none. Reproduce with
`cde hazard --holdout-share` then `cde calibrate`. That fit converged on 9,691,658 rows
carrying 7,416 default events, marginal gaps −0.039% (default) and +0.009% (prepayment).

Three populations, and the first is a **control** rather than a result:

| Population | Rows | Default events | Base rate |
|---|---|---|---|
| train, in-sample (the fitted 80%) | 9,691,658 | 7,416 | 0.0765% |
| holdout, in-time (the unseen 20%) | 2,427,055 | 1,834 | 0.0756% |
| test, out-of-time (2006–2008) | 7,082,483 | 18,978 | 0.2680% |

### The headline: the model is calibrated in-time and badly wrong out of time

| Population | Predicted events | Observed | Ratio | ECE | ECE as % of base rate |
|---|---|---|---|---|---|
| train, in-sample | 7,413 | 7,416 | **0.9996** | 9.69e-05 | 12.7% |
| holdout, in-time | 1,873 | 1,834 | **1.0212** | 1.08e-04 | 14.3% |
| test, out-of-time | 7,055 | 18,978 | **0.3717** | 1.68e-03 | 62.9% |

The in-sample ratio of 0.9996 is not an achievement — a converged logistic regression with
an unpenalised intercept reproduces its own event count by construction, so anything else
would have meant the harness was wrong. It has earned its place: this identity caught two
bugs earlier in the project.

**The holdout is what makes the finding interpretable.** At 1.0212 the model over-predicts
by 2% on unseen loans *of the same vintages*, so overfitting accounts for about two
percentage points and essentially nothing else. The 0.3717 out of time is therefore not a
generalisation failure in the usual sense. It is a regime change: the model is a faithful
description of 2000–2004 and 2000–2004 was a different world.

### The error is an arch, worst in the middle of the risk distribution

Twenty equal-count bins of the out-of-time book, ~354,000 rows each:

| Bin | Predicted | Observed | obs/pred | z |
|---|---|---|---|---|
| 0 (safest) | 0.0020% | 0.0020% | 0.97x | −0.1 |
| 1 | 0.0049% | 0.0042% | 0.87x | −0.5 |
| 4 | 0.0141% | 0.0359% | 2.55x | 10.9 |
| 9 | 0.0402% | 0.1646% | **4.10x** | 37.0 |
| 12 | 0.0713% | 0.2906% | 4.07x | 48.9 |
| 15 | 0.1330% | 0.4801% | 3.61x | **56.7** |
| 18 | 0.3093% | 0.7593% | 2.45x | 48.2 |
| 19 (riskiest) | 0.6089% | 0.8432% | 1.38x | 17.9 |

Not a uniform level error. The safest tenth of the book is calibrated within sampling
error, the riskiest bin is out by 1.38x, and the **middle is out by more than 4x**. Two
consequences worth carrying forward:

- **It is the worst possible shape economically.** Mid-risk loans are the marginal
  approve-or-decline population, so the error is largest exactly where the cutoff sits.
  Stage 4 should expect the profit-maximising threshold to be materially misplaced.
- **It is why Platt cannot help.** Platt applies a two-parameter monotone transform to the
  log-odds, which can move a level but cannot bend an arch.

By contrast the in-time holdout has **no arch at all**: every one of its twenty bins sits
within ±2.9 z of its predicted rate, and the ratios scatter around 1 with no trend in risk.
The two bins that look worst (0.23x, 0.35x) hold one and four events respectively.

### Why in-time recalibration is worth nothing here, and what is

| Recalibrator | Fitted on | Out-of-time ratio | ECE | % of base rate |
|---|---|---|---|---|
| none | — | 0.3717 | 1.68e-03 | 62.9% |
| Platt | in-time holdout | 0.3636 | 1.71e-03 | 63.7% |
| isotonic | in-time holdout | 0.3733 | 1.68e-03 | 62.7% |
| **Platt** | **2006, applied to 2007–08** | **0.8128** | **7.67e-04** | **28.1%** |

**Stated before measuring, and confirmed:** in-time recalibration does not fix out-of-time
miscalibration. Platt makes it marginally *worse*.

The reason is visible in the fitted parameters rather than inferred: Platt's slope came back
at **0.9887** with an intercept of **−0.0924** — very nearly the identity. There was nothing
in-time for it to correct, because in-time the model is right. Isotonic, fitted on the same
sample, learned the same nothing.

So the correction is not unlearnable, only unlearnable *from the training period*. Fitted on
the first test vintage and applied to the next two, a two-parameter Platt closes **72% of
the gap** — ratio 0.3388 to 0.8128, ECE down 57% and the Brier reliability term down 82%
(5.70e-06 to 1.00e-06). Caveat attached and load-bearing: fully observing the 2006 cohort
takes 120 months, so a lender in 2007 did not have this. It is an upper bound on what
recalibration could achieve, not an operating recommendation.

### Cumulative incidence is worse than monthly, and degrades with horizon

The model emits a monthly hazard; a lender prices the chance of default over years, which
is the monthly hazards compounded through the survival product. Errors compound with them.

| Horizon | Loans | Predicted | Observed | Ratio |
|---|---|---|---|---|
| 12m | 129,338 | 0.323% | 0.890% | 0.363 |
| 24m | 129,231 | 0.999% | 3.443% | 0.290 |
| 36m | 129,073 | 1.561% | 6.388% | 0.244 |
| 60m | 128,846 | 2.635% | 10.335% | 0.255 |

Loans censored before a horizon are excluded from it rather than counted as survivors,
which would understate incidence by the share of the book that stopped reporting; the
excluded counts are small here (73 to 565) because these cohorts are mature. Prepaid loans
stay in and count as non-defaults, which is correct under competing risks — for a prepaid
loan, default did not happen first.

### What it costs: the money number, and the cleanest available proof of claim 2

Expected NPV per loan, projected from origination across the 120-month horizon under the
fitted competing-risks hazards. 20,526 test loans are excluded because they enter the panel
after age 0 — an NPV is a value at the decision point, and there is no decision point on
record for a loan first seen at age 6.

Two populations, because the sequential arm needs the first test vintage to fit on:

**All 129,411 eligible test loans** (in-time recalibrators only):

| Variant | NPV per loan | per dollar | vs raw |
|---|---|---|---|
| raw | $2,365 | 1.246% | — |
| Platt (in-time) | $2,382 | 1.255% | +$17, +0.74% |
| isotonic (in-time) | $2,358 | 1.242% | −$7, −0.28% |

**The 87,955 eligible 2007–2008 loans**, adding the arm that actually moves the
probabilities:

| Variant | NPV per loan | per dollar | vs raw |
|---|---|---|---|
| raw | $3,396 | 1.745% | — |
| Platt (in-time) | $3,413 | 1.754% | +$17, +0.50% |
| isotonic (in-time) | $3,392 | 1.743% | −$3, −0.10% |
| **Platt (fitted on 2006)** | **$2,046** | **1.051%** | **−$1,349, −39.7%** |

**Claim 2 is demonstrated, and by the sharpest construction available.** Platt scaling is
*strictly* monotone, so it cannot reorder two loans: AUC is unchanged to the last decimal
place, and the Brier resolution term moves by less than 0.2%. Discrimination is held
provably fixed. On the same loans, with the same ranking, correcting only the *level* of the
probability changes assessed portfolio value by **39.7%**. Excellent ranking with a wrong
level misprices the book, which is exactly what the claim asserts.

Read the direction carefully: recalibration does not *add* $1,349 a loan. It reveals that
the raw model **overstated** the book's value by that much. A lender pricing 2007–2008
originations off a model fitted on 2000–2004 would have valued the portfolio at roughly
1.66 times what the corrected hazard supports, and approved accordingly.

The sign survives every economic assumption tested. Across the severity grid the sequential
delta runs −$1,135 (LGD 0.100) to −$2,056 (LGD 0.241), and across discount spreads of
50–150bp the band never approaches zero. It scales with severity, as it must — the
correction acts through the default leg — so the *magnitude* is assumption-dependent while
the conclusion is not.

**Two caveats that belong next to the number, not in a footnote.**

1. **$1,349 is an upper bound, not an achievable result.** The sequential calibrator was
   fitted on realised 2006 outcomes, and observing that cohort fully takes 120 months. A
   lender in 2007 did not have it. What the number measures is what the mispricing *cost*,
   not what a lender could have recovered at the time.
2. **In-time recalibration — the standard recipe — recovers 0.5% of it.** Hold back part of
   your training data, recalibrate on it, and portfolio value moves by $17 a loan against a
   $1,349 error. That is the practically important finding of this stage, and it is a
   negative one: at the point where it would have mattered, recalibration did not work,
   because the calibration function itself was not stable across the regime break.

Every absolute NPV here carries the terminal-at-par assumption of §8 limitation 10, which
understates value and understates it more for better loans. It very largely cancels in the
*differences* between variants, since all four price the same balances at the same horizon,
and the differences are what the claims rest on — but no absolute NPV above should be quoted
as a valuation.

### The control arm: how much of this is the crisis?

The obvious objection to everything above, and it is a fair one. Training on 2000–2004 and
testing on 2006–2008 straddles the largest US housing crisis in living memory, so
"under-predicts out of time" is close to guaranteed by the split rather than discovered in
it. On its own the −39.7% is uninterpretable: it could be a regime effect specific to a
once-in-a-generation break, or ordinary out-of-time drift that the crisis merely amplifies.

So the whole pipeline was re-run with the break taken out of the comparison —
`configs/calm_control.yaml`, train 2000–2002 against test 2003–2004, every other
assumption copied verbatim except the three inputs measured from the training cohorts
(effective LGD 0.1330 → 0.1045, P(loss disposition) 0.3697 → 0.3384, recovery lag 27 → 26),
which would otherwise leak test-period information into the control. Reproduce the table
with `scripts/compare_split_arms.py`.

| | crisis (2000–04 → 2006–08) | control (2000–02 → 2003–04) |
|---|---|---|
| train / test base rate | 0.0765% / 0.2680% | 0.0862% / 0.0691% |
| shift, test ÷ train | **3.50x** | **0.80x** |
| in-sample ratio *(a control)* | 0.9996 | 0.9996 |
| in-time holdout ratio | 1.0212 | 1.0447 |
| **out-of-time ratio** | **0.3717** | **0.6449** |
| out-of-time ECE ÷ base rate | 62.9% | 35.6% |
| in-time Platt closes | −1.3% | −4.3% |
| sequential closes | 71.7% | 35.7% |
| NPV effect, $ per loan | **−$1,349** | **−$94** |
| NPV effect, pp of principal | **−0.6933pp** | **−0.0610pp** |
| in-time Platt, pp of principal | +0.0088pp | +0.0056pp |

**The answer is: mostly the crisis, and the control can date it.** The control's cumulative
incidence is *well calibrated for the first two years and breaks afterwards*:

| Horizon | crisis ratio | control ratio |
|---|---|---|
| 12m | 0.363 | **1.184** |
| 24m | 0.290 | **1.001** |
| 36m | 0.244 | 0.906 |
| 60m | 0.255 | 0.667 |

A 2003–2004 origination reaches 24 months in 2005–2006 and 60 months in 2008–2009. The
control model is *accurate to within 0.1%* over the window that ends before the crash and
degrades to 0.667 over the window that contains it. That is not generic out-of-time drift
with a date attached by coincidence; the degradation appears exactly when the crisis enters
the test cohorts' lives. The crisis arm, by contrast, is already at 0.363 by 12 months,
because a 2006–2008 origination meets the crash almost immediately.

**In money the crisis effect is eleven times the control's** — −0.6933 against −0.0610
percentage points of principal. So the headline number is substantially a stress result and
§8 records it as one.

**What reproduces in both arms is the mechanism, and that is the transferable finding.**
In-time recalibration is worthless in the mild arm too: it moves NPV by +0.0056pp against a
−0.0610pp error, and *widens* the marginal ratio gap by 4.3%. Recalibrating on the nearest
out-of-time cohort helps in both (36% and 72%). The in-sample control lands at 0.9996 in
both, and the in-time holdout at 1.02–1.04 in both, so overfitting is negligible either way.
The claim is therefore not "calibration fails in crises" but "calibration does not transfer
across time, and holding back training data does not tell you that it won't."

**Two honest caveats on the control, both material.**

1. **It is not a clean control, and cannot be built from this data.** At a 120-month horizon
   no cohort here is crisis-free — the 2004 vintage lives through 2004–2014 and meets the
   crash at ages 48–84. Measured severity confirms it: effective LGD is 0.1045 on 2000–2002
   against 0.1557 on 2003–2004, already 1.49x worse. So the control is
   *less*-crisis-exposed against *more*, which makes −$94 an **upper** bound on non-crisis
   drift and eleven times a **lower** bound on the crisis contribution.
2. **The control's NPV base is near zero, so its percentage is meaningless.** The 2003–2004
   vintages carry the thinnest note-rate-over-discount headroom in the book, 50bp and 39bp
   against 110bp for 2000, and at a 35bp servicing cost their projected NPV per loan is
   −$150 — the book destroys value before any calibration correction. A percentage change
   against that denominator reads −62.7% against the crisis arm's −39.7% and **inverts the
   comparison**. Only the dollar and per-dollar figures are comparable, which is why
   `compare_split_arms.py` reports those and refuses the percentage.

**A prediction recorded in advance, and wrong.** Before running it I expected the control to
*over*-predict, since its test book has the lower base rate (0.80x) and a model carrying its
training level across would therefore be too high. It under-predicts instead, at 0.6449.
The reason is worth more than the prediction was: the miscalibration is not a level error
inherited from the training period at all. The 2003–2004 cohorts look unusually *safe* at
origination — the refinancing wave brought high scores and low loan-to-value — so the model
predicts below even its own training rate, and they then default more than their origination
profile implies because they met the crash in mid-life. **Origination covariates cannot
represent a calendar effect.** That is the age–period–cohort problem of §6a arriving again
by a different route, and it is why no amount of recalibration on origination-time data
fixes this. A normalisation built on the base-rate shift was tried and abandoned for the
same reason; `compare_split_arms.py` documents why so it is not retried.

## 6d. Stage 4 measured results

Measured 8 September 2026 with `cde decide`, on the same 80%-of-train hazard stage 3 used.

### The foundation: realised NPV, and a correction to this document

Stage 4 needed something stage 3 did not: a way to score a decision that is not the
forecast which made it. §5.5 originally specified the ablation on "expected NPV per
applicant", and that is **circular**. Stage 3 measured the size of the problem — the
uncalibrated hazard values the 2007–2008 book at $3,396 a loan against $2,046 for the
honestly recalibrated one, so an expected-NPV scoreboard ranks B3 above B4 by $1,349 and
the better-calibrated model loses by construction. §5.5 is corrected; the rule is that
**decisions come from predictions and scores come from cash.**

`src/cde/economics/realised.py` supplies the cash, rebuilt month by month from the
observed balance path: interest is `UPB(t) × rate(t)/12`, principal is `UPB(t) − UPB(t+1)`,
and a payoff, a curtailment or capitalised arrears after a modification all fall out of
the balance path without special cases. It reads `performance`, not `person_period`,
because person-period is truncated at first 90+ DPD — for a defaulted loan it stops while
the borrower still owes an average of $184,587, and 63% of those loans cure.

**The invariant that says it is right rather than plausible:** undiscounted principal plus
terminal balance equals cash advanced, by telescoping, for every loan. It holds to
**$0.0000 across all 129,449 loans**, with none off by more than $1.

Chasing that invariant found two data problems that would each have gone unnoticed:

1. **`ORIGINAL UPB` is rounded to the nearest $1,000** as a disclosure measure, so the
   identity fails against it for 6.4% of loans. 99.54% sit within the rounding (mean
   −$347, ~$577 spread) but 595 are further out and one loan is **$463,000** below its
   reported original. Realised NPV is therefore valued against the *first reported
   balance* — the money that actually left the lender.
2. **141 loans (0.109%) never report a positive balance at all**, carrying a single
   zero-balance record. Subtracting their reported principal booked a phantom loss
   averaging −$217,440 each — **−$337 per loan across the whole book, a third of the
   headline, manufactured by a tenth of a percent of it.** Excluded and counted.

### What the book actually did

| | per loan | per dollar of principal |
|---|---|---|
| model's expected NPV (raw hazard) | **+$2,365** | +1.25% |
| model's expected NPV (recalibrated) | +$2,046 | +1.05% |
| **realised** | **−$1,035** | **−0.55%** |

The 2006–2008 book destroyed value, and the model expected it to make money. Recalibration
closed $1,349 of a roughly $3,400 gap; the rest is severity (the train-measured effective
LGD of 0.1330 against a realised test figure of 0.2407) and the residual hazard shortfall
that even the sequential calibrator leaves at 0.81 rather than 1.0.

### Claim 3: the cutoff is an economic quantity

Every rule below ranks applicants by the **same** predicted 36-month cumulative default
probability from the **same** fitted hazard. Only the criterion turning that score into a
cutoff differs, so nothing about model quality is in play. Statistical thresholds are
chosen on the in-time holdout, never on this book.

| Rule | Approves | Realised NPV per applicant | vs approve-all |
|---|---|---|---|
| approve all | 100.0% | −$1,034 | — |
| **accuracy-maximising** | **100.0%** | **−$1,034** | **$0** |
| F1-maximising | 98.0% | −$824 | +$209 |
| **expected NPV > 0** | **77.5%** | **+$393** | **+$1,427** |
| oracle (hindsight) | 51.5% | +$784 | +$1,817 |

**The accuracy-maximising cutoff is the do-nothing rule, exactly** — and it is not an
empirical accident but arithmetic. Accuracy is `(TN + TP)/n`; at a 36-month default rate
of 6.4% the true negatives dominate, so declining anyone trades a certain true negative
for a probable false positive and accuracy falls monotonically as approval falls. It
reaches its maximum of **0.936** at 100% approval. "Our model is 93.6% accurate" is the
most common claim in a public credit notebook and it describes a rule that declines nobody
and loses $1,034 an applicant.

F1 is not degenerate — it ignores true negatives, so it does yield a real threshold — but
it is still the wrong one, because it prices a missed default against a wrongly declined
applicant at an implicit 1:1 when the true exchange rate on this book is nearer 30:1. It
recovers $209 of the available $1,817.

The economic rule needs no sweep and no tuning: approve exactly those whose expected NPV
is positive. Under the model that *is* the optimum. It earns **+$1,427 an applicant over
the accuracy criterion** and turns a loss-making book into a profitable one.

### Where claims 2 and 3 meet, which is the more interesting result

Stage 3 found in-time recalibration worth $17 a loan — nothing. Run through the decision
layer on the 2007–2008 loans, correcting the level is worth considerably more, because the
level is what places the cutoff:

| Rule | raw hazard | recalibrated hazard |
|---|---|---|
| approve all | −$79 | −$79 |
| F1-maximising | +$158 @ 98.0% | +$801 @ 87.0% |
| **expected NPV > 0** | **+$919 @ 85.0%** | **+$1,196 @ 71.5%** |
| oracle (hindsight) | +$1,279 @ 63.5% | +$1,252 @ 62.0% |

The raw model under-predicts default, so its expected-NPV rule approves **85%** of the
book when about **63%** was optimal. Correcting the level moves the cutoff to 71.5% and
earns **+$277 an applicant**. In terms of the value actually available on this ranking,
the raw rule captures 71.9% of the oracle and the recalibrated rule **95.5%**.

So calibration's value does not show up in the assessed portfolio total, where stage 3
looked and found almost nothing. It shows up in **where the cutoff lands**. That is the
mechanism the project was built to demonstrate, and it sets the expected sign and rough
magnitude for stage 5's B3→B4 comparison.

### A subtlety that qualifies stage 3's AUC claim

The oracle should be identical between those two arms — it depends only on the ranking and
the realised cash, and Platt scaling is strictly monotone. It is not: $1,279 against
$1,252.

The reason is worth recording. Platt is monotone in the **monthly hazard**, but the
decision scores on **cumulative incidence**, and
`F_d(T) = Σ_{t≤T} S(t)·h_d(t)` with `S(t) = Π_{u<t}(1 − h_d(u) − h_p(u))`. Recalibrating
`h_d` changes the survival product as well as the increments, and loans have different
prepayment profiles, so two loans can swap order. **A monotone correction to the monthly
hazard is not a monotone correction to cumulative incidence.** §8 limitation 11 says AUC
is unchanged to the last decimal; that is true of the monthly hazard and false of the
quantity the decision actually uses. The measured cost here is a $27 reduction in the
oracle, so the recalibrated arm's +$277 is *net* of a small ranking degradation.

### Out of scope, and what remains assumption

**Risk-based pricing and adverse selection are not built.** §5 lists them as "extension if
time allows", and they are a genuinely larger piece: setting the rate as a function of PD
changes who accepts the offer, so price and default rate stop being independent and the
population being modelled becomes endogenous. Nothing in this data identifies an
acceptance function — every loan here was accepted — so it could only ever be simulated
under an assumed elasticity. Recorded in §11 rather than half-done.

Three carried assumptions bound every figure above.

* **The oracle is hindsight.** It is chosen by looking at the realised answer, so it is
  not a rule any lender could run. It bounds what this ranking could have earned with a
  perfectly placed cutoff, which is the only reason the other arms' gaps are interpretable.
* **Realised NPV is itself optimistic by about $350 a loan.** 94.0% of realised loss falls
  inside the 120-month window; the remaining $45.2M belongs to dispositions that concluded
  later and is absent. Same censoring, same direction, as the LGD measurement.
* **The terminal balance is booked at par** on both sides (§8 limitation 10), so it
  cancels in every comparison between rules and does not cancel in any absolute NPV.

## 6e. Stage 5 measured results — the B0-B4 ablation

Measured 9 September 2026 with `cde ablate`, on the eligible **2007-2008** loans. The 2006
cohort is spent fitting B4b's calibrator, and an ablation whose arms are scored on
different books measures the books, so every arm runs on the same 87,854 applicants under
the same economics.

**Everything is held fixed except the thing under test.** The binary comparison model sees
the identical 80% of the training vintages the hazard was fitted on — 199,599 loans, 2,751
in-window defaults, base rate 1.378% — with the identical covariates, encoding, L2
strength and optimiser tolerance. It has **85 features against the hazard's 92**, and the
difference is exactly the seven loan-age spline columns, which it has no time axis for.
Both converged: marginal gaps +0.023% and −0.039%.

| arm | approves | realised | step | its own forecast | AUC | cap/cut |
|---|---|---|---|---|---|---|
| B0 approve all | 100.0% | −$79 | — | — | 0.500 | 0.0% |
| **B1 binary, accuracy-max** | **100.0%** | **−$79** | **$0** | $4,405 | 0.741 | 0.0% |
| B2 binary, profit-max | 89.2% | +$460 | **+$538** | $4,541 | 0.741 | 57.8% |
| B3 hazard, profit-max | 84.5% | +$915 | **+$455** | $3,640 | 0.806 | 67.8% |
| B4a + in-time recalibration | 84.7% | +$896 | **−$18** | $3,652 | 0.806 | 66.8% |
| B4b + sequential recalibration | 71.3% | +$1,460 | **+$563** | $2,660 | 0.803 | 92.9% |

### Reading the table

**B1 adds exactly nothing, on an AUC of 0.741.** Its accuracy-maximising threshold, chosen
honestly on the in-time holdout, comes back at 0.517 with an accuracy of 0.9860 — and
approves 100% of applicants. The model's single most confident prediction is a 51.7%
default probability and accuracy still says write it, because declining it trades a
certain true negative for a probable false positive. B1 is the typical public notebook: it
builds a model that ranks competently and then makes no decisions with it.

**The decision layer alone is worth +$538.** Same model, same AUC, same ranking — only the
criterion changes, from accuracy to positive expected value. That is claim 3, measured
inside the ablation as well as in §6d.

**The survival framing is worth a further +$455**, and it earns it twice over: AUC also
rises 0.741 → 0.806. So knowing *when* a loan defaults improves the ranking as well as the
pricing, which is more than claim 1 strictly asserts.

**Honest recalibration is worth −$18.** B4a is what a lender could actually have run, and
it does nothing, exactly as stage 3 predicted in advance. **B4b — recalibrated on the first
test cohort's realised outcomes — is worth +$563**, and it is an upper bound rather than a
rule, because observing that cohort fully takes 120 months.

### The column that proves the scoring rule matters

Read the forecast column against the realised one. It falls monotonically from B2's $4,541
to B4b's $2,660 while realised profit *rises* from $460 to $1,460. **The arms that expect
less earn more.** Had the ablation been scored on expected NPV as §5.5 originally
specified, the ranking would have been almost exactly reversed and the report would have
concluded that calibration destroys value.

### Failure analysis: which cohorts break it, and why

**By cohort, 2007 is the failure and 2008 is not:**

| vintage | loans | expected | realised | gap | 36m default |
|---|---|---|---|---|---|
| 2007 | 42,051 | $733 | **−$3,274** | $4,006 | 8.86% |
| 2008 | 45,803 | $5,838 | **+$2,854** | $2,984 | 5.59% |

A 2007 origination met the crash about eighteen months into its life, at the point where
the hazard is near its peak and almost no principal has amortised. A 2008 origination was
written *after* prices had begun falling, under visibly tighter underwriting, and made
money. The model cannot tell them apart, because everything separating them is calendar
rather than borrower — the age-period-cohort problem of §6a, arriving for the third time.

**By the model's own risk decile, the error is a SHAPE error:**

| decile | mean score | expected | realised | gap | 36m default |
|---|---|---|---|---|---|
| 0 (safest) | 0.0012 | $3,065 | +$2,231 | $834 | 0.13% |
| 3 | 0.0045 | $4,075 | +$2,207 | $1,868 | 2.22% |
| 5 | 0.0087 | $3,856 | +$1,095 | $2,761 | 4.74% |
| 6 | 0.0124 | $3,704 | −$99 | $3,803 | 7.22% |
| 9 (riskiest) | 0.0626 | $1,268 | **−$7,934** | **$9,202** | 26.08% |

Two things to take from it. The gap varies **elevenfold** across deciles, so this is not a
level error and **a two-parameter monotone recalibration cannot fix it** — which is
precisely why B4a is worth −$18 rather than something. And the model's expected NPV is
**positive in every decile**, from $3,065 down to $1,268: it never expects to lose money on
anybody, while realised value crosses zero between the fifth and sixth deciles. A lender
running it would have written the whole book believing every segment profitable.

### Two oracles, and why one of them exceeds 100%

An early version of this table reported a single "capture" of **115.2%** described as a
share, which is impossible if the description is right. The cause is a result rather than
a bug and it is now reported as two columns.

`cap/cut` measures an arm against the best possible cutoff on the quantity it actually
thresholds, and is bounded by 100%. `cap/rank` measures it against the best cutoff on the
**risk ranking alone** — approve the safest k applicants — and is not bounded, because an
arm approves on ``expected NPV > 0`` and that is *not* a prefix of the risk order.
Expected NPV also depends on the loan's balance, rate and term, so a slightly riskier loan
can be worth writing while a safer one is not.

The pattern across arms is itself informative. For B2 the cutoff oracle sits *below* the
risk oracle (57.8% against 44.4% capture of a lower ceiling), because a flat-hazard NPV is
a poor decision variable. For B3 and B4b it sits *above* it. **The better the timing model,
the more pricing on value beats ranking on risk** — which is the strongest form of claim 1
in the project, and it was not something the ablation was designed to test.

### What bounds all of it

* Every arm carries the terminal-at-par convention (§8 limitation 10) and the ~$350 a loan
  of realised-loss censoring (§6d), identically, so both cancel in the steps between arms
  and neither cancels in a level.
* B4b saw outcomes no lender had. It is the ceiling on what recalibration could be worth,
  not a recommendation.
* One population, one economics, one training sample. The steps are therefore attributable
  — but only to the six things actually varied, and the ablation says nothing about feature
  engineering, alternative learners, or a multi-state treatment of cure (§8 limitation 9).

## 7. Architecture

Mirrors the staged, swappable structure already used in `guitar-transcription`, for the same
reason: each stage must be independently evaluable.

```
credit-decision-engine/
  PROJECT_PLAN.md
  README.md
  pyproject.toml
  configs/default.yaml          # vintages, horizon, discount rate, LGD assumptions
  src/cde/
    config.py      # typed, validated access to the YAML — the only reader of it
    data/          # Release 47 schema, text loader, DuckDB ingest + key validation
    features/      # person-period construction; sql/person_period.sql
    models/        # hazard models (swappable)
    calibration/   # Platt, isotonic, diagnostics
    economics/     # NPV, threshold optimisation, pricing
    eval/          # per-stage evaluation harnesses
    cli.py
  tests/           # incl. a synthetic fixture with outcomes known by construction
  docs/            # one spec note per stage
  notebooks/       # exploration only — never the deliverable
```

Analytical SQL lives in ``.sql`` files rather than in Python strings, so it reads as SQL and
can be shown as SQL. Only the type-cast layer is generated, from the schema's own column map:
that part is boilerplate, and generating it means adding a column cannot leave a cast behind.

Practices: YAML-driven config so no assumption is buried in code, a CLI per stage, pytest, ruff,
mypy. Every economic assumption (discount rate, LGD, servicing cost) lives in config and is
sensitivity-tested, never hard-coded.

## 8. Declared limitations

Written now, so they are design constraints rather than retrofitted excuses.

1. **Selection bias is uncorrectable here.** The population is loans that passed an originator's
   underwriting *and* Freddie Mac's purchase criteria. There are no rejected applications at all.
   This pushes estimates optimistic. Reject inference cannot be validated on this data, so state
   the direction and magnitude of the bias rather than pretending to a correction.
2. **Secured lending, and the parameters are specific to it.** Mortgage loss given default is
   collateral-driven, and the measured 0.360 severity at disposition reflects house-price
   recovery, not the near-total loss an unsecured lender faces. The techniques here — survival
   framing, calibration, an economic decision layer — carry across products; the estimated
   parameters do not, and no claim is made that they do.
3. **US, not UK.** Different macro regime, different regulatory framework, no UK affordability
   rules in the data.
4. **Administrative censoring at 120 months** discards behaviour past ten years by design, and
   still misses 7.5% of eventual loss dispositions. The cost is measured rather than assumed —
   see §4.3 — and a 180-month sensitivity is available from config alone.
5. **`ACTUAL LOSS` is null** for loans disposed within three months of the cutoff and where a
   Defect Settlement Date is populated — LGD estimation is on a subset. It is computed only for
   zero-balance codes `02`, `03`, `09` and `15`, is disclosed as a *positive* value with gains
   negative, and excludes modification costs.
6. **Left truncation, in addition to the selection bias in (1).** The performance file starts
   "from the time of loan acquisition by Freddie Mac", not from origination (user guide,
   "Interpreting the Data"). Seasoned loans therefore enter the panel at loan age > 0, and any
   loan that deteriorated between origination and Freddie Mac's purchase never enters at all.
   The discrete-time hazard handles the first part structurally — a loan contributes rows only
   for the ages at which it is observed — but the second part is an unobservable, and it pushes
   estimates optimistic in the same direction as (1). Stage 1 measures the distribution of
   minimum observed loan age so the size of the exposure is known rather than assumed.
7. **Some censoring is informative, and cannot be made otherwise.** Release 47 zero-balance
   codes `15` (whole loan sale), `16` (reperforming loan securitisation) and `96` (confirmed
   underwriting or major servicing defect) all remove a loan from the panel for reasons
   correlated with its credit state — `16` by construction, since a *reperforming* loan is one
   that was previously delinquent. Treating these as ordinary right-censoring assumes
   non-informative censoring, which is false. The v1 treatment is to censor and report the
   affected volume; there is no correction available on this data.
8. **Prepayment censoring is informative too, and on this dataset it dominates.** These are
   30- and 15-year fixed-rate mortgages, and better-credit borrowers refinance, so treating
   voluntary payoff (`01`) as non-informative censoring biases the hazard in the flattering
   direction. On the 2000–2004 training vintages this is aggravated by the 2003 refinancing
   wave. See open question 4 — the exit mix measured in stage 1 is what should decide it.
9. **The NPV blends cure and loss into one recovery, and cannot represent a cure properly.**
   Default here is first passage to 90+ DPD, and **63.0% of those loans never produce a loss
   disposition at all** — they catch up, are modified into performance, or leave by a route
   carrying no recorded loss. The identity in §5.4 handles the *severity* of that correctly, via
   an effective LGD that is the measured product of P(loss disposition | 90+ DPD) = 0.3697 and
   severity given disposition = 0.3596, and `npv.py` splits the recovery leg so that only the
   disposition half is discounted for the recovery lag.

   What it cannot represent is the *timing*. Booking a cured loan's whole balance at the default
   month treats it as if it had prepaid, discarding every payment it goes on to make. Measured
   by `cde lgd` on the training vintages: **5,830 cured loans, a mean $5,642 of discounted value
   discarded each, $32.9M in total.** Since the note rate exceeds the discount rate on this book
   a loan that carries on paying is worth *more* than par, so this **understates** NPV.

   An earlier draft of this limitation put it at "$36.8k per cured loan across 3,097 loans",
   which was wrong twice over — a total mistaken for a per-loan figure, and a loan count from the
   superseded 60-month horizon. The measurement is now produced by `cure_opportunity_cost` and
   printed by `cde lgd`, so it cannot go stale silently. The real magnitude is **4.7% of the
   balance outstanding at default**, not 25%: the note rate exceeds the discount rate by ~90bp, so
   continued payments are worth only a little over par. The bias is conservative and modest,
   and it falls hardest on exactly the loans the model scores as risky. The correct treatment is
   a multi-state model
   (performing → delinquent → cured or terminated) rather than a two-cause competing-risks one.
   Out of scope for v1, and the single most valuable extension.
10. **The terminal value is booked at par, which it is not worth.** The panel stops at 120 months
   and these are 30-year loans, so 83.7% of principal is still outstanding at the horizon — 46%
   of principal in present value. That balance is booked at par because the alternative,
   projecting hazards past the last month the model has any support for, would be inventing data.
   Par is wrong in a known direction: a loan paying above the discount rate is worth more than
   par, so this understates NPV, and understates it more for the better loans. Every NPV in
   stage 3 and 4 carries this, and it cancels almost entirely in the *differences* between
   variants — which is what the claims rest on — but not in any absolute NPV.
11. **Isotonic recalibration also improves discrimination, so its arm is not a clean test.**
   Isotonic regression pools adjacent violators, which are exactly the prediction intervals where
   the observed rate ran backwards against the model. Collapsing those to ties removes
   locally-wrong orderings, and a tie scores 0.5 in an AUC where a wrong ordering scores 0.
   Measured at +0.007 to +0.011 AUC on the synthetic harness, with only ~20 distinct output
   values surviving from 120,000 rows. Platt, being strictly monotone, leaves AUC unchanged to
   the last decimal place. So the B3→B4 gap in the §5.5 ablation is attributable to calibration
   alone **only for the Platt arm**; the isotonic arm can mix in a discrimination gain. Both are
   reported for exactly this reason, rather than only the better-scoring one.

   **This holds for the monthly hazard and NOT for cumulative incidence**, which is what the
   stage 4 decision layer actually scores on. Since
   `F_d(T) = Σ_{t≤T} S(t)·h_d(t)` with `S(t) = Π_{u<t}(1 − h_d(u) − h_p(u))`, recalibrating
   `h_d` changes the survival product as well as the increments, and loans with different
   prepayment profiles can swap order. So even Platt is not rank-preserving in the quantity
   the cutoff is applied to. Measured in §6d as a $27 reduction in the oracle's realised NPV
   per applicant, against a $277 gain from the recalibration itself — small, but it means
   even the Platt arm is not a perfectly clean calibration-only comparison.

   **On the real out-of-time data the effect did not reproduce**, and the honest reading is that
   this is a caveat to watch rather than a measured problem here: the Brier resolution term on
   the 2006–2008 book is 6.5930e-06 raw against 6.5830e-06 after isotonic — a fractional
   *decrease*, not the increase the synthetic harness showed. The mechanism is real and the
   synthetic demonstration is reproducible, but it needs a prediction vector with genuine local
   rank violations to bite, and the fitted hazard out of time apparently has few.
12. **Isotonic regression is unusable at its tail on rare-event data, and its failure is
   spectacular rather than subtle.** Isotonic fits a free monotone step function, and at a
   0.08% monthly base rate the top of the prediction range holds almost no rows. On the
   in-time holdout (2,427,055 rows, 1,834 events) the fit's knots ran out of data long
   before they ran out of range:

   | knot x | fitted y | calibration rows at or above x |
   |---|---|---|
   | 0.032054 | 0.029630 | 136 |
   | 0.043344 | 0.029630 | 2 |
   | 0.043626 | **1.000000** | **1** — which happened to default |

   So isotonic asserted a *monthly* default hazard of 1.0 — certain default inside the
   month — from a single observation. The highest well-estimated monthly rate anywhere in
   this data is 0.004985 (the worst of twenty equal-count bins of ~121,000 rows), making
   that terminal value **200 times too large, from n = 1**. `out_of_bounds="clip"` does not
   help: the value being clipped *to* is the problem.

   It is not academic. Projecting every loan across every month on book reaches (loan, age)
   combinations the observed panel never contains, so the projection's hazard range runs
   past the calibration sample's — 43 of 15.5 million projected rows sat above the fit's
   support and every one was mapped to 1.0. Added to the prepayment hazard that sent
   `h_default + h_prepayment` over 1 and survival negative, and the guard in
   `HazardModels.survival` refused the frame outright. **That guard was added for an
   unrelated reason and is the only thing that caught this**; without it the run would have
   produced negative survival, negative NPV contributions, and a plausible final number.

   The mitigation is `calibration.isotonic_min_tail_rows: 1000`, which caps the output at
   the highest fitted value supported by at least that many calibration rows. It is a
   disclosed cap, reported by `describe()` and by the share of rows it binds on, not a
   silent clamp. The general lesson belongs in the write-up: isotonic's flexibility is
   bought with its tail, and on rare-event data the tail is where pricing decisions live.

13. **The headline stage 3 magnitude is a stress result, and must be labelled one.** The
   −39.7% NPV effect is measured across the 2008 housing crisis, which is the largest
   regime break available in this data and was chosen deliberately because a milder split
   would leave the effect too small to measure. The control arm of §6c puts a number on how
   much that choice is worth: re-run with the break taken out of the comparison, the same
   experiment gives **−$94 a loan against −$1,349, or −0.061 against −0.693 percentage
   points of principal — eleven times smaller.** So the *mechanism* generalises and the
   *magnitude* does not, and no write-up should quote −39.7% as a typical cost of
   miscalibration. What generalises is the structure: the ranking survives, the level does
   not, and in-time recalibration fails to detect or fix it in both arms.

   The control also cannot be made clean on this data, which bounds the claim in the other
   direction: at a 120-month horizon every cohort here meets the crisis somewhere in its
   life, so eleven times is a *lower* bound on the crisis contribution rather than a
   measurement of it.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Clarity registration latency | Register first, before anything else — the only step with latency outside my control |
| Data volume swallows the schedule | Sample-first; full quarters only if a specific analysis needs them |
| Release 47 layout mismatch | **Resolved:** build the schema from `file_headers_july_2026.zip` (31 / 35 columns, verbatim). Every tutorial, public repo and pre-July-2026 user guide has the old names and the old placement of Servicer Name and MI Cancellation Indicator |
| `LOAN AGE` resets on modification | Detect via `MODIFICATION FLAG`; note that payment deferrals do *not* reset it |
| Scope creep via competing risks / reject inference / pricing | All three are explicitly stretch. Survival, calibration and the decision layer are the spine |
| Theory phase overruns | If it slips, cut reject inference and scorecards to a paragraph each in the write-up |

## 10. Open questions — resolve before stage 1

1. **Discount rate r.** What funding cost plus required return? Needs a defensible source, not a
   round number.
2. **LGD treatment in v1.** Constant assumption, or modelled from realised losses? Modelling it
   is better and costs time.
3. ~~**Train/test vintage cutoff.**~~ **Resolved 7 September 2026: train on vintages
   2000–2004, test on 2006–2008.** The GFC is the largest genuine regime shift in the data, and
   an out-of-time test across it produces degradation worth reporting rather than a null result.
   2005 is omitted deliberately as the ambiguous transition year, so the training set is not
   contaminated by the start of the deterioration it is supposed to fail to anticipate.
   Eight `sample_YYYY.zip` files, ~400,000 loans.

   **Caveat to state in the write-up:** Freddie Mac's Standard dataset is fixed-rate,
   full-documentation, non-government only (user guide, "Loan Selection Criteria"). The worst of
   the boom — Alt-A, no-doc, option ARMs — is filtered out before we see it, so the degradation
   measured here is a *lower bound* on what the market experienced. The rejected alternative was
   train 2015–2018 / test 2019–2022: cleaner data, but COVID forbearance suppressed observed
   defaults, so the model would look better out-of-time than it deserves and the result would be
   a policy artefact rather than a modelling one.
4. ~~**Competing risks in v1 or deferred?**~~ **Resolved 7 September 2026 by
   measurement: prepayment must be modelled as a cause-specific hazard. It moves from
   stretch scope into the spine.**

   Stage 1 measured the exit mix on all eight vintages. Within 120 months on book,
   prepayment accounts for **77% to 94% of each cohort** — 94% for 2000, 91% for 2001,
   85% for 2008 — against cumulative default of 2.85% to 15.66%. Prepayment is not a
   nuisance competing with the event of interest; it is, by an order of magnitude, the
   main thing that happens to these loans.

   The measured consequence, which is the argument rather than the assertion: treating
   prepayment as censoring and applying Kaplan–Meier to default alone overstates
   cumulative default by **6.57x on the 2000 vintage, 4.30x on 2001, 2.02x on 2003** —
   monotonically in the prepayment share, exactly as the theory requires, since
   `F_d(t) <= 1 - KM(t)` with equality only where the prepayment hazard is zero.

   Note this does *not* invalidate the naive counting estimator here. Prepayment is a
   competing risk, not censoring: a repaid loan is a real, final outcome that cannot
   default, so it stays in the denominator, which is what the counting estimator does.
   Across all eight cohorts the counting estimator and Aalen–Johansen agree to within
   0.00068, because every cohort is fully mature at 120 months. It would break on a live
   book, where most vintages are not.
5. ~~**Delinquency-based or termination-based default definition?**~~ **Resolved
   7 September 2026: the PD model dates default at first passage to 90+ days past due
   (`CURRENT LOAN DELINQUENCY STATUS >= "03"`); LGD is estimated separately from the
   zero-balance disposition records.**

   The two definitions were never really competing — they are inputs to different terms
   of PD x LGD x EAD:

   - **90+ DPD dates the borrower's event.** It is the Basel regulatory reference point
     for default, and it is available for every loan including those still alive.
   - **The zero-balance date is the month the legal process concluded.** `ZERO BALANCE
     CODE` is set in the period matching the Zero Balance Effective Date, and US
     foreclosure timelines ran 12-30 months post-2008 — longer in judicial-foreclosure
     states. Predicting that month means partly predicting state foreclosure law and
     servicer capacity. It also breaks the NPV arithmetic: the identity in §5.4 counts
     payments collected before the loss, and disposition-dating credits the book with
     payments the borrower never made, overvaluing it systematically.
   - **But realised loss only exists at disposition.** `ACTUAL LOSS` is attached to the
     zero-balance record, because the loss is unknowable until the property is sold. So
     LGD is estimated there, on codes 02/03/09.

   Two consequences to carry, both stated rather than fixed:

   - 90+ DPD is **not absorbing** — a material fraction of borrowers cure. This is
     therefore a first-passage target: time to *first* 90+ DPD. That is the right target
     for an approve/decline decision (you would rather not have written the loan) but it
     is not the same quantity as "time to loss", and the write-up must not conflate them.
     **Stage 1 measures the cure rate** so the size of this is known.
   - `RA` (REO acquisition) in the delinquency column is a default state, not a missing
     day count, and must be caught explicitly.

   **Uncertain, flag before quoting:** 90 days past due is the standard reference point,
   but for retail mortgage exposures some jurisdictions have historically permitted up to
   180 days. Verify against BIS primary documents before asserting the regulatory
   definition in the write-up.

   `default_definition.mode` keeps `termination` switchable, and stage 1 reports the
   hazard curve both ways along with the distribution of the lag between them.

## 11. Explicitly out of scope

Deployment, serving, or an API. A dashboard. Deep learning. Gradient boosting as the headline
model — an interpretable hazard model that can be reasoned about is the point, and boosting can
appear as a discrimination benchmark at most. Fair-lending and disparate-impact analysis, which
is genuinely important and deserves more than a token section.

---

**Risk-based pricing and adverse selection** (added 8 September 2026). §5 lists these as a
stage 4 extension "if time allows". They are out of scope for v1, and the reason is a data
limitation rather than a time one: setting the rate as a function of PD changes who accepts
the offer, so price and default rate stop being independent and the modelled population
becomes endogenous. **Nothing in this dataset identifies an acceptance function — every loan
in it was accepted, and there are no declined applications at all** (§8 limitation 1). It
could only be simulated under an assumed elasticity, which would produce a result about the
assumption rather than about the data. Named in the write-up as the most valuable extension
alongside the multi-state cure model of §8 limitation 9.

## Provisional schedule

Roughly one week of theory, two to three weeks of building. The spine is stages 1–5; everything
marked stretch is cut first if it slips.

| Week | Focus |
|---|---|
| 0 | Register for Clarity. Theory: credit fundamentals, unit economics, survival analysis |
| 1 | Stage 1 — data pipeline, person-period table, vintage SQL |
| 2 | Stage 2 and 3 — hazard model, calibration, downstream NPV impact |
| 3 | Stage 4 and 5 — decision layer, out-of-time ablation, failure analysis, write-up |
