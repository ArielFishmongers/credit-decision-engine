# Credit Decision Engine — Project Plan

**Status:** draft v0.3, 3 September 2026. Written before any code exists, and expected to change
once stage 0 (theory) is done. Nothing below is settled except the framing and the three claims.

**Author:** Ariel Fishgang
**Companion documents:** `docs/reading-list.md` (to be written in stage 0)

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

Portfolio artefact and interview preparation for UK Summer 2027 internships. The proximate
target is Lendable's Summer Intern 2027 (Growth & Analytics, London, ten weeks from 29 June
2027), whose posting describes the work as *"using data and modelling to ground assumptions for
our NPV model, such as predicting what default rates will be in the future, and using that to
inform our lending decisions."* Their process includes a coding take-home and two case studies,
so the real success criterion is not that the code runs — it is that every modelling decision
can be defended out loud.

It generalises deliberately: the same artefact supports quantitative and data roles elsewhere,
so it must not be written as a Lendable-specific pitch.

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

Freddie Mac loans are 15- and 30-year mortgages; loan age runs to 360 months. Lendable's product
is unsecured consumer instalment lending over 1–7 years. Modelling a 360-month hazard would be
both computationally heavy and economically remote from the target domain.

**Decision: censor administratively at 60 months on book.** Rationale: it brackets consumer-loan
tenor, most mortgage default risk falls within years 2–5 anyway, and it makes the NPV horizon
tractable. This is an assumption, it is stated in the write-up, and its sensitivity is tested.

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

**Deliverable:** `docs/reading-list.md` with verified sources, and the ability to derive the NPV
identity in §5.4 unaided.

### Stage 1 — Data and vintage analysis

Download script against Clarity; parse the pipe-delimited files with an explicit Release 47
column schema; load to DuckDB; construct the person-period table (one row per loan per month at
risk, up to the 60-month administrative censor).

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

**Competing risks:** `ZERO BALANCE CODE` distinguishes voluntary payoff (`01`) from default
outcomes (`03`, `09`). Prepayment is not a nuisance — a loan repaid early stops paying interest,
which is a real NPV term, and Lendable has the same problem on consumer loans. Model default and
prepayment as separate cause-specific hazards if stage 2 lands on schedule; otherwise treat
prepayment as censoring in v1 and document the resulting bias.

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

Reporting expected NPV per applicant across B0–B4 on the out-of-time set is the single most
informative table in the project, and it is what an interviewer will actually interrogate.

**Metrics:** discrimination as AUC and Gini (= 2·AUC − 1, the form credit teams quote); calibration
as reliability curve, Brier and ECE; survival-specific time-dependent AUC and integrated Brier
score; economics as expected NPV per applicant and approval rate at the chosen cutoff.

### Stage 6 — Write-up

A README an interviewer can read in ten minutes and follow the argument, with a limitations
section that is written honestly rather than defensively.

## 6. Definition of done, per stage

| Stage | Done when |
|---|---|
| 0 | Reading list verified; NPV identity derivable unaided |
| 1 | Person-period table built and documented; vintage curves computed in SQL; APC problem written up |
| 2 | Hazard curve by month on book; censoring handling documented; competing-risks decision made and justified |
| 3 | Reliability curves and Brier before/after recalibration; downstream NPV impact quantified |
| 4 | NPV derived in the write-up; expected profit as a function of threshold; profit-maximising cutoff found and compared to accuracy-maximising |
| 5 | Out-of-time ablation table B0–B4; degradation reported honestly; failure analysis naming which cohorts break the model and why |
| 6 | README complete, limitations section included |

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
    data/          # Clarity download, Release 47 schema, parsing, DuckDB load
    features/      # person-period construction
    models/        # hazard models (swappable)
    calibration/   # Platt, isotonic, diagnostics
    economics/     # NPV, threshold optimisation, pricing
    eval/          # per-stage evaluation harnesses
    cli.py
  tests/
  docs/            # reading-list.md, one spec note per stage
  notebooks/       # exploration only — never the deliverable
```

Practices: YAML-driven config so no assumption is buried in code, a CLI per stage, pytest, ruff,
mypy. Every economic assumption (discount rate, LGD, servicing cost) lives in config and is
sensitivity-tested, never hard-coded.

## 8. Declared limitations

Written now, so they are design constraints rather than retrofitted excuses.

1. **Selection bias is uncorrectable here.** The population is loans that passed an originator's
   underwriting *and* Freddie Mac's purchase criteria. There are no rejected applications at all.
   This pushes estimates optimistic. Reject inference cannot be validated on this data, so state
   the direction and magnitude of the bias rather than pretending to a correction.
2. **Secured, not unsecured.** Mortgage LGD is collateral-driven; unsecured consumer LGD is near
   total. EAD and recovery dynamics differ. The technique transfers; the parameters do not.
3. **US, not UK.** Different macro regime, different regulatory framework, no UK affordability
   rules in the data.
4. **Administrative censoring at 60 months** discards long-horizon behaviour by design.
5. **`ACTUAL LOSS` is null** for loans disposed within three months of the cutoff and where a
   Defect Settlement Date is populated — LGD estimation is on a subset.

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
3. **Train/test vintage cutoff.** Which origination vintages train, which test? Must span a
   regime change to be interesting — candidate: train pre-2008, test 2008–2010, versus train
   pre-2020, test 2020–2022.
4. **Competing risks in v1 or deferred?** Decide at the end of stage 2, not before.
5. **Delinquency-based or termination-based default definition?** 90+ days delinquent, or
   zero-balance codes 03/09? These give materially different hazard curves and the choice needs
   justifying.

## 11. Explicitly out of scope

Deployment, serving, or an API. A dashboard. Deep learning. Gradient boosting as the headline
model — an interpretable hazard model that can be reasoned about is the point, and boosting can
appear as a discrimination benchmark at most. Fair-lending and disparate-impact analysis, which
is genuinely important and deserves more than a token section.

---

## Provisional schedule

Roughly one week of theory, two to three weeks of building. The spine is stages 1–5; everything
marked stretch is cut first if it slips.

| Week | Focus |
|---|---|
| 0 | Register for Clarity. Theory: credit fundamentals, unit economics, survival analysis |
| 1 | Stage 1 — data pipeline, person-period table, vintage SQL |
| 2 | Stage 2 and 3 — hazard model, calibration, downstream NPV impact |
| 3 | Stage 4 and 5 — decision layer, out-of-time ablation, failure analysis, write-up |
