# Credit Decision Engine

## What this is

A credit decision engine on public loan-level data. **Not a default classifier.** It predicts
*when* borrowers default, calibrates those probabilities well enough to price a loan with, and
wraps them in a decision rule that maximises expected profit rather than accuracy.

Read `PROJECT_PLAN.md` before doing anything substantive. It carries the framing, the data
decisions, the stage definitions and the open questions, and it is the spec this repo implements.

## The three claims

Everything here exists to demonstrate one of these. **If a proposed component serves none of
them, question whether it belongs.**

1. **Timing matters as much as incidence.** A month-2 default destroys far more value than a
   month-30 one. A binary target throws that away.
2. **Level matters more than ranking.** NPV consumes the probability itself. Excellent AUC with
   systematic overconfidence misprices every loan in the book.
3. **The optimal cutoff is an economic quantity, not a statistical one.** It sits where marginal
   expected profit reaches zero, not where accuracy or F1 peaks.

## Who is working on this

Ariel Fishgang — second-year BSc Natural Sciences (Physics and Statistics) at UCL, graduating
June 2028. Strong university probability, statistical inference, regression and linear algebra.
Works in Python (pandas, NumPy, scikit-learn) and SQL (Athena/Trino), and builds ETL and data
pipelines professionally at WareBee.

- **Assume known:** regression, maximum likelihood, hypothesis testing, cross-validation, and
  handling messy real data.
- **Assume not known:** finance, credit risk, survival analysis, industry vocabulary. These are
  being learned alongside the build. Do not skip the reasoning because it looks elementary.

The deliverable is interview-facing. Every modelling decision must be defensible out loud, so
explaining *why* is part of the work, not an optional extra.

## How to work here

- **Spec before implementation.** Update `PROJECT_PLAN.md` when a decision changes; do not let
  the code and the plan drift apart.
- **Teach, don't just deliver.** When writing code, state the modelling decision behind it and
  what the alternatives were.
- **Challenge cargo-culting.** If a technique is being reached for because it is familiar or
  fashionable rather than because it fits, say so.
- **Ask rather than assume.** If a decision turns on something about the data that is not yet
  known, ask. Several assumptions are still open — see `PROJECT_PLAN.md` §10.
- **Be blunt about scope.** Say when something is a week of work disguised as an afternoon.
- **Flag your own uncertainty**, especially on finance conventions and regulatory detail, where
  the author cannot yet catch an error.

## Hard rules

1. **Never commit data.** `data/` and `docs/freddie-mac/` are gitignored. The Freddie Mac licence
   covers *use*, not redistribution — the royalty-free non-commercial terms accepted at Clarity
   registration, not the fee-based commercial agreement. Ship download scripts, never files.
2. **Schema comes from `file_headers_july_2026.zip`, nothing else.** Release 47 renamed columns
   and moved two between files. Every tutorial, public repo and pre-July-2026 user guide has the
   old layout. `src/cde/data/schema.py` is generated from the authoritative headers; do not
   hand-edit it against a blog post.
3. **Sample first.** Build and validate the whole pipeline on the 50,000-loans-per-vintage sample
   before touching full quarters. The full panel is ~2.9 billion performance rows.
4. **No metric theatre.** If a number looks good because of a leak, a random split or a lucky
   cutoff, find it and say so before it reaches a README.
5. **Out-of-time validation, never a random split.** Random splits leak future information.
6. **Economic assumptions live in `configs/default.yaml`**, never hard-coded, and every one is
   sensitivity-tested.
7. **Never overstate the author's experience.** No production deployment, no C++, no end-to-end
   neural network training. Do not write CV or README language implying otherwise.
8. **State limitations honestly.** Selection bias cannot be corrected here — the population is
   loans that passed both an originator's underwriting and Freddie Mac's purchase criteria, and
   there are no rejected applications at all. Name the direction of the bias; do not fake a fix.

## Layout

```
PROJECT_PLAN.md          the spec — framing, data, stages, open questions
configs/default.yaml     every economic and modelling assumption
scripts/                 fetch_reference_docs.sh (data itself is manual, via Clarity)
src/cde/
  data/      schema.py (Release 47, generated), loader.py
  features/  person_period.py    stage 1
  models/    hazard.py           stage 2
  calibration/                   stage 3
  economics/ npv.py, decision.py stage 4
  eval/      harness.py          stage 5 — the B0-B4 ablation
docs/freddie-mac/        Freddie Mac reference docs (gitignored, fetched)
notebooks/               exploration only — never the deliverable
```

## Commands

```bash
pip install -e ".[dev]"          # install
cde schema                       # print the Release 47 column layout
pytest                           # tests
ruff check . && mypy src         # lint and types
scripts/fetch_reference_docs.sh  # restore docs/freddie-mac/
```

## Data access

The dataset is **not** downloadable without an account. Register at Clarity Data Intelligence,
**CRT portal** (not MBS — the loan-level dataset is a credit-risk-transfer artefact):
<https://claritydownload.fmapps.freddiemac.com/CRT/#/sflld> → Data Download → SFLLD.

Take `sample_YYYY.zip` (containing `sample_orig_YYYY.txt` and `sample_perf_YYYY.txt`) for a few
vintage years. Support: clarity@freddiemac.com.
