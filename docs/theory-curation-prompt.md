# Perplexity prompt — credit risk theory curation

For the credit decision engine project (Lendable Summer Intern 2027 / quant + data roles generally).
Run in Perplexity's **research / deep mode**, not quick mode. Paste everything inside the fence.

---

```
I am a second-year BSc Natural Sciences (Physics and Statistics) undergraduate at UCL. I have
solid university-level probability, statistical inference, regression and linear algebra (marks
in the 80s-90s), and I work in Python (pandas, NumPy, scikit-learn, matplotlib) with some SQL.
I have NO finance background, no survival analysis, and no credit risk knowledge.

I am building a portfolio project and I want to learn the underlying theory properly BEFORE
writing any code. Your job is to CURATE THE LEARNING RESOURCES — I am not asking you to teach
me the theory in your answer, I am asking you to tell me exactly what to read, in what order,
and why each source is the right one.

THE PROJECT
A credit decision engine on public consumer lending data (Lending Club or similar). Rather than
a binary default classifier, I want to:
  (a) model default as a time-to-event process so I get a hazard curve by months-on-book,
  (b) calibrate the predicted probabilities so they are usable as levels, not just rankings,
  (c) feed those into a loan NPV / expected-profit calculation,
  (d) choose the approval threshold (and ideally risk-based pricing) that maximises expected
      profit rather than accuracy,
  (e) validate out-of-time rather than on a random split, and honestly characterise where the
      model fails.

TOPICS I NEED RESOURCES FOR — cover each, in a sensible learning order
 1. Credit risk fundamentals: expected loss as PD x LGD x EAD, and how the Basel IRB framework
    defines these. What LGD and EAD look like for unsecured consumer lending.
 2. Loan unit economics: deriving the NPV of an instalment loan as a discounted cash-flow stream
    under a default hazard. How PD level and PD timing each enter the NPV.
 3. Survival analysis (the core): survival function, hazard rate, right-censoring, Kaplan-Meier,
    Cox proportional hazards, and especially DISCRETE-TIME HAZARD MODELS fitted as logistic
    regression on a person-period dataset. Why censoring makes a naive binary default target
    biased.
 4. Vintage / cohort analysis in consumer credit, and the age-period-cohort identification
    problem (separating loan maturation from vintage quality from macroeconomic effects).
 5. Calibration vs discrimination: AUC and Gini (Gini = 2*AUC - 1), reliability curves, Brier
    score, expected calibration error, and recalibration via Platt scaling and isotonic
    regression. Why calibration matters more than ranking when the output feeds a pricing model.
 6. Reject inference and selection bias: why observing outcomes only for approved applicants
    biases everything, what parcelling / augmentation / Heckman-style bivariate probit
    corrections do, and the evidence on whether reject inference actually helps.
 7. Cost-sensitive decision theory: choosing an approval threshold to maximise expected profit;
    risk-based pricing; adverse selection when the offered rate changes who accepts.
 8. Credit scorecards as practised in industry: weight of evidence (WOE) binning, information
    value (IV), logistic scorecards, points-to-double-odds scaling, and why regulated lenders
    prefer these over gradient boosting.

OUTPUT FORMAT — one block per topic, in reading order, each containing:
 - MUST-KNOW or NICE-TO-HAVE for the project as specified above.
 - Realistic time estimate to reach working understanding, for someone with my background
   specifically (do not pad for someone who lacks the statistics).
 - PRIMARY SOURCE: the single best thing to read. Name the exact book AND chapter numbers, or
   the exact paper with authors, year, venue and DOI, or the exact course with lecture numbers.
   Never a bare book title, never "read about X".
 - BACKUP SOURCE: a second source at a different level or from a different angle, in case the
   primary does not land — same citation standard.
 - FREE or PAYWALLED, marked per source, with a link. Prioritise freely available primary
   sources: official Basel/BIS documents, arXiv, university lecture notes and open textbooks.
 - WHAT TO SKIP within that source — which chapters or sections are irrelevant to my project.
 - The common misconception or thing practitioners get wrong on this topic.
 - PREREQUISITES: which earlier topic in the list this one depends on.

THEN, AFTER THE PER-TOPIC BLOCKS, give me:
 - THE CRITICAL PATH: the minimum subset of topics and sources that is genuinely load-bearing
   for this project. If I only had one week, what exactly would I read?
 - THE TWO OR THREE SOURCES worth reading end to end, versus everything else which is reference.
 - A CONSOLIDATED READING LIST in the order I should work through it, with cumulative hours.
 - PYTHON ECOSYSTEM: which packages handle survival analysis, calibration, and WOE/scorecard
   binning; what each is actually for; which are maintained; and where each one's own docs or
   tutorials are the best available teaching resource.
 - DATASET GUIDANCE: which Lending Club release (or a better public alternative) to use, where
   to get it now that Lending Club no longer publishes it directly, its known problems and
   biases, and the preprocessing traps people fall into with it.
 - WORKED EXAMPLES: any public notebook, tutorial or textbook companion repo that shows a
   discrete-time hazard model or a profit-maximising threshold done properly on credit data.

WHAT I DO NOT WANT
 - Generic "introduction to machine learning" or "intro to statistics" material. Assume I
   already know regression, maximum likelihood, hypothesis testing and cross-validation.
 - Medium posts, listicles, "top 10" articles, SEO blog content, or Kaggle notebook links.
 - Vague recommendations with no specific source, chapter or section.
 - You teaching me the theory instead of pointing me at where to learn it. Brief framing of why
   a source matters is welcome; a lecture is not.

SOURCE QUALITY RULES
 - Every source needs something I can verify: a DOI, an ISBN, or a working URL.
 - State the edition and year of every book, and flag where a newer edition changes the chapter
   numbering you have given me.
 - Where you are uncertain whether a source exists, is available, is the right edition, or is
   any good, SAY SO EXPLICITLY rather than guessing. I would rather have eight confident
   citations and three flagged uncertainties than eleven confident-sounding ones.
```

---

## Before you trust the output

Perplexity hallucinates chapter numbers and occasionally invents plausible DOIs. The sources I'd
expect it to converge on:

- Kleinbaum & Klein, *Survival Analysis: A Self-Learning Text*
- Singer & Willett, *Applied Longitudinal Data Analysis*, ch. 10-12 (discrete-time hazard)
- Baesens, Rosch & Scheule, *Credit Risk Analytics* (the practical one, Python/R throughout)
- Thomas, Edelman & Crook, *Credit Scoring and Its Applications* (academic standard)
- Niculescu-Mizil & Caruana, "Predicting Good Probabilities With Supervised Learning", ICML 2005
- Crook & Banasik on whether reject inference actually improves application scorecards

If it returns something confident that contradicts these, or a source neither of us can find,
treat it as a flag rather than a discovery. Spot-check two or three citations on Google Scholar.

## Follow-ups worth asking once it responds

- "For topic 3, give me a worked numerical example of building a person-period dataset and
  fitting a discrete-time hazard model, with the data layout shown explicitly."
- "What are the standard criticisms of the Lending Club dataset in the academic literature, and
  do they invalidate a project like mine?"
- "What questions would a credit risk interviewer ask to test whether I actually understand
  calibration versus discrimination?"

Ask that last one early. It tells you what the theory is *for* — the Lendable case-study rounds.
