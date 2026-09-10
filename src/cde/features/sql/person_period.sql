-- ==========================================================================
-- Stage 1 -- the person-period table: one row per loan per month at risk.
--
-- This is the layout a discrete-time hazard model is fitted on. Each row asks
-- "given that this loan was still alive at the start of month t, did it default
-- during month t?", and logistic regression over those rows estimates the
-- hazard h(t) directly. Censoring needs no special handling: a loan simply
-- stops contributing rows after its last observed month, which is the whole
-- reason this dataset was chosen over a loan-level snapshot.
--
-- CALENDAR ORDER IS LOAD-BEARING. Everything below sequences by PERIOD, never
-- by LOAN AGE, because LOAN AGE is not monotonic within a loan: a modification
-- resets it to count from the Modification First Payment Date instead of from
-- origination. 10,091 of 400,000 loans in the 2000-2008 sample do this, and
-- 8,832 of those resets fall inside the 120-month horizon. Filter on
-- `loan_age <= exit_age` alone and such a loan contributes each early age twice
-- -- once from its original life and once from its post-modification life --
-- which duplicates rows, fires the event indicator more than once, and can
-- manufacture an in-horizon default out of a delinquency that happened years
-- later.
--
-- F08Q10000392 shows the reset pattern: current for 70 months, delinquent to 27
-- months down, then modified at age 100 with the clock reset to 3 and the term
-- re-amortised from 140 to 477 months, then delinquent again.
--
-- Note that at the current 120-month horizon this particular loan is no longer a
-- case where the bug bites -- its first 90+ DPD is age 73, which is now inside
-- the window, so the loan exits there and the reset at age 100 is never reached.
-- (Under the 60-month horizon this document originally specified, age 73 fell
-- outside the window and the post-reset second pass through age 49 sat inside
-- it, manufacturing a default out of a 2020 delinquency on a 2008 loan.) The bug
-- bites whenever a reset precedes any exit event, which is what the truncation
-- below prevents -- 706 loans censor at `clock_reset_censor` for exactly that
-- reason.
--
-- (loan_identifier, period) is unique and validated at ingest, so period order
-- is well defined even when loan_age is not.
--
-- Substituted from configs/default.yaml, never hard-coded:
--   $delinquency_threshold   missed-payment count that counts as default
--   $default_codes           zero-balance codes that are credit events
--   $informative_codes       zero-balance codes whose censoring is informative
--   $horizon                 administrative censor, months on book
--   $default_age_expression  which clock dates the event (open question 5)
--   $off_clock_filter        whether clock-reset loans are excluded
-- ==========================================================================


-- --------------------------------------------------------------------------
-- 1. Sequence each loan in calendar order and find the discontinuities.
--
-- Two ways the months-on-book axis can break:
--   * the clock resets  -- loan_age fails to increase (a modification)
--   * a month is missing -- loan_age jumps by more than one
-- Both are detected against the *previous row in period order*, never against
-- an assumed starting age.
-- --------------------------------------------------------------------------
CREATE OR REPLACE VIEW performance_sequenced AS
SELECT
    *,
    lag(loan_age) OVER (PARTITION BY loan_identifier ORDER BY period) AS previous_loan_age,
    coalesce(
        loan_age <= lag(loan_age) OVER (PARTITION BY loan_identifier ORDER BY period),
        false
    ) AS loan_age_reset,
    coalesce(
        loan_age - lag(loan_age) OVER (PARTITION BY loan_identifier ORDER BY period) > 1,
        false
    ) AS follows_month_gap
FROM performance;


-- --------------------------------------------------------------------------
-- 2. Classify every loan-month.
--
-- Each flag is wrapped in coalesce(..., false) deliberately. In SQL, NULL >= 3
-- is NULL rather than false, and NULL IN ('02','03') is NULL too, so an
-- unwrapped flag would poison every downstream boolean with three-valued logic
-- and quietly drop rows from aggregates.
-- --------------------------------------------------------------------------
CREATE OR REPLACE VIEW performance_classified AS
SELECT
    loan_identifier,
    origination_vintage,
    period,
    period_date,
    loan_age,
    previous_loan_age,
    remaining_months_to_legal_maturity,
    current_actual_upb,
    current_interest_rate,
    current_loan_delinquency_status,
    modification_flag,
    payment_deferral_flag,
    borrower_assistance_plan,
    delinquency_due_to_disaster,
    zero_balance_code,
    zero_balance_effective_date_date AS zero_balance_effective_month,
    actual_loss,
    loan_age_reset,
    follows_month_gap,

    -- Once the axis breaks it stays broken: a running OR over calendar order,
    -- true from the offending row onward. Everything from there on describes a
    -- different clock and is not comparable with months-since-origination.
    bool_or(loan_age_reset) OVER (
        PARTITION BY loan_identifier ORDER BY period ROWS UNBOUNDED PRECEDING
    ) AS clock_broken,
    bool_or(follows_month_gap) OVER (
        PARTITION BY loan_identifier ORDER BY period ROWS UNBOUNDED PRECEDING
    ) AS after_month_gap,

    -- Missed payments as a number, or NULL where the status is not a count.
    try_cast(current_loan_delinquency_status AS INTEGER) AS missed_payments,

    -- 90+ days past due, when the threshold is 3.
    --
    -- CURRENT LOAN DELINQUENCY STATUS is text and must be read as text. Numeric
    -- codes are zero-padded counts of missed payments capped at 99; 'RA' means
    -- REO acquisition, which is a default state rather than a day count, and
    -- try_cast('RA' AS INTEGER) is NULL so the numeric test alone would miss the
    -- most severe rows in the file. 'XX' (not available) is already NULL by the
    -- time it reaches here, cleared as a sentinel in the typed view.
    --
    -- Note what is NOT written here: status >= '$delinquency_code'. That string
    -- comparison looks equivalent and is not -- both 'RA' and 'XX' sort above
    -- '03' lexically, so it would silently score missing data as severe.
    coalesce(
        current_loan_delinquency_status = '$delinquency_reo'
        OR try_cast(current_loan_delinquency_status AS INTEGER) >= $delinquency_threshold,
        false
    ) AS at_or_past_delinquency_threshold,

    coalesce(zero_balance_code IN $default_codes, false) AS terminated_credit_event,
    coalesce(zero_balance_code = '$prepaid_code', false) AS terminated_prepaid,
    coalesce(zero_balance_code IN $informative_codes, false) AS terminated_informative,

    -- The loan's own term, implied by where it says it is. For a loan on the
    -- origination clock this equals ORIGINAL LOAN TERM in every month, because
    -- one month elapsed is one month less remaining.
    loan_age + remaining_months_to_legal_maturity AS implied_original_term,

    -- Missed payments cannot exceed months on book. When they do, LOAN AGE is
    -- not counting from origination -- see the off-clock note in step 4. The +1
    -- allows for the reporting-cycle offset between the delinquency and investor
    -- reporting systems that the user guide warns about.
    coalesce(try_cast(current_loan_delinquency_status AS INTEGER) > loan_age + 1, false)
        AS delinquency_exceeds_loan_age
FROM performance_sequenced;


-- --------------------------------------------------------------------------
-- 3. The eligible window: the contiguous run of months, from the loan's first
-- observation, over which months-on-book means what it says.
--
-- Truncated at the administrative horizon, at a clock reset, and at a missing
-- month. Within this window loan_age is strictly increasing, so it is once
-- again a usable time index -- which is what makes step 5 safe.
-- --------------------------------------------------------------------------
CREATE OR REPLACE VIEW person_period_eligible AS
SELECT *
FROM performance_classified
WHERE NOT clock_broken
  AND NOT after_month_gap
  AND loan_age >= 0
  AND loan_age <= $horizon;


-- --------------------------------------------------------------------------
-- 4. Resolve each loan to a single exit: when it left the risk set, and why.
--
-- Every loan gets exactly one exit_reason, so the categories must be exhaustive
-- and mutually exclusive. An unclassified exit would be swept into "still
-- alive" and overstate survival.
--
-- OFF-CLOCK LOANS. Some loans enter the panel with LOAN AGE already reset by a
-- modification that happened before Freddie Mac acquired them, so there is no
-- earlier row to detect the reset against. In the Release 47 format example,
-- loan F26Q60000002 enters at loan_age 0 reporting 14 months delinquent and 60
-- months remaining on a 30-year product, with MODIFICATION FLAG blank -- so the
-- reset is invisible from the flag alone. Two independent signals catch it:
-- missed payments exceeding months on book, and the implied term at entry
-- disagreeing with ORIGINAL LOAN TERM.
--
-- This matters because such a loan's first passage to 90+ DPD happened before
-- observation began. Treated naively it records as "defaulted in month 0" and
-- teaches the hazard model that month 0 is near-certain default. It is left
-- truncation with the event already realised, which a first-passage model
-- cannot use -- see PROJECT_PLAN.md section 8 limitation 6.
-- --------------------------------------------------------------------------
CREATE OR REPLACE VIEW loan_exit AS
WITH observed AS (
    -- Over the loan's WHOLE observed history, ignoring eligibility. Two jobs.
    --
    -- First, locating the axis breaks, in months-on-book at the last good row,
    -- ordered by period so "first" means first in calendar time.
    --
    -- Second, diagnostics that deliberately look past the horizon. The lag
    -- between first 90+ DPD and the disposition that follows it is the evidence
    -- for open question 5, and disposition routinely lands well outside 120
    -- months -- so measuring it needs the full history, not the modelling
    -- window. These `_observed` columns are for reporting only. Nothing in the
    -- exit resolution below may use them, or the horizon would leak.
    SELECT
        loan_identifier,
        arg_min(previous_loan_age, period) FILTER (WHERE loan_age_reset)
            AS age_before_clock_reset,
        arg_min(previous_loan_age, period) FILTER (WHERE follows_month_gap)
            AS age_before_month_gap,
        bool_or(loan_age_reset) AS ever_clock_reset,
        bool_or(follows_month_gap) AS ever_month_gap,
        count(*) AS observed_months_total,
        min(loan_age) FILTER (WHERE at_or_past_delinquency_threshold)
            AS first_delinquent_age_observed,
        min(loan_age) FILTER (WHERE terminated_credit_event) AS credit_event_age_observed,
        min(period_date) FILTER (WHERE at_or_past_delinquency_threshold)
            AS first_delinquent_month_observed,
        min(period_date) FILTER (WHERE terminated_credit_event) AS credit_event_month_observed
    FROM performance_classified
    GROUP BY loan_identifier
),
ages AS (
    SELECT
        loan_identifier,
        any_value(origination_vintage) AS origination_vintage,

        -- Left truncation. The performance file starts from Freddie Mac's
        -- acquisition, not from origination, so a seasoned loan enters the panel
        -- at loan_age > 0. Exposed rather than assumed to be zero.
        min(loan_age) AS first_observed_age,
        max(loan_age) AS last_eligible_age,
        count(*) AS eligible_months,

        min(loan_age) FILTER (WHERE at_or_past_delinquency_threshold) AS first_delinquent_age,
        min(loan_age) FILTER (WHERE terminated_credit_event) AS credit_event_age,
        min(loan_age) FILTER (WHERE terminated_prepaid) AS prepaid_age,
        min(loan_age) FILTER (WHERE terminated_informative) AS informative_exit_age,

        bool_or(delinquency_exceeds_loan_age) AS any_delinquency_exceeds_loan_age,
        -- Ordered by PERIOD, not by loan_age: loan_age is exactly the column
        -- whose trustworthiness is in question here.
        arg_min(implied_original_term, period) AS entry_implied_term
    FROM person_period_eligible
    GROUP BY loan_identifier
),
flagged AS (
    SELECT
        a.*,
        b.age_before_clock_reset,
        b.age_before_month_gap,
        b.ever_clock_reset,
        b.ever_month_gap,
        b.observed_months_total,
        b.first_delinquent_age_observed,
        b.credit_event_age_observed,
        b.first_delinquent_month_observed,
        b.credit_event_month_observed,
        o.original_loan_term,
        coalesce(
            o.original_loan_term IS NOT NULL
            AND a.entry_implied_term IS NOT NULL
            AND a.entry_implied_term <> o.original_loan_term,
            false
        ) AS entry_term_mismatch,
        -- Which clock dates the default event. Under `delinquency` this is the
        -- first of 90+ DPD or a credit-event termination, whichever comes first:
        -- a charge-off is a default even in the rare case where the delinquency
        -- column never recorded 90+ DPD before it. least() ignores NULLs, so a
        -- loan with neither gets NULL and never defaulted.
        $default_age_expression AS default_age
    FROM ages AS a
    JOIN observed AS b USING (loan_identifier)
    -- LEFT, not INNER: a loan present in performance but absent from origination
    -- must still appear here so the join gap is visible and counted rather than
    -- silently shrinking the cohort.
    LEFT JOIN origination AS o USING (loan_identifier)
),
resolved AS (
    SELECT
        *,
        (any_delinquency_exceeds_loan_age OR entry_term_mismatch) AS off_clock,
        least(
            default_age,
            prepaid_age,
            informative_exit_age,
            $horizon,
            last_eligible_age
        ) AS exit_age
    FROM flagged
)
SELECT
    loan_identifier,
    origination_vintage,
    first_observed_age,
    last_eligible_age,
    eligible_months,
    observed_months_total,
    original_loan_term,
    entry_implied_term,
    first_delinquent_age,
    credit_event_age,
    prepaid_age,
    informative_exit_age,
    default_age,
    exit_age,
    -- Reporting only, and deliberately unbounded by the horizon. See the note
    -- on the `observed` CTE above.
    first_delinquent_age_observed,
    credit_event_age_observed,
    first_delinquent_month_observed,
    credit_event_month_observed,
    datediff(
        'month', first_delinquent_month_observed, credit_event_month_observed
    ) AS months_delinquent_to_disposition,
    -- Precedence matters only for ties, and default wins them: recording a
    -- defaulted loan as a clean payoff is the expensive direction of that error.
    -- The two clock censors come last because they only apply when the loan was
    -- still at risk and nothing else ended it.
    CASE
        WHEN default_age = exit_age THEN 'default'
        WHEN prepaid_age = exit_age THEN 'prepaid'
        WHEN informative_exit_age = exit_age THEN 'informative_exit'
        WHEN exit_age >= $horizon THEN 'administrative_censor'
        WHEN age_before_clock_reset = exit_age THEN 'clock_reset_censor'
        WHEN age_before_month_gap = exit_age THEN 'month_gap_censor'
        ELSE 'data_censor'
    END AS exit_reason,
    off_clock,
    any_delinquency_exceeds_loan_age,
    entry_term_mismatch,
    ever_clock_reset,
    ever_month_gap,
    age_before_clock_reset,
    coalesce(age_before_clock_reset <= $horizon, false) AS clock_reset_within_horizon,
    -- Did a termination code fire without the delinquency column ever showing
    -- 90+ DPD first? Should be rare; if it is not, the delinquency column has
    -- gaps worth understanding before trusting the delinquency-based definition.
    (
        credit_event_age IS NOT NULL
        AND (first_delinquent_age IS NULL OR credit_event_age < first_delinquent_age)
    ) AS credit_event_preceded_delinquency,
    -- A loan first seen after the administrative horizon contributes no rows at
    -- all, so it is in loan_exit but not in person_period.
    (first_observed_age > $horizon) AS entered_after_horizon
FROM resolved;


-- --------------------------------------------------------------------------
-- 5. The person-period table.
--
-- Rows up to and including the exit month. The exit month is included because
-- the loan was at risk at the start of it -- that is the discrete-time
-- convention: the interval is the unit, and the event indicator says whether
-- the event happened during it.
--
-- default_event is true only on the exit row of a defaulting loan. Every row of
-- a prepaid, censored or informatively-exited loan carries false, which is
-- exactly how censoring enters: those loans contribute risk-set exposure for the
-- months they were observed and then stop, with no event ever recorded.
--
-- Drawn from person_period_eligible, so loan_age is strictly increasing per
-- loan and `loan_age <= exit_age` selects one contiguous run with no repeats.
-- --------------------------------------------------------------------------
CREATE OR REPLACE TABLE person_period AS
SELECT
    p.loan_identifier,
    p.origination_vintage,
    p.loan_age,
    p.period,
    p.period_date,
    e.exit_age,
    e.exit_reason,
    (p.loan_age = e.exit_age) AS is_exit_month,
    (p.loan_age = e.exit_age AND e.exit_reason = 'default') AS default_event,
    (p.loan_age = e.exit_age AND e.exit_reason = 'prepaid') AS prepayment_event,
    e.first_observed_age,
    e.ever_clock_reset,

    -- Time-varying, observed monthly. current_actual_upb is exposure at default
    -- when the event fires, which is the EAD term of PD x LGD x EAD.
    p.current_actual_upb,
    p.current_interest_rate,
    p.current_loan_delinquency_status,
    p.missed_payments,
    p.modification_flag,
    p.payment_deferral_flag,
    p.borrower_assistance_plan,
    p.delinquency_due_to_disaster,
    p.zero_balance_code,
    p.actual_loss,

    -- Origination covariates, carried on every row. EXCLUDE rather than an
    -- explicit list so that adding a column to the schema does not require
    -- editing this query; which of them are actually used is stage 2's problem.
    o.* EXCLUDE (loan_identifier, origination_vintage)
FROM person_period_eligible AS p
JOIN loan_exit AS e USING (loan_identifier)
JOIN origination AS o USING (loan_identifier)
WHERE p.loan_age <= e.exit_age
  $off_clock_filter;
