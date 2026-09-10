-- ==========================================================================
-- Stage 1 -- vintage curves.
--
-- Everything the day-3 charts need, from three counts per (vintage, age):
-- how many loans were at risk, how many defaulted, how many prepaid. The
-- person-period table makes the hazard a division rather than an estimation.
--
--   h_d(t) = P(exit at t by default | still at risk at t)   = defaults / at_risk
--   h_p(t) = P(exit at t by prepay  | still at risk at t)   = prepays  / at_risk
--
-- From those, three cumulative quantities, of which only one is correct in the
-- presence of a competing risk:
--
--   S(t)      = PROD (1 - h_d - h_p)        overall survival
--   F_d(t)    = SUM  S(t-1) * h_d(t)        cumulative incidence  <-- correct
--   naive(t)  = cumulative defaults / N0    counting estimator
--   km_d(t)   = 1 - PROD (1 - h_d)          Kaplan-Meier on default alone
--
-- NOT A DATA CHECK: S(t) + F_d(t) + F_p(t) = 1 is an algebraic identity of these
-- formulas, since F_d + F_p = SUM S(u-1)(h_d + h_p) = 1 - PROD(1 - h_d - h_p).
-- Verifying it numerically tests floating-point arithmetic, nothing more. It says
-- NOTHING about whether the exit taxonomy is exhaustive -- that is checked where
-- it can be, on loan_exit, by requiring every loan to carry exactly one of the
-- enumerated reasons and the counts to reconcile against the origination cohort.
--
-- WHY km_d IS WRONG HERE, and it is the standard mistake. Applying
-- Kaplan-Meier to default while treating prepayment as censoring answers a
-- counterfactual: what default would look like if refinancing were impossible.
-- It drops prepaid loans from the denominator without ever crediting them as
-- non-defaults, so it overstates. The inequality is one-directional:
--
--   F_d(t) <= km_d(t), with equality only if h_p is identically zero.
--
-- Prepayment is not censoring. A repaid loan is a real, final, observed
-- outcome that genuinely cannot default afterwards, so it belongs in the
-- denominator forever. Censoring is *missing information* -- a loan we have
-- not watched long enough. On the 2000-2008 vintages, observed through March
-- 2026, every loan is mature at 120 months, so naive(t) should agree with
-- F_d(t) closely. That agreement validates the arithmetic; it is not a
-- finding. On a live book most vintages are not mature and naive(t) breaks.
--
--   $horizon   administrative censor, months on book
-- ==========================================================================

CREATE OR REPLACE VIEW vintage_hazards AS
WITH cohorts AS (
    -- The denominator for the naive estimator, and a trap worth naming.
    --
    -- "Cohort size" sounds obvious and is not. 17% of these loans enter the
    -- panel after age 0 -- Freddie Mac acquired them a month or more after
    -- origination -- so the risk set GROWS over the first few months. Taking
    -- the age-0 count as the cohort excludes those late entrants from the
    -- denominator while still counting their defaults in the numerator, which
    -- inflates the naive rate by the ratio of the two (28% for the 2000
    -- vintage). Counting distinct loans instead is unambiguous.
    --
    -- The deeper point: under delayed entry the naive estimator has no
    -- principled denominator at all, while the cumulative incidence function
    -- never needs one because it recomputes the risk set every month.
    SELECT origination_vintage AS vintage, count(DISTINCT loan_identifier) AS cohort_size
    FROM person_period
    GROUP BY vintage
),
counts AS (
    SELECT
        origination_vintage AS vintage,
        loan_age,
        count(*) AS at_risk,
        count(*) FILTER (WHERE default_event) AS defaults,
        count(*) FILTER (WHERE prepayment_event) AS prepayments,
        -- Real censoring only: observation lost while the loan was still at risk
        -- (data cutoff, clock reset, missing month, informative exit).
        --
        -- Administrative censoring at the horizon is deliberately NOT counted. It
        -- is not something that happens to a loan; it is the observation window
        -- closing on a loan that is still performing. Counting it made every
        -- surviving loan flip into "censored" in the final month and the composition
        -- chart ended in a cliff.
        count(*) FILTER (
            WHERE is_exit_month
              AND exit_reason NOT IN ('default', 'prepaid', 'administrative_censor')
        ) AS censored
    FROM person_period
    GROUP BY vintage, loan_age
),
rates AS (
    SELECT
        *,
        defaults::DOUBLE / at_risk AS hazard_default,
        prepayments::DOUBLE / at_risk AS hazard_prepay,
        cohort_size
    FROM counts JOIN cohorts USING (vintage)
),
cumulative AS (
    SELECT
        *,
        -- Survival to the END of month t, over both causes.
        product(1.0 - hazard_default - hazard_prepay) OVER (
            PARTITION BY vintage ORDER BY loan_age ROWS UNBOUNDED PRECEDING
        ) AS survival,
        -- Kaplan-Meier on default alone, treating prepayment as censoring.
        -- Computed to be shown as wrong, not to be used.
        1.0 - product(1.0 - hazard_default) OVER (
            PARTITION BY vintage ORDER BY loan_age ROWS UNBOUNDED PRECEDING
        ) AS km_default,
        sum(defaults) OVER (
            PARTITION BY vintage ORDER BY loan_age ROWS UNBOUNDED PRECEDING
        ) AS cumulative_defaults,
        sum(prepayments) OVER (
            PARTITION BY vintage ORDER BY loan_age ROWS UNBOUNDED PRECEDING
        ) AS cumulative_prepayments,
        sum(censored) OVER (
            PARTITION BY vintage ORDER BY loan_age ROWS UNBOUNDED PRECEDING
        ) AS cumulative_censored
    FROM rates
),
lagged AS (
    SELECT
        *,
        -- S(t-1): survival to the START of month t, which is what the loan had
        -- to achieve to be at risk during it. coalesce for the first month.
        coalesce(
            lag(survival) OVER (PARTITION BY vintage ORDER BY loan_age),
            1.0
        ) AS survival_entering
    FROM cumulative
)
SELECT
    vintage,
    loan_age,
    at_risk,
    cohort_size,
    defaults,
    prepayments,
    censored,
    hazard_default,
    hazard_prepay,
    survival,
    survival_entering,
    -- Aalen-Johansen cumulative incidence: for each month, the chance of still
    -- being alive going in times the chance of defaulting during it, summed.
    sum(survival_entering * hazard_default) OVER (
        PARTITION BY vintage ORDER BY loan_age ROWS UNBOUNDED PRECEDING
    ) AS cif_default,
    sum(survival_entering * hazard_prepay) OVER (
        PARTITION BY vintage ORDER BY loan_age ROWS UNBOUNDED PRECEDING
    ) AS cif_prepay,
    -- Raw counting shares of the cohort. Used for the composition chart, because
    -- these four genuinely partition the cohort by construction, whereas mixing
    -- product-limit and counting quantities in one stack does not cohere.
    cumulative_defaults::DOUBLE / cohort_size AS naive_default,
    cumulative_prepayments::DOUBLE / cohort_size AS naive_prepay,
    cumulative_censored::DOUBLE / cohort_size AS naive_censored,
    -- The complement: the share of the cohort that has not yet left for any
    -- reason. A residual of counting quantities, which is legitimate because each
    -- loan exits exactly once, so the four shares partition the cohort exactly.
    --
    -- Not at_risk / cohort_size. A loan's exit month is still a row in
    -- person_period -- it was at risk entering that month -- so that ratio counts
    -- the exiting loans twice and the four shares summed past 1. It also starts
    -- below 1 because of delayed entry: 17% of loans are not yet observed at age 0.
    -- Delayed entry still shows here, as "not yet exited" briefly including "not
    -- yet acquired", but it is confined to the first two months on book.
    1.0 - (cumulative_defaults + cumulative_prepayments + cumulative_censored)::DOUBLE
        / cohort_size AS not_yet_exited_share,
    km_default
FROM lagged
WHERE loan_age <= $horizon;


-- --------------------------------------------------------------------------
-- The age-period-cohort view: the same curves, but carrying calendar time too.
--
-- The identification problem in one line: for a loan from vintage v at age a,
-- the calendar period is p = v + a exactly, with no error term. Age, cohort
-- and period are perfectly collinear, so no data can separate maturation from
-- vintage quality from macroeconomics. Any decomposition needs a restriction
-- imposed by assumption.
--
-- What CAN be shown without one: plot the same numbers against age and then
-- against calendar date. A shock that lines up vertically across vintages
-- sitting at different ages is a period effect, and no age-based story
-- produces it.
-- --------------------------------------------------------------------------
CREATE OR REPLACE VIEW vintage_calendar_hazards AS
SELECT
    origination_vintage AS vintage,
    date_trunc('month', period_date) AS calendar_month,
    -- Loans of one vintage year originate across four quarters, so at a given
    -- calendar month they sit at a spread of ages. Averaging over that spread is
    -- the point: it is the vintage's hazard *in that month of calendar time*.
    --
    -- Do NOT add loan_age to this GROUP BY. Within a vintage, period and age are
    -- nearly one-to-one (period = vintage + age), so grouping by both splits the
    -- cohort into cells of a handful of loans, and one default in a cell of one
    -- reads as a 100% monthly hazard. That produced a chart of spikes to 100%.
    avg(loan_age) AS mean_loan_age,
    count(*) AS at_risk,
    count(*) FILTER (WHERE default_event) AS defaults,
    count(*) FILTER (WHERE default_event)::DOUBLE / count(*) AS hazard_default
FROM person_period
WHERE loan_age <= $horizon
GROUP BY vintage, calendar_month
HAVING count(*) >= 100;  -- a hazard from fewer loans than this is noise, not signal
