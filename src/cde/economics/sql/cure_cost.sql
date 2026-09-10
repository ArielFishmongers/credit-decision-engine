-- ==========================================================================
-- What the NPV throws away when a defaulted loan cures.
--
-- Default in this project is first passage to 90+ days past due, and 63% of
-- those loans never reach a loss disposition: they catch up, are modified into
-- performance, or leave the panel by a route carrying no recorded loss.
--
-- The NPV identity in PROJECT_PLAN.md section 5.4 books B(t) * (1 - LGD) in the
-- month of default and stops the loan's cash flow there. For a loan that cures,
-- that is the treatment a PREPAYMENT gets -- balance back, nothing further --
-- and it is wrong, because the loan goes on paying. Since the note rate on this
-- book exceeds the discount rate, those continued payments are worth more than
-- the balance, so the identity UNDERSTATES the value of every cured loan.
--
-- This query returns the population and the loan terms; the discounting is done
-- in Python, in cde.economics.lgd.cure_opportunity_cost, because it needs the
-- vintage discount rates and the same amortisation functions the NPV uses --
-- reimplementing either in SQL is how two sets of numbers drift apart.
--
-- WHY THE LOSS SIDE IS DEFINED THE SAME WAY AS IN lgd.sql. A loan counts as
-- having produced a loss only if it has a priced disposition (codes 02, 03, 09
-- with ACTUAL LOSS populated). That is the same population lgd.sql measures
-- severity on, so `reached_default - had_loss` here equals the cure count
-- implied by lgd.sql's probability_of_loss. If the two ever disagree, one of
-- them has been edited without the other.
--
--   $vintages  the cohorts to measure on
-- ==========================================================================
WITH defaults AS (
    SELECT
        loan_identifier,
        origination_vintage,
        -- Calendar order is irrelevant here: default_event fires once, and
        -- person_period is already truncated at the exit.
        min(loan_age) AS default_age,
        arg_min(original_upb, loan_age) AS original_upb,
        arg_min(original_interest_rate, loan_age) AS original_interest_rate,
        arg_min(original_loan_term, loan_age) AS original_loan_term
    FROM person_period
    WHERE default_event
      AND origination_vintage IN $vintages
    GROUP BY 1, 2
),
priced_losses AS (
    -- Read from the FULL performance history, not person_period: disposition
    -- routinely lands well past the 120-month horizon.
    SELECT DISTINCT loan_identifier
    FROM performance
    WHERE zero_balance_code IN ('02', '03', '09')
      AND actual_loss IS NOT NULL
      AND origination_vintage IN $vintages
)
SELECT
    d.loan_identifier,
    d.origination_vintage,
    d.default_age,
    d.original_upb,
    d.original_interest_rate,
    d.original_loan_term,
    (l.loan_identifier IS NOT NULL) AS had_loss_disposition
FROM defaults AS d
LEFT JOIN priced_losses AS l USING (loan_identifier);
