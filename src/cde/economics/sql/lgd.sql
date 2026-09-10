-- ==========================================================================
-- Loss given default, measured rather than assumed.
--
-- Freddie Mac publishes ACTUAL LOSS on the disposition record, so LGD is one of
-- the few parameters in this project that does not have to be guessed. It is
-- populated for 90-97% of zero-balance codes 02 (third party sale), 03 (short
-- sale or charge-off), 09 (REO disposition) and 15 (whole loan sale).
--
-- TWO EVENTS, NOT ONE, AND THEY MUST BE KEPT APART.
--
-- Default in this project is *first passage to 90+ days past due*. A loss is
-- only realised much later, at disposition, and only for some of those loans:
-- roughly two-thirds cure, are modified, or otherwise never produce a loss. So the
-- severity that belongs in the NPV identity is not the disposition LGD:
--
--     E[loss | 90+ DPD]  =  P(loss disposition | 90+ DPD)  x  E[LGD | disposition]
--
-- Pairing a 90+ DPD default definition with a disposition-conditional LGD would
-- roughly double-count severity on every loan in the book.
--
-- WHY VALUE-WEIGHTED AND NEVER THE MEAN OF RATIOS.
--
-- The mean of per-loan (loss / exposure) is 11.55 on this data against a median
-- of ~0.37, because foreclosure costs and accrued interest can dwarf a small
-- remaining balance and a material share of dispositions exceed 100% of it. The aggregate
-- that belongs in a pricing calculation is SUM(loss) / SUM(exposure), which is
-- immune to that.
--
-- WHICH EXPOSURE. The NPV identity applies severity to B(t), the balance in the
-- month of default, so severity is measured against that balance -- not against
-- the balance at disposition, which is the conventional denominator. The two
-- differ by under a percentage point on this data, but consistency
-- with the formula matters more than convention.
--
--   $vintages  the cohorts to estimate on
-- ==========================================================================

-- A single read-only query. Unlike person_period.sql and vintage_curves.sql this
-- creates nothing: LGD is a measurement, not a pipeline stage, so it must be
-- runnable against a read-only connection.
WITH population AS (
    -- Loans that reached the default event inside the horizon.
    SELECT
        loan_identifier,
        min(loan_age) AS default_age,
        -- Balance in the month of first 90+ DPD. Guarded with nullif: 716 loans
        -- report a non-positive balance in that month, and dividing by it gave an
        -- infinite severity ratio on the first attempt.
        nullif(arg_min(current_actual_upb, loan_age), 0) AS exposure_at_default
    FROM person_period
    WHERE default_event
      AND origination_vintage IN $vintages
    GROUP BY loan_identifier
),
losses AS (
    -- Realised losses at disposition, from the FULL performance history rather than
    -- person_period: disposition routinely lands well past the 120-month horizon.
    SELECT
        loan_identifier,
        actual_loss,
        nullif(zero_balance_removal_upb, 0) AS exposure_at_disposition
    FROM performance
    WHERE zero_balance_code IN ('02', '03', '09')
      AND origination_vintage IN $vintages
      -- ACTUAL LOSS is null for loans carrying a Defect Settlement Date and for
      -- those disposed within three months of the performance cutoff, so severity
      -- is estimated on a subset. The shortfall is reported, not hidden.
      AND actual_loss IS NOT NULL
),
joined AS (
    SELECT
        p.loan_identifier,
        p.exposure_at_default,
        l.actual_loss,
        l.exposure_at_disposition
    FROM population AS p
    LEFT JOIN losses AS l USING (loan_identifier)
)
SELECT
    count(*) AS reached_default,
    count(actual_loss) AS reached_loss_disposition,
    count(*) FILTER (WHERE actual_loss IS NOT NULL AND exposure_at_default IS NOT NULL)
        AS usable_for_severity,
    count(actual_loss)::DOUBLE / count(*) AS probability_of_loss,
    sum(actual_loss) AS total_loss,
    sum(exposure_at_default) FILTER (WHERE actual_loss IS NOT NULL)
        AS total_exposure_at_default,
    -- Severity on the basis the NPV identity uses: the balance in the month of
    -- default, which is what h_d(t) * B(t) * (1 - LGD) applies severity to.
    sum(actual_loss) FILTER (WHERE exposure_at_default IS NOT NULL)
        / sum(exposure_at_default) FILTER (WHERE actual_loss IS NOT NULL)
        AS lgd_at_default_balance,
    -- ...and on the conventional basis, for comparison with published figures.
    sum(actual_loss) FILTER (WHERE exposure_at_disposition IS NOT NULL)
        / sum(exposure_at_disposition) FILTER (WHERE actual_loss IS NOT NULL)
        AS lgd_at_disposition,
    median(actual_loss / exposure_at_disposition) AS lgd_median,
    quantile_cont(actual_loss / exposure_at_disposition, 0.10) AS lgd_p10,
    quantile_cont(actual_loss / exposure_at_disposition, 0.90) AS lgd_p90
FROM joined;
