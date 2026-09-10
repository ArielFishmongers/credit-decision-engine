-- ==========================================================================
-- REALISED net present value per loan: what the loan actually paid.
--
-- WHY THIS EXISTS, AND WHY IT IS NOT OPTIONAL.
--
-- Stage 3 values a loan under the model's own hazards. That is the right input
-- to a DECISION -- it is all a lender knows at the application -- but it is
-- useless as an EVALUATION, and using it as one would invert the project's own
-- conclusions.
--
-- The measurement that proves it: on the 2007-2008 book the uncalibrated model
-- assesses portfolio value at $$3,396 a loan and the honestly recalibrated one
-- at $$2,046. Score each arm on its own expected NPV and the OVERCONFIDENT model
-- wins by $$1,349, because a model that under-predicts default believes every
-- loan is profitable. The ablation in PROJECT_PLAN.md section 5.5 as originally
-- specified would have rewarded exactly the error stage 3 exists to expose.
--
-- So: the decision comes from predictions, the score comes from cash. This
-- query is the cash.
--
-- SOURCE IS `performance`, NOT `person_period`, AND THAT IS LOAD-BEARING.
-- person_period is truncated at first passage to 90+ DPD, so for a defaulted
-- loan it stops while the borrower still owes an average of $$184,587 and says
-- nothing about what happened next -- and 63% of those loans cure and carry on
-- paying. Realised cash flow therefore has to follow the full history.
--
-- CALENDAR ORDER, NEVER LOAN AGE. Modification resets LOAN AGE, so a loan can
-- traverse the same age twice and `ORDER BY loan_age` interleaves two different
-- clocks. This is the bug that produced 29,747 duplicate default events in
-- stage 1; the month index here is built by row_number() over PERIOD.
--
-- THE CASH FLOWS.
--
--   interest(t)  = UPB(t) * note_rate(t) / 12
--   principal(t) = UPB(t) - UPB(t+1)
--   servicing(t) = UPB(t) * servicing_bps / 10000 / 12
--
-- Principal falls out of the observed balance path, which is what makes this
-- work: a payoff appears as the balance dropping to zero, a curtailment as an
-- extra-large drop, and capitalised arrears after a modification as a NEGATIVE
-- principal payment. That last case is real -- 46,019 of 6.93M transitions on
-- the test book, 1,077 of them over 1% -- and it must not be clipped, because
-- no cash arrived in those months.
--
-- A disposition also drives the balance to zero, which would otherwise book the
-- whole balance as recovered. ACTUAL LOSS is subtracted in that month to leave
-- the genuine recovery. Freddie Mac reports it as a positive number for a loss.
--
-- TERMINAL VALUE, at par, matching stage 3's convention exactly so the two are
-- comparable: whatever balance survives to the horizon is booked there. It is
-- wrong in a known direction -- a loan paying above the discount rate is worth
-- more than par -- and it is the same wrongness on both sides of every
-- comparison. See PROJECT_PLAN.md section 8 limitation 10.
--
--   $vintages         the cohorts to value
--   $horizon          months on book to value over
--   $servicing_bps    annual servicing cost, basis points of balance
--   $discount_rates   VALUES list of (vintage, annual_rate)
-- ==========================================================================
WITH discount AS (
    SELECT * FROM (VALUES $discount_rates) AS t(vintage, annual_rate)
),
sequenced AS (
    -- Month index by CALENDAR order, so a modification that resets LOAN AGE
    -- cannot interleave two clocks.
    SELECT
        loan_identifier,
        origination_vintage,
        period_date,
        loan_age,
        current_actual_upb                                AS upb,
        current_interest_rate / 100.0 / 12.0              AS monthly_note_rate,
        zero_balance_code,
        actual_loss,
        row_number() OVER (PARTITION BY loan_identifier ORDER BY period_date) - 1
                                                          AS month_index,
        min(loan_age) OVER (PARTITION BY loan_identifier) AS first_observed_age
    FROM performance
    WHERE origination_vintage IN $vintages
),
stepped AS (
    -- The next month's balance is taken from the FULL history, before the
    -- horizon filter, deliberately. Filtering first leaves lead() null in month
    -- 119 and the terminal value then falls back to the balance ENTERING that
    -- month rather than the one leaving it -- a month of amortisation booked
    -- twice. Small, and stage 3 books B(T+1), so this is also the difference
    -- between the two sides of every comparison being exactly consistent or
    -- only nearly so.
    SELECT
        *,
        lead(upb) OVER (PARTITION BY loan_identifier ORDER BY period_date) AS next_upb,
        first_value(upb) OVER (PARTITION BY loan_identifier ORDER BY period_date)
                                                                           AS advanced_upb
    FROM sequenced
),
eligible AS (
    -- Only loans observed from origination. An NPV is a value at the decision
    -- point, and a loan Freddie Mac bought at age 6 has no decision point on
    -- record. Same restriction stage 3 applies, so the two populations match.
    SELECT
        *,
        max(month_index) OVER (PARTITION BY loan_identifier) AS last_index
    FROM stepped
    WHERE first_observed_age = 0 AND month_index < $horizon
),
flows AS (
    SELECT
        s.loan_identifier,
        s.origination_vintage,
        s.month_index,
        power(1.0 + (power(1.0 + d.annual_rate, 1.0 / 12.0) - 1.0), -(s.month_index + 1))
                                                                    AS discount_factor,
        s.upb * s.monthly_note_rate                                 AS interest,
        -- On the final observed month inside the horizon there is no next row.
        -- If the loan is still alive its balance is booked as a terminal value
        -- below rather than as principal, so principal is zero here.
        coalesce(s.upb - s.next_upb, 0.0)                            AS principal,
        s.upb * $servicing_bps / 10000.0 / 12.0                      AS servicing,
        coalesce(s.actual_loss, 0.0)                                 AS loss,
        s.upb                                                        AS upb,
        s.next_upb                                                   AS next_upb,
        s.month_index = s.last_index                                 AS is_last,
        s.advanced_upb,
        s.zero_balance_code
    FROM eligible AS s
    JOIN discount AS d ON d.vintage = s.origination_vintage
)
SELECT
    loan_identifier,
    origination_vintage,
    count(*)                                                    AS months_valued,
    sum(interest * discount_factor)                             AS pv_interest,
    sum(principal * discount_factor)                            AS pv_principal,
    -sum(loss * discount_factor)                                AS pv_loss,
    -sum(servicing * discount_factor)                           AS pv_servicing,
    -- Terminal value: the balance still outstanding in the last month valued,
    -- booked at par at that point. Zero for a loan that has already paid off or
    -- been disposed, because its balance is already zero.
    sum(CASE WHEN is_last THEN coalesce(next_upb, upb) * discount_factor ELSE 0.0 END)
                                                                AS pv_terminal,
    max(CASE WHEN is_last THEN coalesce(next_upb, upb) ELSE 0.0 END)
                                                                AS terminal_balance,
    sum(loss)                                                   AS realised_loss,
    max(month_index)                                            AS last_month_index,
    -- The cash actually advanced, which is the first balance ever reported. Returned
    -- alongside the discounted legs so the exact identity below can be asserted:
    --
    --     sum(principal, UNDISCOUNTED) + terminal_balance == advanced_upb
    --
    -- by telescoping, for every loan with no exception. It holds to $$0.0000 across
    -- 129,449 test-book loans and it is the single check that says this reconstruction
    -- of the cash flows is right rather than merely plausible.
    --
    -- It does NOT hold against ORIGINAL UPB, and chasing that discrepancy is how the
    -- reason surfaced: Freddie Mac rounds ORIGINAL UPB to the nearest $$1,000 as a
    -- disclosure measure. 99.54% of loans sit within that, mean -$$347 and about $$577
    -- of spread, with 595 further out and one loan $$463,000 below its reported original.
    -- The NPV below still subtracts ORIGINAL UPB, because that is the field the model
    -- sees and using it keeps the realised and expected sides exactly comparable; the
    -- rounding is mean-zero noise worth +/-$$1.60 on a portfolio mean of 129,000 loans.
    max(advanced_upb)                                           AS advanced_upb,
    sum(principal)                                              AS principal_undiscounted
FROM flows
GROUP BY 1, 2;
