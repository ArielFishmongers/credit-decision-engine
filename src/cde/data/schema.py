"""Freddie Mac Single-Family Loan-Level Dataset — Release 47 (July 2026) schema.

Generated from the authoritative headers in ``file_headers_july_2026.zip``, not from
any user guide, tutorial or public repository. Release 47 renamed columns and moved two
of them between files, so anything published before August 2026 has the wrong layout:

    CREDIT SCORE            -> CLASSIC FICO
    LOAN SEQUENCE NUMBER    -> LOAN IDENTIFIER
    MONTHLY REPORTING PERIOD-> PERIOD
    NET SALE PROCEEDS       -> NET SALES PROCEEDS
    ACTUAL LOSS CALCULATION -> ACTUAL LOSS

``SERVICER NAME`` and ``MORTGAGE INSURANCE CANCELLATION INDICATOR`` moved from the
origination file to the performance file; ``VANTAGESCORE 4.0`` and ``BANKRUPTCY
CRAMDOWN COSTS`` are new.

Files are pipe-delimited with **no header row**, so these orderings are load-bearing.
"""

from __future__ import annotations

RELEASE = 47

# (snake_case name, official Release 47 column name), in file order.
ORIGINATION_COLUMNS: list[tuple[str, str]] = [
    ("classic_fico", "CLASSIC FICO"),
    ("first_payment_date", "FIRST PAYMENT DATE"),
    ("first_time_homebuyer_indicator", "FIRST TIME HOMEBUYER INDICATOR"),
    ("maturity_date", "MATURITY DATE"),
    ("metropolitan_statistical_area_or_metropolitan_division", "METROPOLITAN STATISTICAL AREA (MSA) OR METROPOLITAN DIVISION"),  # noqa: E501 (verbatim Freddie Mac column name)
    ("mortgage_insurance_percentage", "MORTGAGE INSURANCE PERCENTAGE (MI %)"),
    ("number_of_units", "NUMBER OF UNITS"),
    ("occupancy_status", "OCCUPANCY STATUS"),
    ("original_combined_loan_to_value", "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)"),
    ("original_debt_to_income_ratio", "ORIGINAL DEBT-TO-INCOME (DTI) RATIO"),
    ("original_upb", "ORIGINAL UPB"),
    ("original_loan_to_value", "ORIGINAL LOAN-TO-VALUE (LTV)"),
    ("original_interest_rate", "ORIGINAL INTEREST RATE"),
    ("channel", "CHANNEL"),
    ("prepayment_penalty_indicator", "PREPAYMENT PENALTY INDICATOR"),
    ("amortization_type", "AMORTIZATION TYPE"),
    ("property_state", "PROPERTY STATE"),
    ("property_type", "PROPERTY TYPE"),
    ("postal_code", "POSTAL CODE"),
    ("loan_identifier", "LOAN IDENTIFIER"),
    ("loan_purpose", "LOAN PURPOSE"),
    ("original_loan_term", "ORIGINAL LOAN TERM"),
    ("number_of_borrowers", "NUMBER OF BORROWERS"),
    ("seller_name", "SELLER NAME"),
    ("super_conforming_flag", "SUPER CONFORMING FLAG"),
    ("pre_harp_loan_sequence_number", "PRE-HARP LOAN SEQUENCE NUMBER"),
    ("special_eligibility_program", "SPECIAL ELIGIBILITY PROGRAM"),
    ("harp_indicator", "HARP INDICATOR"),
    ("property_valuation_method", "PROPERTY VALUATION METHOD"),
    ("interest_only_indicator", "INTEREST ONLY (I/O) INDICATOR"),
    ("vantagescore_4_0", "VANTAGESCORE 4.0"),
]

PERFORMANCE_COLUMNS: list[tuple[str, str]] = [
    ("loan_identifier", "LOAN IDENTIFIER"),
    ("period", "PERIOD"),
    ("current_actual_upb", "CURRENT ACTUAL UPB"),
    ("current_loan_delinquency_status", "CURRENT LOAN DELINQUENCY STATUS"),
    ("loan_age", "LOAN AGE"),
    ("remaining_months_to_legal_maturity", "REMAINING MONTHS TO LEGAL MATURITY"),
    ("underwriting_defect_and_major_servicing_defect_settlement_date", "UNDERWRITING DEFECT AND MAJOR SERVICING DEFECT SETTLEMENT DATE"),  # noqa: E501 (verbatim Freddie Mac column name)
    ("modification_flag", "MODIFICATION FLAG"),
    ("zero_balance_code", "ZERO BALANCE CODE"),
    ("zero_balance_effective_date", "ZERO BALANCE EFFECTIVE DATE"),
    ("current_interest_rate", "CURRENT INTEREST RATE"),
    ("current_non_interest_bearing_upb", "CURRENT NON-INTEREST BEARING UPB"),
    ("due_date_of_last_paid_installment", "DUE DATE OF LAST PAID INSTALLMENT (DDLPI)"),
    ("mi_recoveries", "MI RECOVERIES"),
    ("net_sales_proceeds", "NET SALES PROCEEDS"),
    ("non_mi_recoveries", "NON MI RECOVERIES"),
    ("total_expenses", "TOTAL EXPENSES"),
    ("legal_costs", "LEGAL COSTS"),
    ("maintenance_and_preservation_costs", "MAINTENANCE AND PRESERVATION COSTS"),
    ("taxes_and_insurance", "TAXES AND INSURANCE"),
    ("miscellaneous_expenses", "MISCELLANEOUS EXPENSES"),
    ("actual_loss", "ACTUAL LOSS"),
    ("cumulative_modification_costs", "CUMULATIVE MODIFICATION COSTS"),
    ("interest_rate_step_indicator", "INTEREST RATE STEP INDICATOR"),
    ("payment_deferral_flag", "PAYMENT DEFERRAL FLAG"),
    ("estimated_loan_to_value", "ESTIMATED LOAN-TO-VALUE (ELTV)"),
    ("zero_balance_removal_upb", "ZERO BALANCE REMOVAL UPB"),
    ("delinquent_accrued_interest", "DELINQUENT ACCRUED INTEREST"),
    ("delinquency_due_to_disaster", "DELINQUENCY DUE TO DISASTER"),
    ("borrower_assistance_plan", "BORROWER ASSISTANCE PLAN"),
    ("current_period_modification_costs", "CURRENT PERIOD MODIFICATION COSTS"),
    ("current_interest_bearing_upb", "CURRENT INTEREST BEARING UPB"),
    ("mortgage_insurance_cancellation_indicator", "MORTGAGE INSURANCE CANCELLATION INDICATOR"),
    ("servicer_name", "SERVICER NAME"),
    ("bankruptcy_cramdown_costs", "BANKRUPTCY CRAMDOWN COSTS"),
]

ORIGINATION_NAMES = [name for name, _ in ORIGINATION_COLUMNS]
PERFORMANCE_NAMES = [name for name, _ in PERFORMANCE_COLUMNS]

#: Join key present in both files.
LOAN_KEY = "loan_identifier"

# ---------------------------------------------------------------------------
# Enumerations.
#
# The column lists above come from ``file_headers_july_2026.zip``. The header files
# carry names only, so the enumerations below come from the other authoritative
# Release 47 artefact — ``general_user_guide_july_2026.pdf``, sections "Zero Balance
# Code", "Current Loan Delinquency Status", "Modification Flag" and "Actual Loss".
# Not from a tutorial, and not from a pre-July-2026 guide.
# ---------------------------------------------------------------------------

#: Every ``ZERO BALANCE CODE`` in Release 47. A loan is removed from the panel when one
#: is set, and is never updated afterwards except in the case of defects.
ZERO_BALANCE_CODES: dict[str, str] = {
    "01": "Prepaid or Matured (Voluntary Payoff)",
    "02": "Third Party Sale",
    "03": "Short Sale or Charge Off",
    "09": "REO Disposition",
    "15": "Whole Loan sales",
    "16": "Reperforming loan securitizations",
    "96": "Confirmed Underwriting Defect or Major Servicing Defect prior to credit event",
}

#: Voluntary payoff — a competing risk, not a censoring nuisance, because a loan repaid
#: early stops paying interest, and that is a real term in the NPV.
ZERO_BALANCE_PREPAID = "01"

#: Borrower credit events. ``02`` is a foreclosure-auction disposition and belongs here:
#: Freddie Mac computes ``ACTUAL LOSS`` for 02, 03, 09 and 15, which is their own
#: statement that these are credit events. Omitting it undercounts defaults.
#: ``15`` is excluded — a whole loan sale is Freddie Mac disposing of the asset, not the
#: borrower failing. The modelling choice is owned by ``configs/default.yaml``; this is
#: the documented default.
ZERO_BALANCE_DEFAULT = ("02", "03", "09")

#: Exits correlated with a loan's credit state, which makes the censoring they induce
#: informative. ``16`` is the clearest case: a *reperforming* loan is by construction one
#: that was previously delinquent. There is no correction for this on this data — the
#: volume gets reported. See PROJECT_PLAN.md section 8.
ZERO_BALANCE_INFORMATIVE_EXIT = ("15", "16", "96")

#: ``CURRENT LOAN DELINQUENCY STATUS`` is alphanumeric and must never be parsed as an
#: integer. Numeric codes are zero-padded counts of missed payments, capped at 99:
#: "00" = current or < 30 days, "01" = 30-59 days, "02" = 60-89, "03" = 90-119, and so on.
DELINQUENCY_REO = "RA"  #: REO acquisition — reported instead of a day count.
DELINQUENCY_NOT_AVAILABLE = "XX"

#: Non-numeric values of ``CURRENT LOAN DELINQUENCY STATUS``, which is why a threshold
#: test cannot be written as a bare string comparison. Both "RA" and "XX" sort above
#: "03" lexically; "RA" *is* a default state and "XX" is missing data, so the two must be
#: separated explicitly rather than swept up by ``status >= threshold``.
DELINQUENCY_NON_NUMERIC = (DELINQUENCY_REO, DELINQUENCY_NOT_AVAILABLE)

#: Values of ``MODIFICATION FLAG`` that reset ``LOAN AGE``, because loan age is then
#: measured from the Modification First Payment Date. Payment deferrals are *not*
#: modifications and do not reset it — that is ``PAYMENT DEFERRAL FLAG``.
MODIFICATION_FLAGS_RESETTING_LOAN_AGE = ("Y", "P")

#: Sentinel values meaning "not available", verified per-attribute against the July 2026
#: user guide. These must become NULL before any arithmetic touches the column, and they
#: are the single most common source of silent nonsense in public work on this dataset:
#: a mean DTI computed without clearing 999 is wrong by a wide margin, and nothing
#: crashes to tell you.
NOT_AVAILABLE_SENTINELS: dict[str, str] = {
    # origination
    "classic_fico": "9999",  # values < 300 or > 850 disclosed as not available
    "vantagescore_4_0": "9999",
    "first_time_homebuyer_indicator": "9",
    "mortgage_insurance_percentage": "999",
    "number_of_units": "99",
    "occupancy_status": "9",
    "original_combined_loan_to_value": "999",
    "original_debt_to_income_ratio": "999",  # ratios > 65% disclosed as not available
    "original_loan_to_value": "999",
    "channel": "9",
    "property_type": "99",
    "loan_purpose": "9",
    "number_of_borrowers": "99",
    "postal_code": "000",  # "000 = Unknown"; three-digit prefix, so 000 is not a real one
    "property_valuation_method": "7",
    # performance
    "estimated_loan_to_value": "999",
    "mortgage_insurance_cancellation_indicator": "7",
    "current_loan_delinquency_status": DELINQUENCY_NOT_AVAILABLE,
}


# ---------------------------------------------------------------------------
# Types.
#
# The files are read as text first and cast explicitly afterwards. Letting a CSV
# reader sniff types on this dataset is how leading zeros vanish from postal codes,
# "RA" crashes an integer parse, and 999 becomes a debt-to-income ratio of 999%.
#
# Money is DOUBLE rather than DECIMAL: this is a modelling pipeline, not a ledger,
# and DECIMAL round-trips into Python as ``decimal.Decimal``, which scikit-learn
# will not accept without another conversion.
# ---------------------------------------------------------------------------

#: Storage type per column, keyed by snake_case name. VARCHAR is deliberate wherever
#: the value is a code rather than a quantity.
COLUMN_TYPES: dict[str, str] = {
    # --- origination ---
    "classic_fico": "SMALLINT",
    "first_payment_date": "VARCHAR",
    "first_time_homebuyer_indicator": "VARCHAR",
    "maturity_date": "VARCHAR",
    "metropolitan_statistical_area_or_metropolitan_division": "VARCHAR",
    "mortgage_insurance_percentage": "SMALLINT",
    "number_of_units": "SMALLINT",
    "occupancy_status": "VARCHAR",
    "original_combined_loan_to_value": "SMALLINT",
    "original_debt_to_income_ratio": "SMALLINT",
    "original_upb": "DOUBLE",
    "original_loan_to_value": "SMALLINT",
    "original_interest_rate": "DOUBLE",
    "channel": "VARCHAR",
    "prepayment_penalty_indicator": "VARCHAR",
    "amortization_type": "VARCHAR",
    "property_state": "VARCHAR",
    "property_type": "VARCHAR",
    # The first three digits of the ZIP only, so "056" is a prefix and its leading zero
    # is meaningful. Text, not a number.
    "postal_code": "VARCHAR",
    "loan_identifier": "VARCHAR",
    "loan_purpose": "VARCHAR",
    "original_loan_term": "SMALLINT",
    "number_of_borrowers": "SMALLINT",
    "seller_name": "VARCHAR",
    "super_conforming_flag": "VARCHAR",
    "pre_harp_loan_sequence_number": "VARCHAR",
    "special_eligibility_program": "VARCHAR",
    "harp_indicator": "VARCHAR",
    "property_valuation_method": "VARCHAR",
    "interest_only_indicator": "VARCHAR",
    "vantagescore_4_0": "SMALLINT",
    # --- performance ---
    "period": "VARCHAR",
    "current_actual_upb": "DOUBLE",
    "current_loan_delinquency_status": "VARCHAR",  # alphanumeric; see DELINQUENCY_*
    "loan_age": "INTEGER",
    "remaining_months_to_legal_maturity": "INTEGER",
    "underwriting_defect_and_major_servicing_defect_settlement_date": "VARCHAR",
    "modification_flag": "VARCHAR",
    "zero_balance_code": "VARCHAR",
    "zero_balance_effective_date": "VARCHAR",
    "current_interest_rate": "DOUBLE",
    "current_non_interest_bearing_upb": "DOUBLE",
    "due_date_of_last_paid_installment": "VARCHAR",
    "mi_recoveries": "DOUBLE",
    "net_sales_proceeds": "DOUBLE",
    "non_mi_recoveries": "DOUBLE",
    "total_expenses": "DOUBLE",
    "legal_costs": "DOUBLE",
    "maintenance_and_preservation_costs": "DOUBLE",
    "taxes_and_insurance": "DOUBLE",
    "miscellaneous_expenses": "DOUBLE",
    "actual_loss": "DOUBLE",
    "cumulative_modification_costs": "DOUBLE",
    "interest_rate_step_indicator": "VARCHAR",
    "payment_deferral_flag": "VARCHAR",
    "estimated_loan_to_value": "SMALLINT",
    "zero_balance_removal_upb": "DOUBLE",
    "delinquent_accrued_interest": "DOUBLE",
    "delinquency_due_to_disaster": "VARCHAR",
    "borrower_assistance_plan": "VARCHAR",
    "current_period_modification_costs": "DOUBLE",
    "current_interest_bearing_upb": "DOUBLE",
    "mortgage_insurance_cancellation_indicator": "VARCHAR",
    "servicer_name": "VARCHAR",
    "bankruptcy_cramdown_costs": "DOUBLE",
}

#: Columns holding a YYYYMM month. Kept as VARCHAR in storage and exposed as a
#: first-of-month DATE alongside, because YYYYMM sorts correctly as text but cannot
#: be subtracted, and month arithmetic is needed all over stage 1.
MONTH_COLUMNS: frozenset[str] = frozenset(
    {
        "first_payment_date",
        "maturity_date",
        "period",
        "zero_balance_effective_date",
        "underwriting_defect_and_major_servicing_defect_settlement_date",
        "due_date_of_last_paid_installment",
    }
)
