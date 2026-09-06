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
    ("metropolitan_statistical_area_or_metropolitan_division", "METROPOLITAN STATISTICAL AREA (MSA) OR METROPOLITAN DIVISION"),
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
    ("underwriting_defect_and_major_servicing_defect_settlement_date", "UNDERWRITING DEFECT AND MAJOR SERVICING DEFECT SETTLEMENT DATE"),
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

#: Zero balance codes. 01 is a voluntary payoff — a competing risk, not a censoring
#: nuisance, because a loan repaid early stops paying interest.
ZERO_BALANCE_PREPAID = "01"
ZERO_BALANCE_DEFAULT = ("03", "09")  # short sale or charge-off; REO disposition

#: ``CURRENT LOAN DELINQUENCY STATUS`` is alphanumeric: "RA" means REO acquisition,
#: so this column must never be parsed as an integer.
DELINQUENCY_REO = "RA"
