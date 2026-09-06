"""Construct the person-period table: one row per loan per month at risk.

Stage 1. The performance file is already in this shape — LOAN AGE is the time
index — so this is a reshape and an administrative censor, not an inference.

Not yet implemented.
"""
