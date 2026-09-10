#!/usr/bin/env bash
# Fetch the Treasury benchmark yield used as the risk-free leg of the discount rate.
#
# WHY THIS EXISTS. The discount rate for valuing a loan's cash flows is the lender's
# required return, and it is nowhere in the Freddie Mac data — that carries the
# *borrower's* note rate, which is a different thing. A single flat assumption is worse
# than no assumption here: vintage mean note rates run from 5.51% (2003) to 8.13%
# (2000), so discounting every cohort at one rate makes 2000 look wonderful and 2003
# look negative before a single default, contaminating every cross-vintage comparison.
#
# So the risk-free leg is taken per vintage from the Treasury curve, and only the spread
# above it stays an assumption:
#
#     r(vintage) = Treasury yield at origination + funding & required-return spread
#
# DGS10 (10-year) because cash flows here are administratively censored at 120 months,
# so a ten-year benchmark is the duration match. THIS MUST TRACK
# horizon.max_loan_age_months: discounting ten years of cash flows at a five-year yield
# misprices the term structure. An earlier version defaulted to DGS5 for a 60-month
# horizon and was not updated when the horizon moved, which broke reproduction from a
# clean clone — the fetch produced treasury_dgs5.csv while the config asked for
# treasury_dgs10.csv, and every command touching the discount rate then failed.
#
# UNLIKE THE FREDDIE MAC DATA, THIS FILE IS COMMITTED. FRED series are US federal
# government work and in the public domain, so this is the one external dataset the repo
# may redistribute — which makes the pipeline reproducible with no network and no
# account. /data/* is gitignored with an explicit negation for /data/benchmark/.
#
# Source: Federal Reserve Bank of St. Louis (FRED). No API key required.
#   https://fred.stlouisfed.org/series/DGS10
set -euo pipefail

SERIES="${1:-DGS10}"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/benchmark"
mkdir -p "$DEST"

target="$DEST/treasury_$(echo "$SERIES" | tr '[:upper:]' '[:lower:]').csv"
printf '%-34s ' "$SERIES"
if curl -fsSL --max-time 60 -o "$target" \
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=${SERIES}"; then
  rows=$(( $(wc -l < "$target") - 1 ))
  echo "ok  ($rows observations -> ${target#"$DEST/"})"
else
  echo "FAILED"
  exit 1
fi

# FRED marks non-trading days with "." rather than an empty field; the loader coerces
# those to null. Fail loudly here if the file does not look like what we expect.
head -1 "$target" | grep -q "observation_date,${SERIES}" || {
  echo "unexpected header in $target — FRED may have changed its CSV format" >&2
  exit 1
}
echo "Benchmark rates in $DEST"
