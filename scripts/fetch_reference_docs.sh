#!/usr/bin/env bash
# Fetch Freddie Mac's public SFLD documentation into docs/freddie-mac/.
#
# These files are public and need no account, but they are Freddie Mac's
# documentation and are not ours to redistribute — so the directory is gitignored
# and this script restores it instead.
#
# The dataset itself is NOT fetched here. It sits behind Clarity Data Intelligence
# (CRT portal -> Data Download -> SFLLD) and requires registration:
#   https://claritydownload.fmapps.freddiemac.com/CRT/#/sflld
set -euo pipefail

BASE="https://www.freddiemac.com/fmac-resources/research"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/docs/freddie-mac"
mkdir -p "$DEST"

files=(
  "docs/file_headers_july_2026.zip"       # authoritative Release 47 column layout
  "docs/release-47-sample-files.zip"      # 1,000-row format illustration (NOT the vintage sample)
  "pdf/file_layout_july_2026.xlsx"
  "pdf/general_user_guide_july_2026.pdf"
  "pdf/release_notes.pdf"
  "pdf/dataset_licensing_agreement.pdf"   # the fee-based commercial licence; not the one we use
)

for f in "${files[@]}"; do
  name="$(basename "$f")"
  printf '%-38s ' "$name"
  curl -fsSL --max-time 120 -o "$DEST/$name" "$BASE/$f" && echo "ok" || { echo "FAILED"; exit 1; }
done

cd "$DEST"
unzip -oq file_headers_july_2026.zip
unzip -oq release-47-sample-files.zip
echo "Reference docs in $DEST"
