#!/bin/bash
# ---------------------------------------------------------------------------
# .pack_zenodo.sh — bundle the heavy artefacts (checkpoints + predictions)
# into a single zip suitable for upload to a Zenodo deposit. Includes an
# auto-generated MANIFEST.txt with SHA-256 of every file, plus a README
# pointing back to the GitHub repository.
#
# Output:  ../flowgat_paper_zenodo_<YYYYMMDD>.zip   (next to release/)
#
# Notes:
#   - The expected upload size is ~8 GB (1.7 GB checkpoints + 6.3 GB
#     predictions). Comfortably under the 50 GB Zenodo per-record limit.
#   - We zip with -0 (store, no compression) because NPZ is already
#     deflate-compressed; recompressing yields ~3 % savings at large CPU
#     cost. Switch to -9 if you want maximum compression.
# ---------------------------------------------------------------------------

set -euo pipefail

RELEASE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$RELEASE_DIR/.." && pwd)"
STAGE="$REPO_ROOT/.zenodo_stage"
DATE="$(date +%Y%m%d)"
OUT="$REPO_ROOT/flowgat_paper_zenodo_${DATE}.zip"

cd "$REPO_ROOT"

if [[ ! -d results/checkpoints ]] && [[ ! -d results/predictions ]]; then
    echo "[zenodo] no checkpoints/predictions to pack -- nothing to do" >&2
    exit 1
fi

rm -rf "$STAGE"
mkdir -p "$STAGE"

# --- 1. copy heavy artefacts via hardlinks (fast, zero disk cost) --------
for sub in checkpoints predictions; do
    if [[ -d "results/$sub" ]]; then
        cp -al "results/$sub" "$STAGE/$sub"
    fi
done

# --- 2. write MANIFEST.txt with SHA-256 of every file --------------------
echo "[zenodo] computing SHA-256 over $(find "$STAGE" -type f | wc -l) files..."
(
    cd "$STAGE"
    find . -type f \! -name MANIFEST.txt -print0 \
        | sort -z \
        | xargs -0 sha256sum \
        > MANIFEST.txt
)

# --- 3. minimal Zenodo-side README ---------------------------------------
cat > "$STAGE/README.md" <<EOF
# flowgat_paper Zenodo deposit (${DATE})

This is the heavy-artefacts companion to the GitHub repository
[<TBD URL>](https://github.com/<TBD>/flowgat-paper).

Contents:

- \`checkpoints/<variant>/seed_<N>/best.pt\` — PyTorch state dicts (48 files
  across four flow domains × four feature variants × three seeds).
- \`predictions/<variant>/seed_<N>/<split>_<case>.npz\` — per-frame velocity
  prediction dumps used by every downstream diagnostic in the paper.
- \`MANIFEST.txt\` — SHA-256 of every file in this deposit.

To consume:

\`\`\`bash
git clone https://github.com/<TBD>/flowgat-paper.git
cd flowgat-paper/release/
unzip ../../flowgat_paper_zenodo_${DATE}.zip -d ../
# Now results/checkpoints/ and results/predictions/ are in place.
# Re-run the post-training diagnostics:
bash jobs/SUBMIT_E7_FOLLOWUP.sh
\`\`\`

License: CC-BY-4.0 for the trained weights and prediction dumps.
The patient-specific aortic geometries used in training remain
© Vascular Model Repository (vascularmodel.com); the synthetic
U-bend CFD cohort remains © newCFD_dataset (polybox.ethz.ch).
EOF

# --- 4. zip ---------------------------------------------------------------
echo "[zenodo] zipping -> $OUT"
( cd "$STAGE" && zip -r0 "$OUT" . )

# --- 5. cleanup hardlink stage --------------------------------------------
rm -rf "$STAGE"

echo "[zenodo] done."
ls -lh "$OUT"
