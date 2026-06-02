#!/bin/bash
# ---------------------------------------------------------------------------
# .build.sh — rebuild the release/ directory from the canonical source
# tree at $REPO_ROOT.  Idempotent: re-running this script overwrites
# everything under release/ except git metadata.
#
# Assumes the script lives inside release/, i.e.  $REPO_ROOT/release/.build.sh.
#
# What it stages (mirrors PUBLICATION_MANIFEST.md, section A):
#   - paper/main.tex, paper/refs.bib, paper/cover_letter.tex, paper/figures/
#   - src/  (excluding __pycache__)
#   - jobs/*.sh
#   - configs/*.yaml
#   - results/manifest.json
#   - results/per_seed/, diagnostics/, bootstrap/, stratified/, figures/
#   - env files, docs/, STRATEGY, PUBLICATION_MANIFEST
#
# What it does NOT touch in release/:
#   - .git/, .github/, .gitignore, README.md, LICENSE, CITATION.cff,
#     environment.yml, environment.lock.yml, requirements.txt,
#     pyproject.toml, Makefile, scripts/, tests/, .build.sh
#     (these are repo-native, not derived)
# ---------------------------------------------------------------------------

set -euo pipefail

RELEASE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$RELEASE_DIR/.." && pwd)"

echo "[build] repo root  = $REPO_ROOT"
echo "[build] release dir = $RELEASE_DIR"

if [[ "$RELEASE_DIR" == "$REPO_ROOT" ]]; then
    echo "[build] refusing to operate when release_dir == repo_root" >&2
    exit 2
fi

cd "$REPO_ROOT"

# --- 1. paper sources -----------------------------------------------------
mkdir -p "$RELEASE_DIR/paper/figures"
cp paper/main.tex          "$RELEASE_DIR/paper/"
cp paper/refs.bib          "$RELEASE_DIR/paper/"
cp paper/cover_letter.tex  "$RELEASE_DIR/paper/"
if [[ -d paper/figures ]]; then
    rsync -a paper/figures/ "$RELEASE_DIR/paper/figures/"
fi

# --- 2. code -------------------------------------------------------------
mkdir -p "$RELEASE_DIR/src"
rsync -a --exclude='__pycache__/' --exclude='*.pyc' \
    src/ "$RELEASE_DIR/src/"

# --- 3. jobs + configs ----------------------------------------------------
mkdir -p "$RELEASE_DIR/jobs" "$RELEASE_DIR/configs"
cp jobs/*.sh        "$RELEASE_DIR/jobs/"
cp configs/*.yaml   "$RELEASE_DIR/configs/"

# --- 4. results (text-only) -----------------------------------------------
mkdir -p "$RELEASE_DIR/results"
cp results/manifest.json "$RELEASE_DIR/results/"
for sub in per_seed diagnostics bootstrap stratified figures; do
    if [[ -d "results/$sub" ]]; then
        rsync -a --delete "results/$sub/" "$RELEASE_DIR/results/$sub/"
    fi
done

# --- 5. docs --------------------------------------------------------------
mkdir -p "$RELEASE_DIR/docs/archive"
cp STRATEGY_SR.md "$RELEASE_DIR/docs/"
if [[ -d docs/archive ]]; then
    cp docs/archive/*.md "$RELEASE_DIR/docs/archive/" 2>/dev/null || true
fi
cp PUBLICATION_MANIFEST.md "$RELEASE_DIR/"

# --- 6. report ------------------------------------------------------------
echo "[build] done.  release/ size:"
du -sh --apparent-size "$RELEASE_DIR"
echo "[build] verify with:  git -C $RELEASE_DIR status"
