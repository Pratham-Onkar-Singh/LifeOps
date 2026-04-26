#!/usr/bin/env bash
# Push a single-commit snapshot of BRANCH to Hugging Face Space remote `hf` as `main`.
# HF rejects pushes when history contains blocked binaries (e.g. old PDF commits).
set -euo pipefail
BRANCH="${1:-main}"
if ! git remote get-url hf >/dev/null 2>&1; then
  echo "No git remote named 'hf'." >&2
  exit 1
fi
prev="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$prev" == "HEAD" ]]; then
  echo "Detached HEAD: checkout a branch first." >&2
  exit 1
fi

tmp="hf-space-snap-$(openssl rand -hex 5 2>/dev/null || echo $$)"
echo "Building orphan snapshot from '$BRANCH' -> hf:main ..."

git checkout "$BRANCH"
git branch -D "$tmp" 2>/dev/null || true
git checkout --orphan "$tmp"
git rm -rf --cached . 2>/dev/null || true
git checkout "$BRANCH" -- .
git add -A
short="$(git rev-parse --short "$BRANCH")"
git commit -m "HF Space deploy snapshot (from $BRANCH @ $short)"
git push hf "${tmp}:main" --force
git checkout "$prev"
git branch -D "$tmp"
echo "Pushed snapshot to hf main (Space will rebuild)."
