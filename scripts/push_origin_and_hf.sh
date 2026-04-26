#!/usr/bin/env bash
# Push the current branch (or a named branch) to GitHub (origin) and Hugging Face Space (hf).
# One-time: git remote add hf https://huggingface.co/spaces/YOUR_USER/YOUR_SPACE.git
set -euo pipefail
BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
if ! git remote get-url hf >/dev/null 2>&1; then
  echo "No git remote named 'hf'. Add it, for example:" >&2
  echo "  git remote add hf https://huggingface.co/spaces/YOUR_USER/YOUR_SPACE.git" >&2
  exit 1
fi
git push origin "$BRANCH"
git push hf "$BRANCH"
echo "Pushed branch '$BRANCH' to origin and hf."
