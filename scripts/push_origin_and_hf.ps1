# Push the current branch (or a named branch) to GitHub (origin) and Hugging Face Space (hf).
# One-time: git remote add hf https://huggingface.co/spaces/YOUR_USER/YOUR_SPACE.git
param(
    [string] $Branch = ""
)
$ErrorActionPreference = "Stop"
if (-not $Branch) {
    $Branch = (git rev-parse --abbrev-ref HEAD).Trim()
}
$remotes = @(git remote)
if ($remotes -notcontains "hf") {
    Write-Host "No git remote named 'hf'. Add it, for example:" -ForegroundColor Yellow
    Write-Host "  git remote add hf https://huggingface.co/spaces/YOUR_USER/YOUR_SPACE.git"
    exit 1
}
git push origin $Branch
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# HF Hub often rejects a normal `git push hf main` when old commits contain blocked binaries (PDFs).
# Use a single-commit snapshot so the Space only sees the current tree.
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$scriptDir/push_hf_space.ps1" $Branch
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Pushed branch '$Branch' to origin and hf (snapshot)."
