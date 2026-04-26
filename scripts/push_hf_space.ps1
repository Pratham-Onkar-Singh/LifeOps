# Push a single-commit snapshot of BRANCH to Hugging Face Space remote `hf` as `main`.
# Hugging Face rejects normal pushes when reachable history contains blocked binaries (e.g. PDFs in old commits).
# This orphan snapshot matches your current tree (.gitignore still excludes *.pdf from the index).
param(
    [string] $Branch = "main"
)
$ErrorActionPreference = "Stop"
$remotes = @(git remote)
if ($remotes -notcontains "hf") {
    Write-Host "No git remote named 'hf'." -ForegroundColor Yellow
    exit 1
}

$prev = (git rev-parse --abbrev-ref HEAD).Trim()
if ($prev -eq "HEAD") {
    Write-Host "Detached HEAD: checkout a branch first." -ForegroundColor Yellow
    exit 1
}

$tmp = "hf-space-snap-" + [guid]::NewGuid().ToString("N").Substring(0, 10)
git rev-parse --verify "refs/heads/$Branch" 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Branch '$Branch' does not exist." -ForegroundColor Yellow
    exit 1
}

Write-Host "Building orphan snapshot from '$Branch' -> hf:main ..."
git checkout $Branch
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (git branch --list $tmp) {
    git branch -D $tmp
}
git checkout --orphan $tmp
if ($LASTEXITCODE -ne 0) { git checkout $prev; exit $LASTEXITCODE }

git rm -rf --cached . 2>$null | Out-Null
git checkout $Branch -- .
if ($LASTEXITCODE -ne 0) {
    git checkout $prev
    if (git branch --list $tmp) { git branch -D $tmp }
    exit $LASTEXITCODE
}

git add -A
$short = (git rev-parse --short $Branch).Trim()
git commit -m "HF Space deploy snapshot (from $Branch @ $short)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Commit failed (nothing to commit?)." -ForegroundColor Yellow
    git checkout $prev
    if (git branch --list $tmp) { git branch -D $tmp }
    exit 1
}

git push hf "${tmp}:main" --force
$pushCode = $LASTEXITCODE

git checkout $prev
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (git branch --list $tmp) {
    git branch -D $tmp
    if ($LASTEXITCODE -ne 0) { Write-Host "Warning: could not delete local temp branch $tmp" -ForegroundColor Yellow }
}

if ($pushCode -ne 0) { exit $pushCode }
Write-Host "Pushed snapshot to hf main (Space will rebuild)."
