param(
  [string]$RepoDir = 'D:\CVio\CVio_Shrimp_Disease_Classification_Capstone_SU26',
  [string]$RepoUrl = 'https://github.com/hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26.git',
  [string]$BaseBranch = 'paper/asl-ldam-simam-dcfr-yolo-noisy-shrimp',
  [string]$NewBranch = 'report/final-xelatex-council-repair-2026-08-11'
)
$ErrorActionPreference='Stop'
if (!(Test-Path $RepoDir)) {
  git clone $RepoUrl $RepoDir
}
Set-Location $RepoDir
git fetch --all --prune
if (git show-ref --verify --quiet "refs/remotes/origin/$BaseBranch") {
  git switch -C $BaseBranch "origin/$BaseBranch"
} else {
  Write-Warning "Base branch $BaseBranch not found; using current/default branch."
}
$branch=$NewBranch
$i=2
while (git show-ref --verify --quiet "refs/heads/$branch") { $branch="$NewBranch-v$i"; $i++ }
git switch -c $branch
Write-Host "Prepared branch: $branch" -ForegroundColor Green
Write-Host 'Copy AGENTS.md and handoff bundle into a non-source temp/reference location as needed; do not commit huge evidence archives unless intentionally requested.'
