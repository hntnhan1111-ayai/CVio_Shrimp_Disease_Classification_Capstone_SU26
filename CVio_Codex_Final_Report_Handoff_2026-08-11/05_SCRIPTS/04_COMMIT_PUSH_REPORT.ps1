param([string]$Message='report: rebuild final thesis in XeLaTeX with validation gates')
$ErrorActionPreference='Stop'
$branch = (git branch --show-current).Trim()
if (!$branch) { throw 'Not on a Git branch.' }
if ($branch -notlike 'report/*') { throw "Refusing to push unexpected branch: $branch" }

gh auth status

git status --short
Write-Host 'Review the diff before continuing. This script assumes validation reports show PASS.' -ForegroundColor Yellow
$confirm = Read-Host 'Type PUSH to stage, commit, and push'
if ($confirm -ne 'PUSH') { throw 'Cancelled.' }

git add reports/final_report_xelatex AGENTS.md
git commit -m $Message
git push -u origin $branch
Write-Host "Pushed $branch" -ForegroundColor Green
