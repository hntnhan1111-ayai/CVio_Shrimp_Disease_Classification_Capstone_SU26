$ErrorActionPreference = 'Stop'
$BundleRoot = Split-Path -Parent $PSScriptRoot
$SkillSource = Join-Path $BundleRoot '04_SKILLS'
$CodexSkills = Join-Path $HOME '.codex\skills'
New-Item -ItemType Directory -Force -Path $CodexSkills | Out-Null

Get-ChildItem $SkillSource -Directory | ForEach-Object {
  $dst = Join-Path $CodexSkills $_.Name
  if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
  Copy-Item -Recurse -Force $_.FullName $dst
  Write-Host "Installed skill: $($_.Name) -> $dst" -ForegroundColor Green
}

Write-Host 'Restart Codex after skill installation. In Codex CLI use /skills or type $cvio-academic-report.' -ForegroundColor Cyan
