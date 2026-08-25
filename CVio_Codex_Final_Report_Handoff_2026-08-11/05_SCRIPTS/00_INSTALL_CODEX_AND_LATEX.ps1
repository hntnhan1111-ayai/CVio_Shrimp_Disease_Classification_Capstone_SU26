$ErrorActionPreference = 'Continue'
Write-Host '=== CVio Codex/LaTeX environment installer ===' -ForegroundColor Cyan

function Ensure-WingetPackage($Id) {
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    Write-Host "Installing/checking $Id"
    winget install --id $Id -e --accept-package-agreements --accept-source-agreements
  } else {
    Write-Warning 'winget not found. Install the package manually.'
  }
}

Ensure-WingetPackage 'Git.Git'
Ensure-WingetPackage 'GitHub.cli'
Ensure-WingetPackage 'OpenJS.NodeJS.LTS'
Ensure-WingetPackage 'Python.Python.3.12'
Ensure-WingetPackage 'MiKTeX.MiKTeX'

# Codex CLI - official npm package route on Windows.
if (Get-Command npm -ErrorAction SilentlyContinue) {
  npm install -g @openai/codex
} else {
  Write-Warning 'npm unavailable; restart PowerShell after Node installation, then run: npm install -g @openai/codex'
}

# PDF QA utilities. Chocolatey is optional; otherwise agent should install equivalents.
if (Get-Command choco -ErrorAction SilentlyContinue) {
  choco install -y qpdf poppler ghostscript
} else {
  Write-Warning 'Chocolatey not found. For full preflight install qpdf + Poppler (pdfinfo/pdffonts/pdftoppm). The agent must verify these commands before final build.'
}

if (Get-Command python -ErrorAction SilentlyContinue) {
  python -m pip install --upgrade pip
  python -m pip install pymupdf pillow pypdf pyyaml
}

Write-Host ''
Write-Host 'After installation:' -ForegroundColor Yellow
Write-Host '1. Restart PowerShell/VS Code/Codex.'
Write-Host '2. Run codex and sign in if needed.'
Write-Host '3. Run gh auth login if gh auth status is not authenticated.'
Write-Host '4. Open MiKTeX Console and allow missing package installation if prompted.'
Write-Host '5. Verify: xelatex --version; latexmk -v; qpdf --version; pdfinfo -v; pdffonts -v; pdftoppm -v.'
