param([string]$ReportDir='reports/final_report_xelatex')
$ErrorActionPreference='Stop'
Set-Location $ReportDir

latexmk -xelatex -interaction=nonstopmode -file-line-error -halt-on-error main.tex

$log = Get-Content main.log -Raw
$bad = @('Undefined control sequence','Emergency stop','Fatal error','Overfull \\hbox','Overfull \\vbox','Float too large','Too many unprocessed floats','There were undefined references','multiply defined','destination with the same identifier')
foreach ($p in $bad) { if ($log -match $p) { throw "LaTeX hard gate failed: $p" } }

$Pdf = 'main.pdf'
if (!(Test-Path $Pdf)) { throw 'main.pdf missing' }
qpdf --check $Pdf | Tee-Object -FilePath validation/qpdf_check.txt
pdfinfo $Pdf | Tee-Object -FilePath validation/pdfinfo.txt
pdffonts $Pdf | Tee-Object -FilePath validation/pdffonts.txt

python scripts/validate_pdf.py $Pdf --out validation/pdf_validation.json
python scripts/validate_text_tokens.py $Pdf --out validation/text_token_validation.json

Remove-Item -Recurse -Force validation/render -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force validation/render | Out-Null
pdftoppm -png -r 180 $Pdf validation/render/page

Write-Host 'Automated gates complete. FULL visual inspection of every rendered page is still mandatory before PASS.' -ForegroundColor Yellow
