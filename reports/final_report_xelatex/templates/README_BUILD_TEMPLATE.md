# Build and validation

Record tool versions and Git commit.

Build:
```powershell
.\scripts\03_BUILD_REPORT.ps1
```

Validate:
```powershell
.\scripts\04_VALIDATE_REPORT.ps1 -FullThesis
```

Do not call the PDF final unless all generated validation JSON files report PASS and manual high-risk visual review is complete.
