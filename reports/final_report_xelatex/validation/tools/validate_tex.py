from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument("--root",required=True)
ap.add_argument("--bad-tokens",required=True)
ap.add_argument("--out",required=True)
args=ap.parse_args()

root=Path(args.root)
errors=[]; warnings=[]
tex_files=[p for p in root.rglob("*.tex") if "build" not in p.parts and "artifacts" not in p.parts]
if not tex_files: errors.append("No .tex files found")
bad=[x.strip() for x in Path(args.bad_tokens).read_text(encoding="utf-8").splitlines() if x.strip()]
labels={}; all_text=""

for p in tex_files:
    t=p.read_text(encoding="utf-8",errors="replace")
    all_text+="\n"+t
    rel=str(p.relative_to(root))
    for ch,name in [("\u00ad","soft hyphen"),("\u200b","zero-width space"),("\ufffd","replacement character")]:
        if ch in t: errors.append(f"{rel}: contains {name}")
    if "$$" in t: errors.append(f"{rel}: raw $$ display math forbidden")
    if re.search(r"\\begin\{eqnarray\*?\}",t): errors.append(f"{rel}: eqnarray forbidden")
    if re.search(r"\\includepdf(?:set)?|\\begin\{pdfpages\}",t): errors.append(f"{rel}: pdfpages body assembly forbidden")
    for tok in bad:
        if tok in t: errors.append(f"{rel}: malformed token found: {tok}")
    if re.search(r"\\(?:texttt|verb)\b[^{|]*(?:Disease-ShrimpID|yolo26m_asl|yolo11n-seg)",t,re.I):
        errors.append(f"{rel}: known filename/pattern rendered as monospace")
    if re.search(r"\\(?:tiny|scriptsize)\b",t):
        errors.append(f"{rel}: tiny/scriptsize workaround forbidden")
    for m in re.finditer(r"\\vspace\*?\{\s*-\s*([0-9.]+)\s*(em|cm|mm|pt)",t):
        val=float(m.group(1)); unit=m.group(2)
        if (unit=="em" and val>0.8) or (unit=="cm" and val>0.25) or (unit=="mm" and val>2.5) or (unit=="pt" and val>7):
            errors.append(f"{rel}: large negative vspace {m.group(0)}")
    for lab in re.findall(r"\\label\{([^}]+)\}",t):
        labels.setdefault(lab,[]).append(rel)
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",t):
        if "\\" in g or "#" in g: continue
        candidates=[p.parent/g,root/g]
        exts=["",".pdf",".png",".jpg",".jpeg",".eps"]
        if not any(Path(str(c)+e).exists() for c in candidates for e in exts):
            errors.append(f"{rel}: includegraphics asset not found: {g}")

for lab,files in labels.items():
    if len(files)>1: errors.append(f"Duplicate label {lab}: {files}")

for item in [r"\setmainfont{Times New Roman}",r"\raggedbottom"]:
    if item not in all_text: errors.append(f"Required report style command missing: {item}")
if "\\urlstyle{same}" not in all_text:
    warnings.append("Expected \\urlstyle{same} not found; filename typography must be manually verified.")

result={"status":"PASS" if not errors else "FAIL","errors":errors,"warnings":warnings,"tex_files":len(tex_files)}
Path(args.out).write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
sys.exit(0 if not errors else 2)
