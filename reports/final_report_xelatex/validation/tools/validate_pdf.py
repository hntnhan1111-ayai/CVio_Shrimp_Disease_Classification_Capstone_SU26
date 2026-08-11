from __future__ import annotations
import argparse, json, re, sys, statistics
from pathlib import Path
import pymupdf
from PIL import Image, ImageDraw

ap=argparse.ArgumentParser()
ap.add_argument("--pdf",required=True)
ap.add_argument("--manifest",required=True)
ap.add_argument("--bad-tokens",required=True)
ap.add_argument("--out",required=True)
ap.add_argument("--render-dir",required=True)
ap.add_argument("--dpi",type=int,default=144)
ap.add_argument("--full-thesis",action="store_true")
args=ap.parse_args()

manifest=json.loads(Path(args.manifest).read_text(encoding="utf-8"))
bad=[x.strip() for x in Path(args.bad_tokens).read_text(encoding="utf-8").splitlines() if x.strip()]
errors=[]; warnings=[]; review=[]
doc=pymupdf.open(args.pdf)
if args.full_thesis and len(doc)<manifest.get("minimum_full_thesis_pages",120):
    errors.append(f"Full thesis unexpectedly short: {len(doc)} pages")

texts=[]; records=[]; caption_pages={}
for i,p in enumerate(doc):
    txt=p.get_text("text")
    texts.append(txt)
    w,h=p.rect.width,p.rect.height
    if abs(w-595.28)>2 or abs(h-841.89)>2:
        errors.append(f"Page {i+1}: not A4 ({w:.2f}x{h:.2f})")
    for ch,name in [("\u00ad","soft hyphen"),("\u200b","zero-width space"),("\ufffd","replacement character")]:
        if ch in txt: errors.append(f"Page {i+1}: {name} in extracted text")
    for tok in bad:
        if tok in txt: errors.append(f"Page {i+1}: malformed token `{tok}`")
    if re.search(r"\[\s+\d+\]\s*,?\s*\d+\]",txt):
        errors.append(f"Page {i+1}: suspicious broken numeric citation")

    d=p.get_text("dict")
    for b in d.get("blocks",[]):
        if "lines" not in b: continue
        for line in b["lines"]:
            for s in line["spans"]:
                if s["bbox"][1]>h-42:
                    st=s["text"].strip()
                    if st and not re.fullmatch(r"(?:\d+|[ivxlcdm]+)",st,re.I):
                        errors.append(f"Page {i+1}: footer-zone junk: {st[:80]!r}")

    for b in d.get("blocks",[]):
        lines=b.get("lines",[])
        if len(lines)<4: continue
        bt=" ".join("".join(s["text"] for s in line["spans"]) for line in lines)
        if len(bt)<140: continue
        ys=[]; sizes=[]
        for line in lines:
            if not line["spans"]: continue
            ys.append(line["bbox"][1])
            sizes.extend(s["size"] for s in line["spans"] if 9.5<=s["size"]<=13.5)
        if len(ys)>=4 and sizes:
            ds=[b-a for a,b in zip(ys,ys[1:]) if b>a]
            if ds:
                ratio=statistics.median(ds)/statistics.median(sizes)
                if ratio<0.97:
                    errors.append(f"Page {i+1}: compressed body line spacing ratio {ratio:.2f}")

    blocks=[b for b in p.get_text("blocks") if str(b[4]).strip()]
    imgs=len(p.get_images(full=True)); drawings=len(p.get_drawings())
    if len(txt.strip())>500 and imgs==0 and drawings<8 and len(blocks)>=2:
        ranges=sorted((b[1],b[3]) for b in blocks)
        merged=[]
        for y0,y1 in ranges:
            if not merged or y0>merged[-1][1]: merged.append([y0,y1])
            else: merged[-1][1]=max(merged[-1][1],y1)
        gaps=[b[0]-a[1] for a,b in zip(merged,merged[1:])]
        if gaps and max(gaps)>220:
            errors.append(f"Page {i+1}: giant prose-only vertical gap {max(gaps):.1f} pt")
        elif gaps and max(gaps)>160:
            review.append(f"Page {i+1}: inspect large prose gap {max(gaps):.1f} pt")

    for fid in manifest.get("high_risk_figures",[]):
        if f"Figure {fid}" in txt: caption_pages[fid]=i
    records.append({"page":i+1,"images":imgs,"drawings":drawings,"chars":len(txt.strip())})

TEXT="\n".join(texts)
for item in manifest.get("figures",[]):
    c=TEXT.count(f"Figure {item['id']}")
    if c<item.get("min_occurrences",1):
        errors.append(f"Missing figure {item['id']}: occurrences={c}")
for item in manifest.get("tables",[]):
    c=TEXT.count(f"Table {item['id']}")
    if c<item.get("min_occurrences",1):
        errors.append(f"Missing table {item['id']}: occurrences={c}")

for fid in manifest.get("high_risk_figures",[]):
    if fid not in caption_pages:
        errors.append(f"High-risk figure caption not found: Figure {fid}")
        continue
    pi=caption_pages[fid]
    near={pi}
    if pi>0: near.add(pi-1)
    if pi+1<len(doc): near.add(pi+1)
    if not any(len(doc[j].get_images(full=True))>0 or len(doc[j].get_drawings())>=12 for j in near):
        errors.append(f"Figure {fid}: caption exists but no nearby raster/vector content detected")

needles=["Disease-ShrimpID","yolo26m_asl_ldam_simam_dcfr_fp32","yolo11n-seg.pt"]
for needle in needles:
    found=False
    for i,p in enumerate(doc):
        d=p.get_text("dict")
        for b in d.get("blocks",[]):
            for line in b.get("lines",[]):
                lt="".join(s["text"] for s in line["spans"])
                if needle in lt:
                    found=True
                    fonts={s["font"] for s in line["spans"] if s["text"].strip()}
                    if not any("Times" in f or "Tinos" in f for f in fonts):
                        errors.append(f"Page {i+1}: `{needle}` not Times-family: {sorted(fonts)}")
    if args.full_thesis and not found:
        warnings.append(f"Literal font-check target not found: {needle}")

render_dir=Path(args.render_dir); render_dir.mkdir(parents=True,exist_ok=True)
scale=args.dpi/72
thumbs=[]
for i,p in enumerate(doc):
    pix=p.get_pixmap(matrix=pymupdf.Matrix(scale,scale),alpha=False)
    path=render_dir/f"page_{i+1:04d}.png"
    pix.save(path)
    im=Image.open(path).convert("L")
    small=im.resize((max(1,im.width//8),max(1,im.height//8)))
    hist=small.histogram()
    ratio=sum(hist[:245])/(small.width*small.height)
    if i>0 and ratio<0.0015 and records[i]["chars"]<20:
        errors.append(f"Page {i+1}: suspicious blank page")
    t=Image.open(path).convert("RGB"); t.thumbnail((210,297))
    thumbs.append((i+1,t.copy()))

for start in range(0,len(thumbs),20):
    batch=thumbs[start:start+20]
    sheet=Image.new("RGB",(230*4,325*5),"white")
    draw=ImageDraw.Draw(sheet)
    for k,(n,im) in enumerate(batch):
        x=(k%4)*230+10; y=(k//4)*325+18
        sheet.paste(im,(x,y)); draw.text((x,y-15),f"p{n}",fill="black")
    sheet.save(render_dir/f"contact_{start+1:04d}_{start+len(batch):04d}.jpg",quality=88)

result={
    "status":"PASS" if not errors and not review else "FAIL",
    "errors":errors,
    "warnings":warnings,
    "manual_review_required":review,
    "pages":len(doc),
    "renders":len(doc),
    "caption_pages":{k:v+1 for k,v in caption_pages.items()}
}
Path(args.out).write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
sys.exit(0 if result["status"]=="PASS" else 2)
