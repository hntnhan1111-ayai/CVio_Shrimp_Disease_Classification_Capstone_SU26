import argparse, json, re
from pathlib import Path
import fitz

ap=argparse.ArgumentParser(); ap.add_argument('pdf'); ap.add_argument('--out',required=True); a=ap.parse_args()
doc=fitz.open(a.pdf)
fail=[]
for i,p in enumerate(doc):
    r=p.rect
    if abs(r.width-595.28)>3 or abs(r.height-841.89)>3:
        fail.append({'page':i+1,'kind':'not_A4','size':[r.width,r.height]})
    words=p.get_text('words')
    for w in words:
        x0,y0,x1,y1,txt=w[:5]
        if x0 < -0.5 or y0 < -0.5 or x1 > r.width+0.5 or y1 > r.height+0.5:
            fail.append({'page':i+1,'kind':'object_outside_page','text':txt,'bbox':[x0,y0,x1,y1]})
        if y0 > r.height-38 and not re.fullmatch(r'(?:\d+|[ivxlcdm]+)',txt.lower()):
            fail.append({'page':i+1,'kind':'non_page_number_in_footer_zone','text':txt,'bbox':[x0,y0,x1,y1]})
    if '9A' in p.get_text(): fail.append({'page':i+1,'kind':'legacy_9A'})
status='PASS' if not fail else 'FAIL'
Path(a.out).write_text(json.dumps({'status':status,'pages':len(doc),'failures':fail},indent=2),encoding='utf-8')
print(status, len(doc), 'pages', len(fail), 'failures')
raise SystemExit(0 if status=='PASS' else 2)
