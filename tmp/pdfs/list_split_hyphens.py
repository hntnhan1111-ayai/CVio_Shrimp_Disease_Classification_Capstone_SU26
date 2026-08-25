import json,re,sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
d=json.loads(Path('reports/final_report_xelatex/source_data/chapters_1_2_semantic.json').read_text(encoding='utf-8'))
vals=set()
for pages in d.values():
  for p in pages:
    for b in p['blocks']:
      vals.update(m.group(0) for m in re.finditer(r"\b[\w]+-\s+[\w]+\b",b['text']))
    for t in p['tables']:
      for row in t['rows']:
        for cell in row: vals.update(m.group(0) for m in re.finditer(r"\b[\w]+-\s+[\w]+\b",cell))
for x in sorted(vals): print(x)
