import json,sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
d=json.loads(Path('reports/final_report_xelatex/source_data/chapters_1_2_semantic.json').read_text(encoding='utf-8'))
chapter=sys.argv[1]; page=int(sys.argv[2])
for i,b in enumerate(d[chapter][page-1]['blocks']): print(f'{i}: h={b["highlight_ratio"]} {b["text"]}')
