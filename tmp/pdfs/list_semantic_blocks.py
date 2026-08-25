import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
data=json.loads(Path('reports/final_report_xelatex/source_data/chapters_1_2_semantic.json').read_text(encoding='utf-8'))
for chapter,pages in data.items():
    print('\n===',chapter,'===')
    for page in pages:
        print('\n-- page',page['page'],'--')
        for i,b in enumerate(page['blocks']):
            print(i, f"y={b['bbox'][1]:.1f}", f"h={b['highlight_ratio']}", b['text'])
