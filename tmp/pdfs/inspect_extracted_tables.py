import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

path = Path(r"reports/final_report_xelatex/source_data/chapters_1_2_semantic.json")
data = json.loads(path.read_text(encoding="utf-8"))
for chapter, pages in data.items():
    print("\n", chapter)
    for page in pages:
        for index, table in enumerate(page["tables"], 1):
            rows = table["rows"]
            print("page", page["page"], "table", index, "shape", table["row_count"], table["col_count"])
            for row in rows[:3]:
                print("  ", row)
            if len(rows) > 4:
                print("  ...")
                for row in rows[-2:]:
                    print("  ", row)
