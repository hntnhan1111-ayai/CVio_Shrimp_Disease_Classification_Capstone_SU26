#!/usr/bin/env bash
set +e
OUT="${1:-}"
if [ -z "$OUT" ]; then echo "Usage: bash verify_v5_stage1_split.sh OUT_DIR"; exit 2; fi
/opt/miniconda3/bin/python - <<PY
from pathlib import Path
import pandas as pd
out = Path('$OUT')
cands = list(out.rglob('split_manifest.csv'))
print('split_manifest candidates:', [str(x) for x in cands[:5]])
if not cands:
    raise SystemExit('No split_manifest.csv found')
df = pd.read_csv(cands[0])
if 'split' not in df.columns:
    raise SystemExit(f'split column missing. columns={list(df.columns)}')
print('split counts:', df['split'].value_counts().to_dict())
label_col = 'class_dir' if 'class_dir' in df.columns else ('display_class_name' if 'display_class_name' in df.columns else ('label' if 'label' in df.columns else 'project_label'))
print('class counts by split:')
print(pd.crosstab(df['split'], df[label_col]))
expected = {'train':804, 'val':172, 'test':173}
actual = df['split'].value_counts().to_dict()
if actual != expected:
    raise SystemExit(f'FAIL: expected Stage1/ShrimpXNet 70/15/15 counts {expected}, got {actual}')
# duplicate path/hash checks if columns exist
for col in ['path', 'src', 'md5']:
    if col in df.columns:
        bad = []
        for v, g in df.groupby(col):
            if g['split'].nunique() > 1:
                bad.append(v)
                if len(bad) >= 5: break
        print(f'{col} crossing split examples:', bad)
        if bad:
            raise SystemExit(f'FAIL: {col} duplicates across splits')
print('PASS: fixed random stratified image-level 70/15/15 split, seed 42 style counts verified. Not group-safe.')
PY
