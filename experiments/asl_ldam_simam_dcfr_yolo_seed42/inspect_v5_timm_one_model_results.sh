#!/usr/bin/env bash
set -u
set +e
OUT="${1:-}"
LOG="${2:-}"
ZIP="${3:-}"

echo "=== ERRORS ==="
if [ -n "$LOG" ] && [ -f "$LOG" ]; then
  grep -nEi "KeyError|FAILED|failed|Traceback|RuntimeError|ValueError|CUDA out of memory|No space left|nan|nonfinite" "$LOG" | tail -160 || true
else
  echo "missing log: $LOG"
fi

echo

echo "=== CLEAN RANKING ==="
if [ -f "$OUT/A_clean_metrics/timm_clean_ranking.csv" ]; then
  python - <<PY
import pandas as pd
p='$OUT/A_clean_metrics/timm_clean_ranking.csv'
df=pd.read_csv(p)
cols=[c for c in ['model','variant_key','loss_key','attention_key','seed','clean_macro_f1','clean_accuracy','clean_cohen_kappa','clean_fps','clean_latency_ms','params_m','model_size_mb','status'] if c in df.columns]
print(df[cols].to_string(index=False))
PY
else
  echo "missing $OUT/A_clean_metrics/timm_clean_ranking.csv"
fi

echo

echo "=== ROBUSTNESS RANKING ==="
if [ -f "$OUT/F_robustness_drops/timm_robustness_ranking.csv" ]; then
  python - <<PY
import pandas as pd
p='$OUT/F_robustness_drops/timm_robustness_ranking.csv'
df=pd.read_csv(p)
print(df.head(30).to_string(index=False))
PY
else
  echo "missing $OUT/F_robustness_drops/timm_robustness_ranking.csv"
fi

echo

echo "=== SPLIT VERIFY ==="
if [ -f "$OUT/S_split_manifests/split_manifest.csv" ]; then
  python - <<PY
import pandas as pd
p='$OUT/S_split_manifests/split_manifest.csv'
df=pd.read_csv(p)
print(df['split'].value_counts().to_dict())
print(pd.crosstab(df['split'], df['class_folder']))
PY
else
  echo "missing split manifest"
fi

echo

echo "=== ZIP ==="
if [ -n "$ZIP" ]; then ls -lh "$ZIP" 2>/dev/null || true; fi

echo

echo "=== REPORT FILES ==="
find "$OUT" -maxdepth 3 -type f \( -name "*.csv" -o -name "*.json" -o -name "*.md" \) -print | sort | tail -80 2>/dev/null || true
