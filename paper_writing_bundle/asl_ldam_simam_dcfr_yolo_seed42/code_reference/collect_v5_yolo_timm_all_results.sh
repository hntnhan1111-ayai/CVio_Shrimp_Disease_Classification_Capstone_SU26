#!/usr/bin/env bash
set -u
set +e
cd /home/drnguyenvinh/notebooks
OUT="/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_YOLO_TIMM_AGGREGATED_outputs"
ZIP="/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_YOLO_TIMM_AGGREGATED_RESULTS_ONLY_for_review.zip"
rm -rf "$OUT" "$ZIP"
mkdir -p "$OUT"

# First collect TIMM if possible.
bash /home/drnguyenvinh/notebooks/collect_v5_timm_results.sh || true
# Then collect YOLO if user's YOLO collector exists.
bash /home/drnguyenvinh/notebooks/collect_v5_all_results.sh || true

python - <<'PY'
from pathlib import Path
import pandas as pd, json, shutil
base=Path('/home/drnguyenvinh/notebooks')
out=Path('/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_YOLO_TIMM_AGGREGATED_outputs')
out.mkdir(parents=True, exist_ok=True)
rows=[]
# TIMM clean aggregate
p=base/'final_loss_cbam_top5_noise_v5_stage1split_TIMM_AGGREGATED_outputs'/'all_timm_clean_results.csv'
if p.exists():
    df=pd.read_csv(p)
    keep=[]
    for _,r in df.iterrows():
        rows.append({'backend':'timm_pytorch','model':r.get('model'),'variant_key':r.get('variant_key'),'loss_key':r.get('loss_key'),'attention_key':r.get('attention_key'),'seed':r.get('seed'),'test_macro_f1':r.get('clean_macro_f1'),'test_accuracy':r.get('clean_accuracy'),'cohen_kappa':r.get('clean_cohen_kappa'),'latency_ms':r.get('clean_latency_ms'),'fps':r.get('clean_fps'),'params_m':r.get('params_m'),'model_size_mb':r.get('model_size_mb')})
# YOLO scan compact summaries from full outputs.
for c in base.glob('final_loss_cbam_top5_noise_v5_stage1split_*_seed*_outputs/final_reports/compact_summary_table.csv'):
    if 'SMOKE' in str(c) or 'TIMM' in str(c): continue
    try:
        df=pd.read_csv(c)
        clean=df[(df.get('corruption','')=='clean') | (df.get('split_name','')=='clean')].copy()
        for _,r in clean.iterrows():
            rows.append({'backend':'native_ultralytics_yolo','model':c.parts[-3].split('stage1split_')[-1].rsplit('_seed',1)[0], 'variant_key':r.get('variant_key'), 'loss_key':r.get('loss_key'), 'attention_key':r.get('attention_key'), 'seed':c.parts[-3].rsplit('_seed',1)[-1].split('_outputs')[0], 'test_macro_f1':r.get('test_macro_f1'), 'test_accuracy':r.get('test_accuracy'), 'cohen_kappa':r.get('cohen_kappa'), 'latency_ms':r.get('latency_ms'), 'fps':r.get('fps'), 'params_m':r.get('params_m'), 'model_size_mb':r.get('model_size_mb')})
    except Exception as e:
        print('skip yolo compact',c,e)
if rows:
    allc=pd.DataFrame(rows)
    allc.to_csv(out/'all_yolo_timm_clean_results.csv', index=False)
    group_cols=['backend','model','variant_key','loss_key','attention_key']
    metrics=['test_macro_f1','test_accuracy','cohen_kappa','latency_ms','fps','params_m','model_size_mb']
    agg=allc.groupby(group_cols, as_index=False).agg(**{f'{m}_mean':(m,'mean') for m in metrics if m in allc.columns}, **{f'{m}_std':(m,'std') for m in metrics if m in allc.columns})
    agg=agg.sort_values(['test_macro_f1_mean','cohen_kappa_mean','test_accuracy_mean'], ascending=[False,False,False])
    agg.to_csv(out/'overall_yolo_timm_clean_mean_std_ranking.csv', index=False)
(out/'README.md').write_text('Overall YOLO + TIMM aggregation. YOLO rows come from native Ultralytics runner; TIMM rows from PyTorch/TIMM runner.\n', encoding='utf-8')
PY
cd "$OUT" && zip -qr "$ZIP" .
echo "ZIP=$ZIP"
ls -lh "$ZIP"
