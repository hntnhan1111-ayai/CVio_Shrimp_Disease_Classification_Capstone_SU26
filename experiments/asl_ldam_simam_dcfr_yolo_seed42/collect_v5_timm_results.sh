#!/usr/bin/env bash
set -u
set +e
cd /home/drnguyenvinh/notebooks
OUT="/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_TIMM_AGGREGATED_outputs"
ZIP="/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_TIMM_AGGREGATED_RESULTS_ONLY_for_review.zip"
rm -rf "$OUT" "$ZIP"
mkdir -p "$OUT"
python - <<'PY'
from pathlib import Path
import pandas as pd, json, shutil, subprocess, os
base=Path('/home/drnguyenvinh/notebooks')
out=Path('/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_TIMM_AGGREGATED_outputs')
out.mkdir(parents=True, exist_ok=True)
clean=[]; noise=[]; fails=[]
for d in base.glob('final_loss_cbam_top5_noise_v5_stage1split_TIMM_*_outputs'):
    if 'SMOKE' in d.name or 'AGGREGATED' in d.name: continue
    p=d/'A_clean_metrics'/'timm_clean_results_all_methods.csv'
    if p.exists(): clean.append(pd.read_csv(p))
    n=d/'E_noise_metrics'/'timm_noise_metrics_all.csv'
    if n.exists(): noise.append(pd.read_csv(n))
    for fp in (d/'K_status_failures').glob('*failed.json'):
        try: fails.append(json.loads(fp.read_text()))
        except Exception: pass
if clean:
    df=pd.concat(clean, ignore_index=True)
    df.to_csv(out/'all_timm_clean_results.csv', index=False)
    # mean +- std by model/method
    metric_cols=['clean_macro_f1','clean_accuracy','clean_cohen_kappa','clean_latency_ms','clean_fps','params_m','model_size_mb']
    group_cols=['backend','model','variant_key','loss_key','attention_key','method_family']
    agg=df.groupby(group_cols, as_index=False).agg(**{f'{m}_mean':(m,'mean') for m in metric_cols if m in df.columns}, **{f'{m}_std':(m,'std') for m in metric_cols if m in df.columns})
    sort_cols=[c for c in ['clean_macro_f1_mean','clean_cohen_kappa_mean','clean_accuracy_mean','clean_latency_ms_mean'] if c in agg.columns]
    asc=[False,False,False,True][:len(sort_cols)]
    agg=agg.sort_values(sort_cols, ascending=asc) if sort_cols else agg
    agg.to_csv(out/'timm_clean_mean_std_ranking.csv', index=False)
if noise:
    nd=pd.concat(noise, ignore_index=True)
    nd.to_csv(out/'all_timm_noise_metrics.csv', index=False)
    rb=nd.groupby(['backend','model','variant_key','loss_key','attention_key','seed'], as_index=False).agg(mean_corrupted_macro_f1=('test_macro_f1','mean'), worst_corrupted_macro_f1=('test_macro_f1','min'), mean_drop_macro_f1=('drop_macro_f1_vs_own_clean','mean'), mean_accuracy=('test_accuracy','mean'), mean_kappa=('cohen_kappa','mean'))
    rb.to_csv(out/'timm_robustness_by_seed.csv', index=False)
    rb2=rb.groupby(['backend','model','variant_key','loss_key','attention_key'], as_index=False).agg(mean_corrupted_macro_f1_mean=('mean_corrupted_macro_f1','mean'), mean_corrupted_macro_f1_std=('mean_corrupted_macro_f1','std'), worst_corrupted_macro_f1_mean=('worst_corrupted_macro_f1','mean'), mean_drop_macro_f1_mean=('mean_drop_macro_f1','mean'), mean_kappa_mean=('mean_kappa','mean')).sort_values(['mean_corrupted_macro_f1_mean','worst_corrupted_macro_f1_mean','mean_kappa_mean'], ascending=[False,False,False])
    rb2.to_csv(out/'timm_robustness_mean_std_ranking.csv', index=False)
if fails:
    pd.DataFrame(fails).to_csv(out/'timm_failures.csv', index=False)
(out/'README.md').write_text('TIMM aggregated results. Combine with YOLO aggregated CSV for overall paper ranking.\n', encoding='utf-8')
PY
cd "$OUT" && zip -qr "$ZIP" .
echo "ZIP=$ZIP"
ls -lh "$ZIP"
