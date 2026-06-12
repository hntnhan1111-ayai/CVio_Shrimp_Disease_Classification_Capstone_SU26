#!/usr/bin/env bash
set +e
cd /home/drnguyenvinh/notebooks
OUT_ROOT="/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_AGGREGATED"
ZIP="/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_AGGREGATED_RESULTS_ONLY_for_review.zip"
rm -rf "$OUT_ROOT" "$ZIP"
mkdir -p "$OUT_ROOT/collected_reports"
python - <<'PY'
from pathlib import Path
import pandas as pd, json, shutil, zipfile, os
root = Path('/home/drnguyenvinh/notebooks')
out_root = Path('/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_AGGREGATED')
rows=[]
for p in root.glob('final_loss_cbam_top5_noise_v5_stage1split_*_seed*_outputs/final_reports/compact_summary_table.csv'):
    try:
        df=pd.read_csv(p)
        run_dir=p.parents[1]
        df['run_output_dir']=str(run_dir)
        rows.append(df)
        dst=out_root/'collected_reports'/run_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for q in p.parent.glob('*'):
            if q.is_file() and q.suffix.lower() in ['.csv','.json','.xlsx','.md','.txt','.log']:
                shutil.copy2(q,dst/q.name)
    except Exception as e:
        print('WARN',p,e)
if rows:
    all_df=pd.concat(rows,ignore_index=True)
    all_df.to_csv(out_root/'all_compact_summary_rows.csv',index=False)
    if {'variant_key','model','test_macro_f1','cohen_kappa','test_accuracy','corruption','severity','status'}.issubset(all_df.columns):
        clean=all_df[(all_df['status'].eq('evaluated')) & (all_df['corruption'].eq('clean'))].copy()
        clean.to_csv(out_root/'clean_rows_all_models_seeds.csv',index=False)
        group_cols=['model','variant_key','loss_key','attention_key']
        rank=clean.groupby(group_cols,dropna=False).agg(mean_macro_f1=('test_macro_f1','mean'),std_macro_f1=('test_macro_f1','std'),mean_kappa=('cohen_kappa','mean'),mean_accuracy=('test_accuracy','mean'),n=('test_macro_f1','count')).reset_index().sort_values(['mean_macro_f1','mean_kappa','mean_accuracy'],ascending=[False,False,False])
        rank.to_csv(out_root/'clean_mean_std_ranking_all_models_seeds.csv',index=False)
        noise=all_df[(all_df['status'].eq('evaluated')) & (~all_df['corruption'].isin(['clean']))].copy()
        noise.to_csv(out_root/'noise_rows_all_models_seeds.csv',index=False)
        if len(noise):
            robust=noise.groupby(group_cols,dropna=False).agg(mean_noise_macro_f1=('test_macro_f1','mean'),worst_noise_macro_f1=('test_macro_f1','min'),mean_noise_kappa=('cohen_kappa','mean'),n=('test_macro_f1','count')).reset_index().sort_values(['mean_noise_macro_f1','worst_noise_macro_f1'],ascending=[False,False])
            robust.to_csv(out_root/'noise_robustness_ranking_all_models_seeds.csv',index=False)
else:
    (out_root/'NO_RESULTS_FOUND.txt').write_text('No per-run compact_summary_table.csv files found.\n')
# index
files=[str(x.relative_to(out_root)) for x in out_root.rglob('*') if x.is_file()]
(out_root/'ARTIFACT_INDEX.md').write_text('# Aggregated V5 Artifact Index\n\n'+'\n'.join(f'- {f}' for f in files)+'\n')
# zip
zip_path=Path('/home/drnguyenvinh/notebooks/final_loss_cbam_top5_noise_v5_stage1split_AGGREGATED_RESULTS_ONLY_for_review.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for f in out_root.rglob('*'):
        if f.is_file(): z.write(f,f.relative_to(out_root.parent))
print('ZIP=',zip_path)
print('SIZE_MB=',zip_path.stat().st_size/1024/1024)
PY
ls -lh "$ZIP"
