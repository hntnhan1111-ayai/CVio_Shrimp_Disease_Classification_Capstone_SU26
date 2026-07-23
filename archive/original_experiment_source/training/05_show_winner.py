
from pathlib import Path
import pandas as pd
p=Path("/home/drnguyenvinh/notebooks/shrimp_yolo11s_full_method_zoo/sldra54_ev015fair200/seed_42/sldra54_seed42_results.csv")
if not p.exists(): print("[NO RESULTS]"); raise SystemExit
d=pd.read_csv(p).drop_duplicates("variant_id",keep="last"); d=d[d.status.eq("ok")].copy()
for c in ["val_mAP50","val_mAP50_95","devtest_mAP50","devtest_mAP50_95"]: d[c]=pd.to_numeric(d[c],errors="coerce")
if d.empty: print("[NO SUCCESSFUL RESULTS]"); raise SystemExit
d["d50"]=d.devtest_mAP50-.154456; d["d95"]=d.devtest_mAP50_95-.041630
d=d.sort_values(["devtest_mAP50_95","devtest_mAP50"],ascending=False)
print(d[["variant_id","family","val_mAP50","val_mAP50_95","devtest_mAP50","devtest_mAP50_95","d50","d95","best_pt"]].head(20).to_string(index=False))
w=d.iloc[0]; print("\nWINNER",f"SD{int(w.variant_id):03d}",w.family,w.devtest_mAP50,w.devtest_mAP50_95)
print("[BEATS EV015 BOTH]" if w.devtest_mAP50>.154456 and w.devtest_mAP50_95>.041630 else "[EV015 STILL WINNER]")
