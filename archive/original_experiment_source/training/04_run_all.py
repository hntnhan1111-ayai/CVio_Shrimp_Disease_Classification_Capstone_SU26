
import argparse,csv,json,subprocess,sys
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument("--start-id",type=int,default=0); ap.add_argument("--end-id",type=int,default=53)
ap.add_argument("--epochs",type=int,default=200); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--device",default="0")
ap.add_argument("--pretrain",default="/home/drnguyenvinh/notebooks/shrimp_yolo11s_full_method_zoo/baseline_reuse_lock/yolo11s_baseline_best.pt")
ap.add_argument("--data",default="/home/drnguyenvinh/notebooks/shrimp_yolo_canonical_bg_wssv_2cls_split70_15_15/data.yaml")
ap.add_argument("--out",default="/home/drnguyenvinh/notebooks/shrimp_yolo11s_full_method_zoo/sldra54_ev015fair200")
a=ap.parse_args(); out=Path(a.out); result=out/f"seed_{a.seed}"/f"sldra54_seed{a.seed}_results.csv"
done=set()
if result.exists():
    d=pd.read_csv(result); d["variant_id"]=pd.to_numeric(d["variant_id"],errors="coerce")
    done=set(d[d.status.eq("ok")].variant_id.dropna().astype(int))
for r in csv.DictReader((P/"SLDRA54_MANIFEST.csv").open()):
    v=int(r["variant_id"])
    if not a.start_id<=v<=a.end_id: continue
    if v in done: print("[SKIP DONE]",v,flush=True); continue
    cfg=P/"configs"/f"yolo11s_SD{v:03d}_{r['family']}.yaml"
    cmd=[sys.executable,"-u",str(P/"03_single_run.py"),"--variant-id",str(v),"--family",r["family"],
      "--yaml",str(cfg),"--pretrain",a.pretrain,"--data",a.data,"--out",a.out,
      "--epochs",str(a.epochs),"--seed",str(a.seed),"--device",a.device]
    print("="*110,flush=True); print(f"[RUN START] SD{v:03d} {r['family']}",flush=True)
    p=subprocess.Popen(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=1)
    for line in p.stdout: print(line,end="",flush=True)
    rc=p.wait(); rid=f"SD{v:03d}_{r['family']}_EV015FAIR_200E_SEED{a.seed}"
    jp=out/f"seed_{a.seed}"/"per_run_json"/f"{rid}.json"
    row=json.loads(jp.read_text()) if jp.exists() else {"variant_id":v,"family":r["family"],"status":"fail","error":f"exit={rc}"}
    pd.DataFrame([row]).to_csv(result,mode="a",header=not result.exists(),index=False)
