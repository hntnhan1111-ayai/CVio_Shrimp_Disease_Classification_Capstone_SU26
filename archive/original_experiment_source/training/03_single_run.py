
import argparse,gc,json,random,time
from pathlib import Path
import numpy as np, torch
from ultralytics import YOLO
def clean():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def load_ev015_fair_pretrain(target_yolo, checkpoint):
    """Load direct tensors and remap C3k2 weights into layers 4/6/8 block wrappers."""
    source_yolo = YOLO(checkpoint)

    source_state = source_yolo.model.state_dict()
    target_state = target_yolo.model.state_dict()

    loadable = {}
    direct_count = 0
    wrapper_count = 0

    for target_key, target_tensor in target_state.items():
        candidates = [target_key]

        for layer_id in (4, 6, 8):
            wrapper_prefix = f"model.{layer_id}.block."

            if target_key.startswith(wrapper_prefix):
                original_key = (
                    f"model.{layer_id}."
                    + target_key[len(wrapper_prefix):]
                )
                candidates.insert(0, original_key)

        selected_key = None

        for source_key in candidates:
            if source_key not in source_state:
                continue

            if source_state[source_key].shape != target_tensor.shape:
                continue

            selected_key = source_key
            break

        if selected_key is None:
            continue

        loadable[target_key] = source_state[selected_key].detach().clone()

        if selected_key == target_key:
            direct_count += 1
        else:
            wrapper_count += 1

    incompatible = target_yolo.model.load_state_dict(
        loadable,
        strict=False,
    )

    total_target = len(target_state)

    print(
        f"[FAIR PRETRAIN] direct={direct_count} "
        f"wrapper_remap={wrapper_count} "
        f"total={len(loadable)}/{total_target}",
        flush=True,
    )

    print(
        f"[FAIR PRETRAIN] remaining_missing="
        f"{len(incompatible.missing_keys)}",
        flush=True,
    )

    del source_yolo
    clean()

    if wrapper_count < 20:
        raise RuntimeError(
            "Wrapper remapping is unexpectedly low: "
            f"{wrapper_count}. Refusing unfair training."
        )

    if len(loadable) < 390:
        raise RuntimeError(
            "Too few pretrained tensors were loaded: "
            f"{len(loadable)}/{total_target}"
        )

    return direct_count, wrapper_count, len(loadable)

def metrics(m,p):
    a=list(m.box.maps)
    return {f"{p}_mAP50":float(m.box.map50),f"{p}_mAP50_95":float(m.box.map),
      f"{p}_precision":float(m.box.mp),f"{p}_recall":float(m.box.mr),
      f"{p}_AP_BG_mAP50_95":float(a[0]),f"{p}_AP_WSSV_mAP50_95":float(a[1])}
ap=argparse.ArgumentParser()
for x in ["variant-id","family","yaml","pretrain","data","out"]: ap.add_argument("--"+x,required=True)
ap.add_argument("--epochs",type=int,default=200); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--device",default="0")
a=ap.parse_args(); v=int(a.variant_id); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
out=Path(a.out)/f"seed_{a.seed}"; rid=f"SD{v:03d}_{a.family}_EV015FAIR_200E_SEED{a.seed}"
row={"variant_id":v,"family":a.family,"status":"fail","best_pt":"","error":""}; t=time.time()
try:
    m = YOLO(a.yaml)
    m.load(a.pretrain)

    if not getattr(m, "ckpt", None):
        raise RuntimeError(
            "YOLO checkpoint bookkeeping was not initialized. "
            "Refusing possible train-from-scratch execution."
        )

    load_ev015_fair_pretrain(m, a.pretrain)
    m.train(data=a.data,pretrained=True,epochs=a.epochs,patience=0,imgsz=1280,batch=2,workers=8,
      device=a.device,seed=a.seed,deterministic=True,project=str(out/"runs_train"),name=rid,exist_ok=True,
      optimizer="AdamW",lr0=.0005,lrf=.01,cos_lr=True,mosaic=0.,mixup=0.,copy_paste=0.,erasing=0.,
      close_mosaic=0,hsv_h=.01,hsv_s=.35,hsv_v=.25,degrees=3.,translate=.08,scale=.2,shear=1.,
      perspective=.0002,flipud=.05,fliplr=.5,box=10.,cls=.35,dfl=2.,plots=True,verbose=True)
    best=out/"runs_train"/rid/"weights/best.pt"
    if not best.exists(): raise FileNotFoundError(best)
    row["best_pt"]=str(best); del m; clean()
    for split,prefix,proj in [("val","val","runs_val1536"),("test","devtest","runs_devtest1536")]:
        m=YOLO(str(best)); q=m.val(data=a.data,split=split,imgsz=1536,iou=.55,conf=.0005,max_det=600,
          batch=1,workers=4,device=a.device,project=str(out/proj),name=rid,exist_ok=True,plots=False,verbose=False)
        row.update(metrics(q,prefix)); del m,q; clean()
    row["status"]="ok"
except Exception as e: row["error"]=repr(e); clean()
row["elapsed_sec"]=time.time()-t
j=out/"per_run_json"; j.mkdir(parents=True,exist_ok=True); (j/f"{rid}.json").write_text(json.dumps(row,indent=2))
print(json.dumps(row))
if row["status"]!="ok": raise SystemExit(1)
