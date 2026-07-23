
from pathlib import Path
import csv,copy,yaml
P=Path(__file__).resolve().parent
O=Path("/home/drnguyenvinh/notebooks/cvio_yolo11s_method_zoo/third_party/ultralytics/ultralytics/cfg/models/11/yolo11.yaml")
B=yaml.safe_load(O.read_text()); B["nc"]=2; B["scale"]="s"
R={"mad":0,"rms":1,"topk":2}; F={"sum":0,"product":1,"competitive":2}
out=P/"configs"; out.mkdir(exist_ok=True)
for r in csv.DictReader((P/"SLDRA54_MANIFEST.csv").open()):
    c=copy.deepcopy(B); v=int(r["variant_id"]); fam=v//9
    for idx,shortcut in [(4,False),(6,True),(8,True)]:
        c2=c["backbone"][idx][3][0]
        c["backbone"][idx][2]="C3k2SLDRA"
        c["backbone"][idx][3]=[c2,shortcut,0.25,v,fam,int(r["channel_kernel"]),
            int(r["channel_dilation"]),int(r["spatial_kernel"]),int(r["groups"]),
            R[r["robust_mode"]],F[r["fusion_mode"]],float(r["alpha"]),
            float(r["temperature"]),float(r["boundary_weight"])]
    p=out/f"yolo11s_SD{v:03d}_{r['family']}.yaml"
    p.write_text(yaml.safe_dump(c,sort_keys=False)); print("[CONFIG]",p)
