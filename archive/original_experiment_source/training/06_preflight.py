
from pathlib import Path
import csv,torch
from ultralytics import YOLO
from ultralytics.nn.modules.sldra_block import C3k2SLDRA
P=Path(__file__).resolve().parent; r=list(csv.DictReader((P/"SLDRA54_MANIFEST.csv").open()))
assert len(r)==54 and len(list((P/"configs").glob("yolo11s_SD*.yaml")))==54
for v in [0,8,9,17,18,26,27,35,36,44,45,53]:
    q=r[v]; R={"mad":0,"rms":1,"topk":2}; F={"sum":0,"product":1,"competitive":2}
    m=C3k2SLDRA(128,256,1,False,.25,v,v//9,int(q["channel_kernel"]),int(q["channel_dilation"]),
      int(q["spatial_kernel"]),int(q["groups"]),R[q["robust_mode"]],F[q["fusion_mode"]],
      float(q["alpha"]),float(q["temperature"]),float(q["boundary_weight"])).eval()
    with torch.no_grad(): y=m(torch.randn(1,128,32,32))
    assert y.shape==(1,256,32,32) and torch.isfinite(y).all()
m=YOLO(str(P/"configs/yolo11s_SD000_A_RCCM.yaml")); n=sum(x.numel() for x in m.model.parameters())
assert m.model.yaml.get("scale")=="s" and n>8_000_000
print("[PASS] 54 variants; scale=s; params",n)
