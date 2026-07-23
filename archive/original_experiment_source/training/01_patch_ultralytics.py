
from pathlib import Path
import shutil
P=Path(__file__).resolve().parent
U=Path("/home/drnguyenvinh/notebooks/cvio_yolo11s_method_zoo/third_party/ultralytics")
M=U/"ultralytics/nn/modules"; T=U/"ultralytics/nn/tasks.py"; I=M/"__init__.py"
shutil.copy2(P/"sldra_block.py",M/"sldra_block.py")
s=I.read_text()
if "# BEGIN CVIO SLDRA" not in s:
    I.write_text(s.rstrip()+"\n# BEGIN CVIO SLDRA\nfrom .sldra_block import C3k2SLDRA, SLDRA\n# END CVIO SLDRA\n")
s=T.read_text()
if "# CVIO_SLDRA_IMPORT" not in s:
    a=s.find("from ultralytics.nn.modules import ("); b=s.find(")",a)
    s=s[:b]+"\n    # CVIO_SLDRA_IMPORT\n    C3k2SLDRA,\n"+s[b:]
def addset(s,name):
    a=s.find(f"{name} = frozenset("); b=s.find("{",a); d=0
    for i in range(b,len(s)):
        d += (s[i]=="{")-(s[i]=="}")
        if d==0:
            if "C3k2SLDRA" not in s[b:i]: s=s[:i]+"\n            C3k2SLDRA,\n"+s[i:]
            return s
    raise RuntimeError(name)
s=addset(s,"base_modules"); s=addset(s,"repeat_modules"); T.write_text(s)
print("[PATCHED SLDRA]")
