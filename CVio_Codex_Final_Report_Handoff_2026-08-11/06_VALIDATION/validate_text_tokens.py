import argparse,json
from pathlib import Path
import fitz
ap=argparse.ArgumentParser(); ap.add_argument('pdf'); ap.add_argument('--out',required=True); a=ap.parse_args()
text='\n'.join(p.get_text() for p in fitz.open(a.pdf))
bad=['predictionlevel','173image','primarytest','MacroF1','SDI4','EXT3Original','crossentropy','trainingonly','testpartition','modelfamily','classspecific','additionalsource','mixedsource','outofsource','finalmodel','featurerecalibration','runtorun','errortransition','unseenspecimen','shrimpbody','methoddependent','WSSVrecall','singledisease','datasetspecific','decom posed','con figuration','re peated']
hits=[x for x in bad if x in text]
status='PASS' if not hits else 'FAIL'
Path(a.out).write_text(json.dumps({'status':status,'hits':hits},indent=2),encoding='utf-8')
print(status,hits)
raise SystemExit(0 if status=='PASS' else 2)
