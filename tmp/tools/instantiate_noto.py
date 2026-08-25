from __future__ import annotations

import sys
from pathlib import Path


FONTTOOLS_DIR = Path(__file__).resolve().parent / "fonttools"
FONTS_DIR = Path(__file__).resolve().parent / "tectonic-0.17.0" / "fonts"
sys.path.insert(0, str(FONTTOOLS_DIR))

from fontTools.ttLib import TTFont  # noqa: E402
from fontTools.varLib.instancer import instantiateVariableFont  # noqa: E402


source = FONTS_DIR / "NotoSerif-Variable.ttf"
if not source.exists():
    source = FONTS_DIR.parent / "variable-font-backup" / source.name
for weight, label in [(400, "Regular"), (700, "Bold")]:
    font = TTFont(source)
    instance = instantiateVariableFont(
        font,
        {"wght": weight, "wdth": 100},
        inplace=False,
        overlap=False,
    )
    output = FONTS_DIR / f"NotoSerif-{label}.ttf"
    names = {
        2: label,
        4: f"Noto Serif {label}",
        6: f"NotoSerif-{label}",
        17: label,
    }
    for record in instance["name"].names:
        if record.nameID in names:
            record.string = names[record.nameID].encode(record.getEncoding())
    instance["OS/2"].usWeightClass = weight
    if weight >= 700:
        instance["OS/2"].fsSelection = (
            instance["OS/2"].fsSelection | (1 << 5)
        ) & ~(1 << 6)
        instance["head"].macStyle |= 1
    instance.save(output)
    print(output)
