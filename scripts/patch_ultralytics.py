#!/usr/bin/env python3
from recsra.ultralytics_patch import patch_ultralytics

if __name__ == "__main__":
    root = patch_ultralytics()
    print(f"[PATCHED] {root}")
    print("[MODULE] C3k2SLDRA checkpoint compatibility")
    print("[ACADEMIC NAME] RECSRA")
