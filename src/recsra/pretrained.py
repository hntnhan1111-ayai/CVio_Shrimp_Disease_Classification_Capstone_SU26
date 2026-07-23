"""Fair pretrained-weight transfer used by the frozen winner."""

from __future__ import annotations

import gc
from typing import Any

import torch


def clean_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_fair_pretrained(target_yolo: Any, checkpoint: str) -> tuple[int, int, int]:
    """Load direct tensors and remap C3k2 weights into wrapper layers 4, 6 and 8.

    The guard thresholds are frozen from the successful SD001_A_RCCM run.
    """
    from ultralytics import YOLO

    source_yolo = YOLO(checkpoint)
    source_state = source_yolo.model.state_dict()
    target_state = target_yolo.model.state_dict()

    loadable = {}
    direct_count = 0
    wrapper_count = 0

    for target_key, target_tensor in target_state.items():
        candidates = [target_key]
        for layer_id in (4, 6, 8):
            prefix = f"model.{layer_id}.block."
            if target_key.startswith(prefix):
                candidates.insert(0, f"model.{layer_id}." + target_key[len(prefix):])

        selected = None
        for source_key in candidates:
            if source_key in source_state and source_state[source_key].shape == target_tensor.shape:
                selected = source_key
                break
        if selected is None:
            continue

        loadable[target_key] = source_state[selected].detach().clone()
        if selected == target_key:
            direct_count += 1
        else:
            wrapper_count += 1

    incompatible = target_yolo.model.load_state_dict(loadable, strict=False)
    total_target = len(target_state)
    print(
        f"[FAIR PRETRAIN] direct={direct_count} wrapper_remap={wrapper_count} "
        f"total={len(loadable)}/{total_target}",
        flush=True,
    )
    print(f"[FAIR PRETRAIN] remaining_missing={len(incompatible.missing_keys)}", flush=True)

    del source_yolo
    clean_cuda()

    if wrapper_count < 20:
        raise RuntimeError(f"Wrapper remapping unexpectedly low: {wrapper_count}")
    if len(loadable) < 390:
        raise RuntimeError(f"Too few pretrained tensors: {len(loadable)}/{total_target}")

    return direct_count, wrapper_count, len(loadable)
