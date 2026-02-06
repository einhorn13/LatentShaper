# gui/actions/checkpoint_merge.py

import gradio as gr
from core.configs import CheckpointMergeConfig
from core.jobs import CheckpointMergeJob
from ..context import queue_manager
from .common import resolve_path_priority

def submit_checkpoint_merge(
    path_a, path_b, path_c, 
    lora_disk, lora_ws, lora_up,
    weight_str, out_name, mode, te_p, vae_p, prec, lora_str
):
    if not path_a:
        gr.Warning("Model A is required.")
        return
    
    try:
        weights = [float(x.strip()) for x in weight_str.split(",")]
        if len(weights) < 31:
            weights += [weights[-1]] * (31 - len(weights))
        weights = weights[:31]
    except Exception:
        gr.Error("Invalid weights.")
        return

    lora_paths = resolve_path_priority(lora_ws, lora_disk, lora_up)
    lora_path = lora_paths[0] if lora_paths else None

    config = CheckpointMergeConfig(
        mode=mode, te_policy=te_p, vae_policy=vae_p, precision=prec,
        lora_strength=float(lora_str), weights=weights, lora_path=lora_path
    )

    job = CheckpointMergeJob(config, [path_a, path_b, path_c], out_name or "merged.safetensors")
    queue_manager.submit_job(job)
    gr.Info("Queued.")
    return