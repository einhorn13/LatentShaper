# core/comfy_utils.py

import torch
import os
import json
import copy
import comfy.utils
import comfy.sd
from collections import OrderedDict
from safetensors.torch import save_file, safe_open
from typing import Dict, Any, Optional, List, Callable, Tuple
from .format_handler import FormatHandler
from .math import MathKernel
from .model_specs import ModelRegistry

# --- DEVICE UTILS ---
def get_optimal_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# --- CACHING ---
_CACHE_LIMIT = 10
_LORA_CACHE: OrderedDict[str, Tuple[Dict[str, torch.Tensor], Dict[str, str]]] = OrderedDict()

def load_lora_cached(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    global _LORA_CACHE
    if path in _LORA_CACHE:
        _LORA_CACHE.move_to_end(path)
        sd, meta = _LORA_CACHE[path]
        return {k: v.clone() for k, v in sd.items()}, copy.deepcopy(meta)
    
    try:
        sd = comfy.utils.load_torch_file(path)
        metadata = {}
        if path.lower().endswith(".safetensors"):
            try:
                with safe_open(path, framework="pt", device="cpu") as f:
                    metadata = f.metadata() or {}
            except Exception: pass 
        
        if len(_LORA_CACHE) >= _CACHE_LIMIT:
            _LORA_CACHE.popitem(last=False)
            
        _LORA_CACHE[path] = (sd, metadata)
        return {k: v.clone() for k, v in sd.items()}, copy.deepcopy(metadata)
    except Exception as e:
        print(f"[LoRA Studio] Error loading {path}: {e}")
        return {}, {}

def save_z_lora(z_lora: Dict, path: str, precision: str = "FP16", save_meta: bool = True) -> str:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sd = z_lora.get("sd", {})
        meta = z_lora.get("metadata", {}) if save_meta else {}
        
        dtype_map = {"FP16": torch.float16, "BF16": torch.bfloat16, "FP32": torch.float32}
        target_dtype = dtype_map.get(precision, torch.float16)
        
        save_dict = {}
        for k, v in sd.items():
            save_dict[k] = v.to(device="cpu", dtype=target_dtype).contiguous()
        
        save_file(save_dict, path, metadata=meta)
        return path
    except Exception as e:
        print(f"[LoRA Studio] Save failed: {e}")
        raise e

# --- MATRIX OPS ---

def reconstruct_matrix(state_dict: Dict[str, torch.Tensor], group, device: str) -> Optional[torch.Tensor]:
    if not group: return None
    if group.down_key not in state_dict or group.up_key not in state_dict: return None

    try:
        down = state_dict[group.down_key].to(device).float()
        up = state_dict[group.up_key].to(device).float()
        rank = down.shape[0]
        scale = 1.0
        if group.alpha_key and group.alpha_key in state_dict:
            alpha_val = state_dict[group.alpha_key].item()
            if rank > 0: scale = alpha_val / rank
        return (up @ down) * scale
    except Exception as e:
        print(f"[LoRA Studio] Matrix reconstruction failed for {group.base_name}: {e}")
        return None

def apply_lora_dict(model, clip, lora_dict: Dict[str, torch.Tensor], strength: float):
    if not lora_dict: return (model, clip)
    print(f"[LoRA Studio] Patching model with {len(lora_dict)} keys (Strength: {strength})...")
    new_model, new_clip = comfy.sd.load_lora_for_models(
        model, clip, lora_dict, strength_model=strength, strength_clip=strength
    )
    return (new_model, new_clip)

def process_lora_dict(sd: Dict[str, torch.Tensor], callback: Callable) -> Dict[str, torch.Tensor]:
    keys = list(sd.keys())
    # Use normalized grouping to handle internal consistency
    groups = FormatHandler.group_keys(keys, normalize=True)
    spec = ModelRegistry.get_spec(keys)
    device = get_optimal_device()
    new_sd = {}
    
    for grp in groups:
        delta = reconstruct_matrix(sd, grp, device)
        if delta is None: continue
        
        b_idx = spec.get_block_number(grp.base_name)
        region = spec.get_region(b_idx)
        
        delta = callback(delta, b_idx, region, grp)
        if delta is None: continue 
        
        orig_rank = sd[grp.down_key].shape[0]
        delta = torch.nan_to_num(delta.float())
        nd, nu, nr = MathKernel.svd_decomposition(delta, orig_rank)
        
        # Use ORIGINAL keys to ensure ComfyUI compatibility
        new_sd[grp.down_key] = nd.to("cpu", dtype=torch.float16)
        new_sd[grp.up_key] = nu.to("cpu", dtype=torch.float16)
        if grp.alpha_key:
            new_sd[grp.alpha_key] = torch.tensor(float(nr), dtype=torch.float16)
            
    return new_sd

def process_merge_dict(active_loras: List[Dict], algorithm: str, target_rank: int, global_strength: float) -> Dict[str, torch.Tensor]:
    print(f"[LoRA Studio] Merging {len(active_loras)} LoRAs via {algorithm}...")

    all_block_names = set()
    max_input_rank = 0 
    
    # Map: base_name -> {struct: str, prefix: str}
    # Stores the naming convention of the first LoRA that has this block
    block_naming = {}

    for item in active_loras:
        if item["sd"] is None:
            item["sd"], _ = load_lora_cached(item["path"])
        
        keys = list(item["sd"].keys())
        # Normalize keys to align blocks from different sources
        grps = FormatHandler.group_keys(keys, normalize=True)
        item["groups"] = {g.base_name: g for g in grps}
        
        for g in grps:
            all_block_names.add(g.base_name)
            
            # Capture naming convention from the first source
            if g.base_name not in block_naming:
                block_naming[g.base_name] = {
                    "struct": g.structure_name,
                    "prefix": g.prefix
                }
            
            if g.down_key in item["sd"]:
                max_input_rank = max(max_input_rank, item["sd"][g.down_key].shape[0])

    print(f"[LoRA Studio] Found {len(all_block_names)} unique blocks to merge.")
    merged_sd = {}
    device = get_optimal_device()

    for block_name in all_block_names:
        deltas = []
        ratios = []
        ref_shape = None
        
        # Find reference shape
        for item in active_loras:
            grp = item["groups"].get(block_name)
            if grp:
                mat = reconstruct_matrix(item["sd"], grp, "cpu")
                if mat is not None:
                    ref_shape = mat.shape
                    break
        
        if ref_shape is None: continue

        for item in active_loras:
            grp = item["groups"].get(block_name)
            if not grp:
                mat = torch.zeros(ref_shape, device=device, dtype=torch.float32)
            else:
                mat = reconstruct_matrix(item["sd"], grp, device)
                if mat is None or mat.shape != ref_shape:
                    mat = torch.zeros(ref_shape, device=device, dtype=torch.float32)
            
            deltas.append(mat)
            ratios.append(item["ratio"])

        final_delta = None
        
        if algorithm.startswith("Median"):
            weighted_deltas = [d * r for d, r in zip(deltas, ratios)]
            final_delta = MathKernel.median_merge(weighted_deltas)
            
        elif algorithm.startswith("SLERP"):
            if len(deltas) == 1: final_delta = deltas[0] * ratios[0]
            else:
                curr = deltas[0]
                for i in range(1, len(deltas)):
                    curr = MathKernel.slerp(curr, deltas[i], 0.5) 
                final_delta = curr
        elif algorithm.startswith("Orthogonal"):
            final_delta = deltas[0] * ratios[0]
            for i in range(1, len(deltas)):
                ortho = MathKernel.orthogonalize_update(final_delta, deltas[i])
                final_delta += ortho * ratios[i]
        elif algorithm.startswith("TIES"):
            final_delta = MathKernel.ties_trim_and_elect_streaming(deltas, ratios, density=0.5)
        else: # SVD Sum
            final_delta = torch.zeros_like(deltas[0])
            for d, r in zip(deltas, ratios):
                final_delta.add_(d, alpha=r)

        if global_strength != 1.0: final_delta *= global_strength

        final_target_rank = target_rank if target_rank > 0 else max_input_rank
        if final_target_rank < 1: final_target_rank = 64

        final_delta = torch.nan_to_num(final_delta.float())
        new_down, new_up, new_rank = MathKernel.svd_decomposition(final_delta, final_target_rank)

        # FIX: Reconstruct key using EXACTLY the original prefix and structure
        naming = block_naming.get(block_name)
        prefix = naming["prefix"]
        struct = naming["struct"]
        
        base_key = f"{prefix}{struct}"
        
        merged_sd[f"{base_key}.lora_down.weight"] = new_down.to("cpu", dtype=torch.float16)
        merged_sd[f"{base_key}.lora_up.weight"] = new_up.to("cpu", dtype=torch.float16)
        merged_sd[f"{base_key}.alpha"] = torch.tensor(float(new_rank), dtype=torch.float16)

        del deltas, final_delta, new_down, new_up

    return merged_sd