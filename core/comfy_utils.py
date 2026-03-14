
import torch
import os
import threading
import comfy.utils
import comfy.sd
from collections import OrderedDict
from safetensors.torch import save_file, safe_open
from typing import Dict, Any, Optional, List, Tuple
from .math import MathKernel
from .model_specs import ModelRegistry
from .structs_assembly import LoRAAssembly

# --- CACHING SYSTEM ---
_CACHE_LIMIT = 10
_LORA_CACHE: OrderedDict[str, LoRAAssembly] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_STATS = {"hits": 0, "misses": 0}

def get_optimal_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_lora_cached(path: str) -> LoRAAssembly:
    """
    Thread-safe LoRA loader with caching and statistics.
    """
    global _LORA_CACHE, _CACHE_STATS
    
    with _CACHE_LOCK:
        if path in _LORA_CACHE:
            _CACHE_STATS["hits"] += 1
            _LORA_CACHE.move_to_end(path)
            return _LORA_CACHE[path].clone()
        
        _CACHE_STATS["misses"] += 1

    try:
        sd = comfy.utils.load_torch_file(path)
        metadata = {}
        if path.lower().endswith(".safetensors"):
            try:
                with safe_open(path, framework="pt", device="cpu") as f:
                    metadata = f.metadata() or {}
            except Exception: pass 
        
        assembly = LoRAAssembly.from_state_dict(sd, metadata)
        
        with _CACHE_LOCK:
            if len(_LORA_CACHE) >= _CACHE_LIMIT:
                _LORA_CACHE.popitem(last=False)
            _LORA_CACHE[path] = assembly
            return assembly.clone()
            
    except Exception as e:
        print(f"[Latent Shaper] Error loading {path}: {e}")
        return LoRAAssembly()

def save_ls_lora(ls_lora: Dict, path: str, precision: str = "FP16", save_meta: bool = True) -> str:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        assembly: LoRAAssembly = ls_lora.get("assembly")
        if not assembly:
            raise ValueError("No assembly found in LS_LORA")

        sd = assembly.to_state_dict()
        meta = assembly.metadata if save_meta else {}
        
        dtype_map = {"FP16": torch.float16, "BF16": torch.bfloat16, "FP32": torch.float32}
        target_dtype = dtype_map.get(precision, torch.float16)
        
        save_dict = {}
        for k, v in sd.items():
            # Ensure dedicated storage to prevent file bloat
            save_dict[k] = v.to(device="cpu", dtype=target_dtype).detach().clone().contiguous()
        
        save_file(save_dict, path, metadata=meta)
        return path
    except Exception as e:
        print(f"[Latent Shaper] Save failed: {e}")
        raise e

def apply_lora_assembly(model, clip, assembly: LoRAAssembly, strength: float):
    if not assembly.modules: return (model, clip)
    lora_dict = assembly.to_state_dict()
    new_model, new_clip = comfy.sd.load_lora_for_models(
        model, clip, lora_dict, strength_model=strength, strength_clip=strength
    )
    return (new_model, new_clip)

def process_merge_dict(active_loras: List[Dict], algorithm: str, target_rank: int, global_strength: float) -> Dict[str, torch.Tensor]:
    print(f"[Latent Shaper] Merging {len(active_loras)} LoRAs via {algorithm}...")
    all_block_names = set()
    for item in active_loras:
        if "assembly" not in item:
            item["assembly"] = LoRAAssembly.from_state_dict(item.get("sd", {}))
        for name in item["assembly"].modules.keys():
            all_block_names.add(name)

    merged_assembly = LoRAAssembly()
    device = get_optimal_device()

    for block_name in all_block_names:
        deltas = []
        ratios = []
        ref_shape = None
        
        for item in active_loras:
            mod = item["assembly"].modules.get(block_name)
            if mod:
                if mod.is_decomposed: mod.compose()
                mat = (mod.up.float() @ mod.down.float())
                scale = mod.alpha / mod.rank if mod.rank > 0 else 1.0
                mat *= scale
                ref_shape = mat.shape
                del mat
                break
        
        if ref_shape is None: continue

        for item in active_loras:
            mod = item["assembly"].modules.get(block_name)
            ratio = item["ratio"]
            if not mod:
                mat = torch.zeros(ref_shape, device=device, dtype=torch.float32)
            else:
                if mod.is_decomposed: mod.compose()
                mat = (mod.up.to(device).float() @ mod.down.to(device).float())
                scale = mod.alpha / mod.rank if mod.rank > 0 else 1.0
                mat *= scale
                if mat.shape != ref_shape:
                     mat = torch.zeros(ref_shape, device=device, dtype=torch.float32)
            deltas.append(mat)
            ratios.append(ratio)

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
        final_target_rank = target_rank if target_rank > 0 else 64
        final_delta = torch.nan_to_num(final_delta.float())
        new_down, new_up, new_rank = MathKernel.svd_decomposition(final_delta, final_target_rank)

        new_mod = LoRAModule(
            new_down.to("cpu", dtype=torch.float16),
            new_up.to("cpu", dtype=torch.float16),
            float(new_rank)
        )
        merged_assembly.modules[block_name] = new_mod
        for item in active_loras:
            if block_name in item["assembly"].key_map:
                merged_assembly.key_map[block_name] = item["assembly"].key_map[block_name]
                break

    return merged_assembly.to_state_dict()