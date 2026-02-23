
import torch
import os
import copy
import comfy.utils
import comfy.sd
from collections import OrderedDict
from safetensors.torch import save_file, safe_open
from typing import Dict, Any, Optional, List, Callable, Tuple
from .format_handler import FormatHandler
from .math import MathKernel
from .model_specs import ModelRegistry
from .structs_assembly import LoRAAssembly, LoRAModule

# --- DEVICE UTILS ---
def get_optimal_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# --- CACHING ---
_CACHE_LIMIT = 10
# Cache stores LoRAAssembly now
_LORA_CACHE: OrderedDict[str, LoRAAssembly] = OrderedDict()

def load_lora_cached(path: str) -> LoRAAssembly:
    global _LORA_CACHE
    if path in _LORA_CACHE:
        _LORA_CACHE.move_to_end(path)
        # Return a clone to prevent mutation of cached object by nodes
        return _LORA_CACHE[path].clone()
    
    try:
        sd = comfy.utils.load_torch_file(path)
        metadata = {}
        if path.lower().endswith(".safetensors"):
            try:
                with safe_open(path, framework="pt", device="cpu") as f:
                    metadata = f.metadata() or {}
            except Exception: pass 
        
        assembly = LoRAAssembly.from_state_dict(sd, metadata)
        
        if len(_LORA_CACHE) >= _CACHE_LIMIT:
            _LORA_CACHE.popitem(last=False)
            
        _LORA_CACHE[path] = assembly
        return assembly.clone()
    except Exception as e:
        print(f"[Latent Shaper] Error loading {path}: {e}")
        return LoRAAssembly()

def save_ls_lora(ls_lora: Dict, path: str, precision: str = "FP16", save_meta: bool = True) -> str:
    """
    Saves LS_LORA (which wraps LoRAAssembly) to disk.
    """
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
            save_dict[k] = v.to(device="cpu", dtype=target_dtype).contiguous()
        
        save_file(save_dict, path, metadata=meta)
        return path
    except Exception as e:
        print(f"[Latent Shaper] Save failed: {e}")
        raise e

# --- MATRIX OPS ---

def apply_lora_assembly(model, clip, assembly: LoRAAssembly, strength: float):
    """
    Applies LoRAAssembly to ComfyUI Model/CLIP.
    Requires converting back to state_dict temporarily.
    """
    if not assembly.modules: return (model, clip)
    
    # Convert to flat dict for ComfyUI's loader
    lora_dict = assembly.to_state_dict()
    
    print(f"[Latent Shaper] Patching model with {len(lora_dict)} keys (Strength: {strength})...")
    new_model, new_clip = comfy.sd.load_lora_for_models(
        model, clip, lora_dict, strength_model=strength, strength_clip=strength
    )
    return (new_model, new_clip)

def process_merge_dict(active_loras: List[Dict], algorithm: str, target_rank: int, global_strength: float) -> Dict[str, torch.Tensor]:
    # Legacy merge support or refactor for Assembly?
    # For now, let's keep the logic but adapt it to use Assemblies if passed.
    # NOTE: This function was heavily dependent on raw dicts. 
    # Since LS_Merger now receives LS_LORA (Assembly), we should rewrite it.
    
    print(f"[Latent Shaper] Merging {len(active_loras)} LoRAs via {algorithm}...")
    
    # Collect all unique block names
    all_block_names = set()
    
    # Pre-load assemblies
    for item in active_loras:
        if "assembly" not in item:
            # Fallback for legacy dicts (should not happen with new nodes)
            item["assembly"] = LoRAAssembly.from_state_dict(item.get("sd", {}))
        
        for name in item["assembly"].modules.keys():
            all_block_names.add(name)

    merged_assembly = LoRAAssembly()
    device = get_optimal_device()

    for block_name in all_block_names:
        deltas = []
        ratios = []
        ref_shape = None
        
        # Find reference shape
        for item in active_loras:
            mod = item["assembly"].modules.get(block_name)
            if mod:
                # Reconstruct matrix on CPU/Device
                # We need full matrix for merging usually
                if mod.is_decomposed: mod.compose()
                mat = (mod.up.float() @ mod.down.float())
                
                # Apply alpha scaling
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

        # Create new module
        new_mod = LoRAModule(
            new_down.to("cpu", dtype=torch.float16),
            new_up.to("cpu", dtype=torch.float16),
            float(new_rank)
        )
        merged_assembly.modules[block_name] = new_mod
        
        # Copy key mapping from first source that has this block
        for item in active_loras:
            if block_name in item["assembly"].key_map:
                merged_assembly.key_map[block_name] = item["assembly"].key_map[block_name]
                break

        del deltas, final_delta, new_down, new_up

    return merged_assembly.to_state_dict()