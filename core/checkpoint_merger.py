
import torch
import os
import gc
import time
from typing import Dict, List, Optional, Generator, Tuple, Any
from safetensors import safe_open
from .math import MathKernel
from .format_handler import FormatHandler
from .model_specs import ModelSpec, ModelRegistry
from .logger import Logger

class CheckpointMerger:
    """
    High-performance streaming merger for checkpoints.
    Includes Key Deduplication and explicit resource management.
    """

    @staticmethod
    @torch.no_grad()
    def get_block_index(key: str, spec: ModelSpec) -> int:
        if "transformer.blocks" in key or "model.diffusion_model" in key:
            idx = spec.get_block_number(key)
            return idx + 1 if idx != -1 else 0
        return 0

    @staticmethod
    @torch.no_grad()
    def bake_lora_to_tensor(base_tensor: torch.Tensor, key: str, lora_sd: Dict[str, torch.Tensor], strength: float) -> torch.Tensor:
        kohya_base = FormatHandler.convert_to_kohya_key(key)
        down_key = f"{kohya_base}.lora_down.weight"
        up_key = f"{kohya_base}.lora_up.weight"
        alpha_key = f"{kohya_base}.alpha"

        if down_key in lora_sd and up_key in lora_sd:
            ld = lora_sd[down_key].float()
            lu = lora_sd[up_key].float()
            rank = ld.shape[0]
            alpha = lora_sd[alpha_key].item() if alpha_key in lora_sd else rank
            scale = alpha / rank if rank > 0 else 1.0
            
            delta = torch.matmul(lu, ld).mul_(scale * strength)
            if base_tensor.shape == delta.shape:
                return base_tensor.float().add_(delta.to(base_tensor.device))
        return base_tensor

    @classmethod
    @torch.no_grad()
    def merge_streamed(
        cls,
        paths: Dict[str, str],
        lora_path: Optional[str],
        weights: List[float],
        params: Dict[str, Any]
    ) -> Generator[Tuple[float, str, Optional[Dict]], None, None]:
        
        # 1. Validate Inputs
        if not {'A', 'B'}.issubset(paths.keys()):
            raise ValueError("merge_streamed expects a paths dictionary with at least 'A' and 'B' keys.")

        lora_sd = {}
        prec_str = str(params.get("precision", "BF16")).strip().upper()
        target_dtype = torch.bfloat16 if prec_str == "BF16" else (torch.float16 if prec_str == "FP16" else torch.float32)
        
        if lora_path and os.path.exists(lora_path):
            with safe_open(lora_path, framework="pt", device="cpu") as f:
                lora_sd = {k: f.get_tensor(k) for k in f.keys()}

        handles = {}
        try:
            # Explicitly open handles A and B
            for alias in ('A', 'B'):
                p = paths.get(alias)
                if not p or not os.path.exists(p):
                    raise FileNotFoundError(f"Missing or invalid path for model {alias}: {p}")
                handles[alias] = safe_open(p, framework="pt", device="cpu")

            # Optional Model C for "Add Difference" mode
            if params.get("mode") == "Add Difference" and 'C' in paths and paths['C']:
                if os.path.exists(paths['C']):
                    handles['C'] = safe_open(paths['C'], framework="pt", device="cpu")

            raw_keys = set().union(*(h.keys() for h in handles.values()))
            unique_map = {}
            prefixes = ["model.diffusion_model.", "transformer.", "model."]
            
            for k in raw_keys:
                norm_k = k
                for p in prefixes:
                    if k.startswith(p):
                        norm_k = k[len(p):]
                        break
                if norm_k not in unique_map or len(k) > len(unique_map[norm_k]):
                    unique_map[norm_k] = k
            
            all_keys = sorted(list(unique_map.values()))
            total_keys = len(all_keys)
            
            Logger.info(f"Key Deduplication: {len(raw_keys)} -> {total_keys} unique tensors.")
            spec = ModelRegistry.get_spec(all_keys[:100])

            output_sd = {}
            mode = params.get("mode", "Weighted Sum")
            te_policy = params.get("te_policy", "Copy A")
            vae_policy = params.get("vae_policy", "Copy A")
            lora_strength = params.get("lora_strength", 1.0)

            for i, key in enumerate(all_keys):
                if i % 200 == 0:
                    yield i / total_keys, f"Merging: {int((i/total_keys)*100)}% | {key[:35]}...", None

                is_te = any(x in key for x in ["text_encoders", "conditioner", "qwen"])
                is_vae = any(x in key.lower() for x in ["vae", "first_stage"])

                if is_te and te_policy != "Merge":
                    src = 'A' if te_policy == "Copy A" else 'B'
                    if src in handles and key in handles[src].keys():
                        output_sd[key] = handles[src].get_tensor(key).to(target_dtype).clone().contiguous()
                    continue

                if is_vae:
                    src = 'A' if vae_policy == "Copy A" else 'B'
                    if src in handles and key in handles[src].keys():
                        output_sd[key] = handles[src].get_tensor(key).to(target_dtype).clone().contiguous()
                    continue

                t_a = handles['A'].get_tensor(key).float() if key in handles['A'].keys() else None
                t_b = handles['B'].get_tensor(key).float() if key in handles['B'].keys() else None
                
                b_idx = cls.get_block_index(key, spec)
                alpha = weights[b_idx] if b_idx < len(weights) else weights[0]

                res = None
                if t_a is not None and t_b is not None:
                    if mode == "Weighted Sum":
                        res = torch.lerp(t_a, t_b, alpha)
                    elif mode == "Add Difference" and 'C' in handles and key in handles['C'].keys():
                        t_c = handles['C'].get_tensor(key).float()
                        res = t_a + (t_b - t_c) * alpha
                    else:
                        res = torch.lerp(t_a, t_b, alpha)
                else:
                    res = t_a if t_a is not None else t_b

                if lora_sd and res is not None:
                    res = cls.bake_lora_to_tensor(res, key, lora_sd, lora_strength)

                if res is not None:
                    output_sd[key] = res.to(dtype=target_dtype, device="cpu").clone().contiguous()

                if i % 1000 == 0:
                    gc.collect()

            yield 1.0, "Merge complete. Saving...", output_sd

        except Exception as e:
            Logger.error(f"Merge Error: {e}")
            raise e
        finally:
            # Ensure all handles are explicitly closed if the backend supports it, 
            # and references are deleted.
            for h in handles.values():
                try:
                    if hasattr(h, 'close'): h.close()
                except: pass
            handles.clear()
            gc.collect()