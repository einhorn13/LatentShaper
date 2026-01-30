# core/pipeline/transform/extract.py

import torch
import os
import gc
import numpy as np
from typing import List, Generator, Tuple, Union, Dict
from ...io_manager import SafeStreamer
from ...math import MathKernel
from ...naming import NamingManager
from ...logger import Logger

class ExtractMixin:
    """
    Robust LoRA extraction logic using SVD on weight differences.
    v2.1:
    - Noise Gating for BF16/FP16.
    - Kohya-ss Style Naming for ComfyUI Compatibility.
    - Uses Base Model keys for structure alignment.
    """

    # Prefixes used for intersection matching (Normalization)
    _NORM_PREFIXES = [
        "model.diffusion_model.", "transformer.", "model.", "diffusion_model.", 
        "first_stage_model.", "cond_stage_model.", "lora_unet_", "lora_te_"
    ]
    
    # Prefixes to explicitly STRIP when generating output names (Cleaning)
    _WRAPPER_PREFIXES = [
        "model.diffusion_model.", "diffusion_model.", "model.", 
        "first_stage_model.", "cond_stage_model."
    ]

    def _normalize_key(self, key: str) -> str:
        """Strips ALL known prefixes to get a canonical match key."""
        norm_key = key
        sorted_prefixes = sorted(self._NORM_PREFIXES, key=len, reverse=True)
        changed = True
        while changed:
            changed = False
            for p in sorted_prefixes:
                if norm_key.startswith(p):
                    norm_key = norm_key[len(p):]
                    changed = True
                    break
        return norm_key

    def _convert_to_kohya_key(self, base_key: str) -> str:
        """
        Converts a Base Model key to Kohya-ss format for ComfyUI compatibility.
        Example: model.diffusion_model.transformer.blocks.0.attn.weight 
             -> lora_unet_transformer_blocks_0_attn
        """
        # 1. Remove .weight
        core_name = base_key
        if core_name.endswith(".weight"):
            core_name = core_name[:-7]
            
        # 2. Strip Wrapper Prefixes
        sorted_wrappers = sorted(self._WRAPPER_PREFIXES, key=len, reverse=True)
        for prefix in sorted_wrappers:
            if core_name.startswith(prefix):
                core_name = core_name[len(prefix):]
                break 
        
        # 3. Determine LoRA Prefix
        if any(x in base_key for x in ["text_encoders", "text_encoder", "te.", "clip"]):
            lora_prefix = "lora_te_"
        else:
            lora_prefix = "lora_unet_"
            
        # 4. Replace separators
        kohya_name = core_name.replace(".", "_")
        kohya_name = kohya_name.replace("__", "_")
        
        # 5. Safety: Remove leading underscores if stripping left artifacts
        if kohya_name.startswith("_"):
            kohya_name = kohya_name.lstrip("_")
        
        return f"{lora_prefix}{kohya_name}"

    def _smart_cast(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float()

    def _prepare_tensor_for_svd(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if len(tensor.shape) == 2: return tensor, False
        if len(tensor.shape) == 4: return tensor.reshape(tensor.shape[0], -1), True
        return None, False

    def extract_lora_gen(
        self, 
        base_path: str, 
        tuned_paths: Union[str, List[str]],
        output_target: str,
        rank: int = 64, 
        threshold: float = 1e-4, 
        save_to_workspace: bool = False
    ) -> Generator[Tuple[float, str], None, None]:
        
        inputs = [tuned_paths] if isinstance(tuned_paths, str) else tuned_paths
        is_batch = len(inputs) > 1
        total_files = len(inputs)
        
        yield 0.0, "Initializing Extraction..."
        
        try:
            base_source = self._resolve_source(base_path)
            
            with SafeStreamer(base_source, device="cpu") as base_io:
                if base_io.load_error:
                    raise ValueError(f"Failed to load Base Model: {base_io.load_error}")

                base_map = {self._normalize_key(k): k for k in base_io.keys}
                
                for idx, tuned_path_or_name in enumerate(inputs):
                    filename = os.path.basename(tuned_path_or_name)
                    tuned_source = self._resolve_source(tuned_path_or_name)
                    
                    yield idx / total_files, f"[{idx+1}/{total_files}] {filename}: Analyzing..."
                    
                    with SafeStreamer(tuned_source, device="cpu") as tuned_io:
                        if tuned_io.load_error: continue

                        intersection = []
                        for t_key in tuned_io.keys:
                            if not t_key.endswith(".weight"): continue
                            if "norm" in t_key or "bias" in t_key: continue
                            
                            norm_t = self._normalize_key(t_key)
                            if norm_t in base_map:
                                intersection.append((t_key, base_map[norm_t]))
                        
                        if not intersection: continue

                        output_tensors = {}
                        total_layers = len(intersection)
                        skipped_noise = 0
                        
                        for i, (t_key, b_key) in enumerate(intersection):
                            if i % 20 == 0:
                                yield (idx + (i / total_layers)) / total_files, f"[{idx+1}/{total_files}] Processing {i}/{total_layers}..."
                            
                            try:
                                w_tuned = tuned_io.get_tensor(t_key)
                                w_base = base_io.get_tensor(b_key)
                                
                                if w_tuned is None or w_base is None: continue
                                if w_tuned.shape != w_base.shape:
                                    if w_tuned.numel() == w_base.numel(): w_base = w_base.view_as(w_tuned)
                                    else: continue
                                if len(w_tuned.shape) not in [2, 4]: continue

                                delta = self._smart_cast(w_tuned) - self._smart_cast(w_base)
                                
                                if threshold > 0:
                                    delta[torch.abs(delta) < threshold] = 0.0
                                    if torch.mean(torch.abs(delta)) < (threshold * 0.1):
                                        skipped_noise += 1
                                        continue

                                delta_flat, _ = self._prepare_tensor_for_svd(delta)
                                if delta_flat is None: continue
                                
                                lora_down, lora_up, effective_rank = MathKernel.svd_decomposition(
                                    delta_flat, 
                                    rank=rank, 
                                    auto_rank_threshold=0.0,
                                    clamp_threshold=threshold
                                )
                                
                                lora_down = torch.clamp(lora_down, -1.0, 1.0)
                                lora_up = torch.clamp(lora_up, -1.0, 1.0)

                                safe_name = self._convert_to_kohya_key(b_key)
                                
                                output_tensors[f"{safe_name}.lora_down.weight"] = lora_down.to(dtype=torch.bfloat16)
                                output_tensors[f"{safe_name}.lora_up.weight"] = lora_up.to(dtype=torch.bfloat16)
                                output_tensors[f"{safe_name}.alpha"] = torch.tensor(float(effective_rank), dtype=torch.bfloat16)

                                del w_tuned, w_base, delta, delta_flat, lora_down, lora_up
                                
                            except Exception as e:
                                Logger.error(f"Error extracting {t_key}: {e}")
                                continue
                        
                        if skipped_noise > 0:
                            Logger.info(f"Skipped {skipped_noise} layers due to noise threshold.")

                        if not output_tensors: continue

                        yield (idx + 0.99) / total_files, f"[{idx+1}/{total_files}] Saving..."
                        
                        meta = {
                            "ss_network_dim": str(rank), 
                            "ss_network_alpha": str(rank), 
                            "modelspec.title": f"Extracted {filename} (Rank {rank})",
                            "modelspec.description": f"Z-Image Turbo Extraction (Thresh={threshold})"
                        }
                        
                        if save_to_workspace:
                            out_name = output_target if not is_batch else f"{output_target}_{filename}"
                            self.workspace.add_model(out_name, output_tensors, meta, {"source": filename, "op": "extract"})
                        else:
                            current_out = NamingManager.resolve_output_path(
                                tuned_path_or_name, 
                                output_target, 
                                suffix="_extracted", 
                                is_batch=is_batch
                            )
                            SafeStreamer.save_tensors(output_tensors, current_out, metadata=meta)
                        
                        del output_tensors
                        self.garbage_collect()

            yield 1.0, "Extraction Complete"
            
        except Exception as e:
            Logger.error(f"Critical Extract Error: {e}")
            yield 0.0, f"Error: {e}"