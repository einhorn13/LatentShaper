# core/pipeline/transform/utils.py

import torch
import os
from typing import List, Generator, Tuple, Union, Dict
from ...io_manager import SafeStreamer
from ...format_handler import FormatHandler
from ...naming import NamingManager
from ...logger import Logger

class UtilsMixin:
    """
    Utilities for LoRA maintenance:
    - Format Conversion (FP32/FP16/BF16)
    - Key Normalization (Kohya-ss standard)
    - Alpha Rescaling (Physical weight adjustment)
    """

    def process_utils_gen(
        self,
        lora_paths: Union[str, List[str]],
        output_target: str,
        params: Dict[str, any],
        save_to_workspace: bool = False
    ) -> Generator[Tuple[float, str], None, None]:
        
        inputs = [lora_paths] if isinstance(lora_paths, str) else lora_paths
        is_batch = len(inputs) > 1
        total_files = len(inputs)
        
        # Parse Params
        target_precision = params.get("precision", "Keep") # Keep, FP32, FP16, BF16
        do_normalize = params.get("normalize_keys", False)
        target_alpha = params.get("target_alpha", None) # None or float
        force_alpha_rank = params.get("alpha_equals_rank", False)
        
        dtype_map = {
            "FP32": torch.float32,
            "FP16": torch.float16,
            "BF16": torch.bfloat16
        }
        target_dtype = dtype_map.get(target_precision, None)

        for idx, lora_path in enumerate(inputs):
            filename = os.path.basename(lora_path)
            source = self._resolve_source(lora_path)
            
            yield idx / total_files, f"[{idx+1}/{total_files}] Processing {filename}..."
            
            try:
                with SafeStreamer(source, device="cpu") as io:
                    if io.load_error:
                        Logger.error(f"Skipping {filename}: {io.load_error}")
                        continue
                    
                    # Group keys to handle Alpha/Down/Up relationships
                    groups = FormatHandler.group_keys(io.keys)
                    output_tensors = {}
                    
                    # Keep track of processed keys to handle non-LoRA keys (like embeddings)
                    processed_keys = set()
                    
                    for grp in groups:
                        processed_keys.add(grp.down_key)
                        processed_keys.add(grp.up_key)
                        if grp.alpha_key: processed_keys.add(grp.alpha_key)
                        
                        # Load Tensors
                        ld = io.get_tensor(grp.down_key)
                        lu = io.get_tensor(grp.up_key)
                        la = io.get_tensor(grp.alpha_key) if grp.alpha_key else None
                        
                        if ld is None or lu is None: continue
                        
                        # 1. Alpha Rescaling Logic
                        current_rank = float(ld.shape[0])
                        current_alpha = float(la.item()) if la is not None else current_rank
                        
                        final_alpha = current_alpha
                        
                        if force_alpha_rank:
                            final_alpha = current_rank
                        elif target_alpha is not None and target_alpha > 0:
                            final_alpha = float(target_alpha)
                            
                        # If alpha changed, we must rescale weights to preserve inference result
                        # Formula: W_new = W_old * (Alpha_old / Alpha_new)
                        if final_alpha != current_alpha and final_alpha > 0:
                            scale_factor = current_alpha / final_alpha
                            # Apply scale to Up matrix (arbitrary choice, math is commutative)
                            lu = lu.float() * scale_factor
                        
                        # 2. Key Normalization
                        if do_normalize:
                            # Use base_name to reconstruct standard key
                            # We need full key reconstruction logic
                            # FormatHandler.convert_to_kohya_key expects the FULL original key
                            # But here we have groups. Let's use the structure name.
                            
                            # Reconstruct a "fake" base key to pass to converter if we want pure standardization
                            # Or just use the converter on the original key
                            new_down_key = FormatHandler.convert_to_kohya_key(grp.down_key) + ".weight"
                            new_up_key = FormatHandler.convert_to_kohya_key(grp.up_key) + ".weight"
                            new_alpha_key = FormatHandler.convert_to_kohya_key(grp.down_key).replace("lora_down", "alpha")
                        else:
                            new_down_key = grp.down_key
                            new_up_key = grp.up_key
                            new_alpha_key = grp.alpha_key or f"{grp.down_key.split('.')[0]}.alpha"

                        # 3. Precision Casting
                        if target_dtype:
                            ld = ld.to(dtype=target_dtype)
                            lu = lu.to(dtype=target_dtype)
                        
                        output_tensors[new_down_key] = ld
                        output_tensors[new_up_key] = lu
                        
                        # Always save alpha if we touched it or if it existed
                        # If we normalized keys, we must save alpha even if it didn't exist before (standard practice)
                        if la is not None or final_alpha != current_rank or do_normalize:
                            output_tensors[new_alpha_key] = torch.tensor(final_alpha, dtype=target_dtype or ld.dtype)

                    # Handle remaining keys (e.g. embeddings, metadata tensors)
                    for key in io.keys:
                        if key not in processed_keys:
                            t = io.get_tensor(key)
                            if target_dtype and t.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                                t = t.to(dtype=target_dtype)
                            output_tensors[key] = t

                    # Save
                    yield (idx + 0.9) / total_files, f"[{idx+1}/{total_files}] Saving..."
                    
                    meta = io.metadata.copy()
                    meta["modelspec.title"] = f"{meta.get('modelspec.title', filename)} (Optimized)"
                    
                    if save_to_workspace:
                        out_name = output_target if not is_batch else f"{output_target}_{filename}"
                        self.workspace.add_model(out_name, output_tensors, meta, {"source": filename, "op": "utils"})
                    else:
                        suffix = "_opt"
                        if target_precision != "Keep": suffix += f"_{target_precision}"
                        current_out = NamingManager.resolve_output_path(lora_path, output_target, suffix, is_batch)
                        SafeStreamer.save_tensors(output_tensors, current_out, metadata=meta)
                    
                    del output_tensors
                    self.garbage_collect()

            except Exception as e:
                Logger.error(f"Error processing {filename}: {e}")
                yield 0.0, f"Error: {e}"
        
        yield 1.0, "Batch Complete"