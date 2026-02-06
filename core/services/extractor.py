# core/services/extractor.py

import torch
import os
from typing import List, Generator, Tuple, Union
from .base import BaseService
from ..configs import ExtractConfig
from ..structs import ModelReference
from ..io_manager import SafeStreamer
from ..math import MathKernel
from ..format_handler import FormatHandler
from ..naming import NamingManager
from ..logger import Logger

class ExtractorService(BaseService):
    def process(
        self, 
        config: ExtractConfig, 
        base_ref: ModelReference, 
        tuned_refs: List[ModelReference], 
        output_target: str
    ) -> Generator[Tuple[float, str], None, None]:
        
        is_batch = len(tuned_refs) > 1
        total_files = len(tuned_refs)
        
        yield 0.0, "Initializing Extraction..."
        
        try:
            # Load Base
            with self._resolve_source(base_ref) as base_io:
                if base_io.load_error:
                    raise ValueError(f"Failed to load Base: {base_io.load_error}")

                # Create normalized map for base keys
                base_map = {FormatHandler.fix_key_name(k): k for k in base_io.keys}
                
                for idx, tuned_ref in enumerate(tuned_refs):
                    filename = tuned_ref.name
                    yield idx / total_files, f"[{idx+1}/{total_files}] Analyzing {filename}..."
                    
                    with self._resolve_source(tuned_ref) as tuned_io:
                        if tuned_io.load_error: continue

                        output_tensors = {}
                        intersection = []
                        stats = {"bias_skipped": 0, "no_match": 0, "shape_mismatch": 0, "dim_skipped": 0}

                        # Find matching keys
                        for t_key in tuned_io.keys:
                            if not t_key.endswith(".weight"): continue
                            if "bias" in t_key: 
                                stats["bias_skipped"] += 1
                                continue
                            
                            norm_t = FormatHandler.fix_key_name(t_key)
                            # Try exact match on normalized key
                            if norm_t in base_map:
                                intersection.append((t_key, base_map[norm_t]))
                            else:
                                stats["no_match"] += 1
                        
                        total_layers = len(intersection)
                        if total_layers == 0:
                            Logger.warning(f"No matching layers found for {filename}")
                            continue

                        for i, (t_key, b_key) in enumerate(intersection):
                            if i % 50 == 0:
                                yield (idx + (i / total_layers)) / total_files, f"Processing {i}/{total_layers}..."
                            
                            try:
                                w_tuned = tuned_io.get_tensor(t_key)
                                w_base = base_io.get_tensor(b_key)
                                
                                if w_tuned is None or w_base is None: continue
                                
                                if w_tuned.shape != w_base.shape:
                                    if w_tuned.numel() == w_base.numel(): w_base = w_base.view_as(w_tuned)
                                    else:
                                        stats["shape_mismatch"] += 1
                                        continue
                                
                                # Allow 1D (Norms), 2D (Linear), 4D (Conv)
                                if len(w_tuned.shape) not in [1, 2, 4]:
                                    stats["dim_skipped"] += 1
                                    continue

                                # Delta
                                delta = w_tuned.float() - w_base.float()
                                
                                # Threshold
                                if config.threshold > 0:
                                    delta[torch.abs(delta) < config.threshold] = 0.0
                                
                                # Scale
                                if config.baked_scale != 1.0:
                                    delta *= config.baked_scale

                                # Prepare for SVD
                                delta_flat = None
                                is_conv = False
                                if len(delta.shape) == 2: 
                                    delta_flat = delta
                                elif len(delta.shape) == 4: 
                                    delta_flat = delta.reshape(delta.shape[0], -1)
                                    is_conv = True
                                elif len(delta.shape) == 1:
                                    delta_flat = delta.unsqueeze(1) # Rank 1 for vectors

                                if delta_flat is None: continue

                                local_rank = config.rank
                                if delta_flat.shape[1] == 1: local_rank = 1

                                ld, lu, eff_rank = MathKernel.svd_decomposition(
                                    delta_flat, local_rank, clamp_threshold=config.threshold
                                )
                                
                                # Naming
                                safe_name = FormatHandler.convert_to_kohya_key(b_key)
                                alpha = float(config.manual_alpha) if config.manual_alpha is not None else float(eff_rank)
                                
                                output_tensors[f"{safe_name}.lora_down.weight"] = ld.to(dtype=torch.bfloat16)
                                output_tensors[f"{safe_name}.lora_up.weight"] = lu.to(dtype=torch.bfloat16)
                                output_tensors[f"{safe_name}.alpha"] = torch.tensor(alpha, dtype=torch.bfloat16)
                                
                                del w_tuned, w_base, delta, delta_flat, ld, lu

                            except Exception as e:
                                Logger.error(f"Error extracting {t_key}: {e}")
                                continue

                        # Save
                        meta = {
                            "ss_network_dim": str(config.rank),
                            "ss_network_alpha": str(config.manual_alpha if config.manual_alpha else config.rank),
                            "modelspec.title": f"Extracted {filename}"
                        }
                        
                        if config.save_to_workspace:
                            out_name = output_target if not is_batch else f"{output_target}_{filename}"
                            self.workspace.add_model(out_name, output_tensors, meta)
                        else:
                            out_path = NamingManager.resolve_output_path(tuned_ref.path, output_target, "_extracted", is_batch)
                            SafeStreamer.save_tensors(output_tensors, out_path, meta)
                        
                        del output_tensors
                        self.garbage_collect()

            yield 1.0, "Extraction Complete"
        except Exception as e:
            Logger.error(f"Extract Error: {e}")
            yield 0.0, f"Error: {e}"