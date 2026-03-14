
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
from ..model_specs import ModelRegistry
from ..architectures.base import UnknownArchitecture

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
            with self._resolve_source(base_ref) as base_io:
                if base_io.load_error:
                    raise ValueError(f"Failed to load Base: {base_io.load_error}")

                spec = ModelRegistry.get_spec(base_io.keys)
                
                # SAFETY ABORT
                if isinstance(spec, UnknownArchitecture):
                    raise ValueError("Base model architecture is not supported by any plugin.")
                    
                Logger.info(f"Extractor detected architecture: {spec.name}")

                base_map = {FormatHandler.fix_key_name(k, spec): k for k in base_io.keys}
                
                for idx, tuned_ref in enumerate(tuned_refs):
                    filename = tuned_ref.name
                    yield idx / total_files, f"[{idx+1}/{total_files}] Analyzing {filename}..."
                    
                    with self._resolve_source(tuned_ref) as tuned_io:
                        if tuned_io.load_error: continue

                        output_tensors = {}
                        intersection = []
                        stats = {"bias_skipped": 0, "no_match": 0, "shape_mismatch": 0, "dim_skipped": 0}

                        for t_key in tuned_io.keys:
                            if not t_key.endswith(".weight"): continue
                            if "bias" in t_key: 
                                stats["bias_skipped"] += 1
                                continue
                            
                            norm_t = FormatHandler.fix_key_name(t_key, spec)
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
                                
                                w_tuned = spec.preprocess_tensor(t_key, w_tuned)
                                w_base = spec.preprocess_tensor(b_key, w_base)
                                
                                if w_tuned.shape != w_base.shape:
                                    if w_tuned.numel() == w_base.numel(): w_base = w_base.view_as(w_tuned)
                                    else:
                                        stats["shape_mismatch"] += 1
                                        continue
                                
                                if len(w_tuned.shape) not in [1, 2, 4]:
                                    stats["dim_skipped"] += 1
                                    continue

                                delta = w_tuned.float() - w_base.float()
                                
                                if config.threshold > 0:
                                    delta[torch.abs(delta) < config.threshold] = 0.0
                                
                                if config.baked_scale != 1.0:
                                    delta *= config.baked_scale

                                delta_flat = None
                                if len(delta.shape) == 2: delta_flat = delta
                                elif len(delta.shape) == 4: delta_flat = delta.reshape(delta.shape[0], -1)
                                elif len(delta.shape) == 1: delta_flat = delta.unsqueeze(1) 

                                if delta_flat is None: continue

                                local_rank = config.rank
                                if delta_flat.shape[1] == 1: local_rank = 1

                                ld, lu, eff_rank = MathKernel.svd_decomposition(
                                    delta_flat, local_rank, clamp_threshold=config.threshold
                                )
                                
                                safe_name = FormatHandler.convert_to_kohya_key(b_key, spec)
                                alpha = float(config.manual_alpha) if config.manual_alpha is not None else float(eff_rank)
                                
                                output_tensors[f"{safe_name}.lora_down.weight"] = ld.to(dtype=torch.bfloat16)
                                output_tensors[f"{safe_name}.lora_up.weight"] = lu.to(dtype=torch.bfloat16)
                                output_tensors[f"{safe_name}.alpha"] = torch.tensor(alpha, dtype=torch.bfloat16)
                                
                                del w_tuned, w_base, delta, delta_flat, ld, lu

                            except Exception as e:
                                Logger.error(f"Error extracting {t_key}: {e}")
                                continue

                        meta = {
                            "ss_network_dim": str(config.rank),
                            "ss_network_alpha": str(config.manual_alpha if config.manual_alpha else config.rank),
                            "modelspec.title": f"Extracted {filename}",
                            "modelspec.architecture": spec.name
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