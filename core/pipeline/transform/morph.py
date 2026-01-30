# core/pipeline/transform/morph.py

import torch
import os
import numpy as np
from typing import List, Generator, Tuple, Union, Dict
from ...io_manager import SafeStreamer
from ...math import MathKernel
from ...format_handler import FormatHandler
from ...model_specs import ModelRegistry
from ...naming import NamingManager
from ...logger import Logger
from ...tensor_processor import TensorProcessor

class MorphMixin:
    def _get_block_multipliers(self, spec, params: Dict) -> np.ndarray:
        count = spec.block_count if spec.block_count > 0 else 30
        eq_in, eq_mid, eq_out = params.get("eq_in", 1.0), params.get("eq_mid", 1.0), params.get("eq_out", 1.0)

        if params.get("eq_interpolate", False):
            first_half = np.linspace(eq_in, eq_mid, count // 2)
            second_half = np.linspace(eq_mid, eq_out, count - (count // 2))
            multipliers = np.concatenate([first_half, second_half])
            return multipliers
        
        multipliers = np.ones(count)
        for i in range(count):
            reg = spec.get_region(i).upper()
            multipliers[i] = params.get(f"eq_{reg.lower()}", 1.0)
        return multipliers

    def morph_lora_gen(self, lora_paths: Union[str, List[str]], output_target: str, params: Dict[str, any], save_to_workspace: bool = False) -> Generator[Tuple[float, str], None, None]:
        inputs = [lora_paths] if isinstance(lora_paths, str) else lora_paths
        total_files = len(inputs)

        # Pre-parse params for TensorProcessor
        erase_blocks_str = params.get("erase_blocks", "")
        params["erase_blocks_set"] = MathKernel.parse_block_string(erase_blocks_str)
        
        # Flags for optimization
        has_heavy_filters = any([
            params.get("dare_enabled", False),
            params.get("spectral_enabled", False),
            params.get("eraser_end", 0) > params.get("eraser_start", 0),
            params.get("fft_cutoff", 1.0) < 1.0,
            params.get("clamp_quantile", 1.0) < 1.0,
            params.get("homeostatic", False),
            params.get("band_stop_enabled", False)
        ])
        
        fix_alpha = params.get("fix_alpha", False)
        global_scale = params.get("eq_global", 1.0)

        for idx, lora_path in enumerate(inputs):
            source = self._resolve_source(lora_path)
            yield idx / total_files, f"Morphing {os.path.basename(lora_path)}..."
            
            try:
                with SafeStreamer(source, device=self.device) as io:
                    spec = ModelRegistry.get_spec(io.keys)
                    groups = FormatHandler.group_keys(io.keys)
                    block_weights = self._get_block_multipliers(spec, params)
                    output_tensors = {}

                    for grp in groups:
                        ld = io.get_tensor(grp.down_key).to(self.device).float()
                        lu = io.get_tensor(grp.up_key).to(self.device).float()
                        b_idx = spec.get_block_number(grp.base_name)
                        
                        # Calculate EQ factor
                        m_val = block_weights[b_idx] if 0 <= b_idx < len(block_weights) else 1.0
                        total_m = m_val * global_scale

                        if not has_heavy_filters and not params["erase_blocks_set"]:
                            # Optimized Path (No SVD)
                            if fix_alpha:
                                curr_rank = ld.shape[0]
                                at = io.get_tensor(grp.alpha_key)
                                curr_alpha = float(at.item()) if at is not None else curr_rank
                                if curr_alpha != curr_rank:
                                    total_m *= (curr_alpha / curr_rank)
                            
                            output_tensors[grp.down_key] = ld.to("cpu", dtype=torch.bfloat16)
                            output_tensors[grp.up_key] = (lu * total_m).to("cpu", dtype=torch.bfloat16)
                            if grp.alpha_key:
                                output_tensors[grp.alpha_key] = torch.tensor(float(ld.shape[0]), dtype=torch.bfloat16)
                        else:
                            # Heavy Path via TensorProcessor
                            delta = lu @ ld
                            
                            if fix_alpha:
                                cr = ld.shape[0]
                                at = io.get_tensor(grp.alpha_key)
                                ca = float(at.item()) if at is not None else cr
                                if ca != cr: delta = MathKernel.rescale_alpha(delta, ca, cr)

                            # Apply all filters via Processor
                            delta = TensorProcessor.apply_filters(delta, params, eq_factor=total_m, b_idx=b_idx)
                            
                            if delta is None: continue # Block erased

                            rank = ld.shape[0]
                            nd, nu, _ = MathKernel.svd_decomposition(delta, rank)
                            output_tensors[grp.down_key], output_tensors[grp.up_key] = nd.to("cpu", dtype=torch.bfloat16), nu.to("cpu", dtype=torch.bfloat16)
                            if grp.alpha_key: output_tensors[grp.alpha_key] = torch.tensor(float(rank), dtype=torch.bfloat16)
                            del delta, nd, nu

                        del ld, lu

                    meta = io.metadata.copy()
                    if save_to_workspace: self.workspace.add_model(output_target, output_tensors, meta)
                    else: SafeStreamer.save_tensors(output_tensors, NamingManager.resolve_output_path(lora_path, output_target, "_morphed"), meta)
            except Exception as e:
                Logger.error(f"Error morphing {lora_path}: {e}")
        yield 1.0, "Morphing Complete"