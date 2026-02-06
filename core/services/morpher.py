# core/services/morpher.py

import torch
import numpy as np
import re
from typing import List, Generator, Tuple
from .base import BaseService
from ..configs import MorphConfig
from ..structs import ModelReference
from ..io_manager import SafeStreamer
from ..math import MathKernel
from ..format_handler import FormatHandler
from ..model_specs import ModelRegistry
from ..tensor_processor import TensorProcessor
from ..naming import NamingManager
from ..logger import Logger

class MorpherService(BaseService):
    def process(
        self, 
        config: MorphConfig, 
        inputs: List[ModelReference], 
        output_target: str
    ) -> Generator[Tuple[float, str], None, None]:
        
        if config.is_bridge:
            yield from self._process_bridge(config, inputs[0], output_target)
            return

        is_batch = len(inputs) > 1
        total = len(inputs)
        
        tp_params = {
            "erase_blocks_set": MathKernel.parse_block_string(config.erase_blocks),
            "dare_enabled": config.dare_enabled, "dare_rate": config.dare_rate,
            "fft_cutoff": config.fft_cutoff,
            "band_stop_enabled": config.band_stop_enabled, "band_stop_start": config.band_stop_start, "band_stop_end": config.band_stop_end,
            "spectral_enabled": config.spectral_enabled, "spectral_threshold": config.spectral_threshold, "spectral_adaptive": config.spectral_adaptive,
            "spectral_remove_structure": config.spectral_remove_structure,
            "eraser_start": config.eraser_start, "eraser_end": config.eraser_end,
            "homeostatic": config.homeostatic, "homeostatic_thr": config.homeostatic_thr,
            "clamp_quantile": config.clamp_quantile
        }

        for idx, ref in enumerate(inputs):
            yield idx / total, f"Morphing {ref.name}..."
            
            try:
                with self._resolve_source(ref) as io:
                    spec = ModelRegistry.get_spec(io.keys)
                    groups = FormatHandler.group_keys(io.keys)
                    
                    count = spec.block_count if spec.block_count > 0 else 30
                    multipliers = np.ones(count)
                    if config.eq_interpolate:
                        half = count // 2
                        multipliers[:half] = np.linspace(config.eq_in, config.eq_mid, half)
                        multipliers[half:] = np.linspace(config.eq_mid, config.eq_out, count - half)
                    else:
                        for i in range(count):
                            reg = spec.get_region(i)
                            if reg == "IN": multipliers[i] = config.eq_in
                            elif reg == "MID": multipliers[i] = config.eq_mid
                            elif reg == "OUT": multipliers[i] = config.eq_out

                    output_tensors = {}
                    
                    for grp in groups:
                        ld = io.get_tensor(grp.down_key).to(self.device).float()
                        lu = io.get_tensor(grp.up_key).to(self.device).float()
                        b_idx = spec.get_block_number(grp.base_name)
                        
                        eq_val = multipliers[b_idx] if 0 <= b_idx < len(multipliers) else 1.0
                        total_scale = eq_val * config.eq_global
                        
                        delta = lu @ ld
                        
                        if config.fix_alpha:
                            rank = ld.shape[0]
                            at = io.get_tensor(grp.alpha_key)
                            alpha = float(at.item()) if at else rank
                            if alpha != rank:
                                delta = MathKernel.rescale_alpha(delta, alpha, rank)
                        
                        if config.temperature != 1.0:
                             delta = MathKernel.apply_eigen_temperature(delta, config.temperature)

                        delta = TensorProcessor.apply_filters(delta, tp_params, eq_factor=total_scale, b_idx=b_idx)
                        
                        if delta is None: continue 
                        
                        nd, nu, _ = MathKernel.svd_decomposition(delta, ld.shape[0])
                        
                        output_tensors[grp.down_key] = nd.to("cpu", dtype=torch.bfloat16)
                        output_tensors[grp.up_key] = nu.to("cpu", dtype=torch.bfloat16)
                        if grp.alpha_key:
                            output_tensors[grp.alpha_key] = torch.tensor(float(ld.shape[0]), dtype=torch.bfloat16)

                    meta = io.metadata.copy()
                    if config.save_to_workspace:
                        out_name = output_target if not is_batch else f"{output_target}_{ref.name}"
                        self.workspace.add_model(out_name, output_tensors, meta)
                    else:
                        out_path = NamingManager.resolve_output_path(ref.path, output_target, "_morphed", is_batch)
                        SafeStreamer.save_tensors(output_tensors, out_path, meta)
                    
                    del output_tensors
                    self.garbage_collect()

            except Exception as e:
                Logger.error(f"Morph failed: {e}")
        
        yield 1.0, "Morph Complete"

    def _align_neural_space(self, tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        curr_h, curr_w = tensor.shape
        t_f = tensor.float()
        
        U, S, Vh = torch.linalg.svd(t_f, full_matrices=False)
        rank = S.shape[0]
        
        new_U = torch.zeros((target_h, rank), device=tensor.device, dtype=torch.float32)
        new_V = torch.zeros((target_w, rank), device=tensor.device, dtype=torch.float32)
        
        if target_h == 3840:
            HEADS = 32
            HEAD_DIM = 120
            fit_h_per_head = min(curr_h // HEADS, HEAD_DIM)
            u_reshaped = new_U.view(HEADS, HEAD_DIM, rank)
            for h in range(HEADS):
                src_start = h * (curr_h // HEADS)
                u_reshaped[h, :fit_h_per_head, :] = U[src_start : src_start + fit_h_per_head, :]
        else:
            stride = target_h / curr_h
            for i in range(min(curr_h, target_h)):
                idx = int(i * stride)
                if idx < target_h:
                    new_U[idx, :] = U[i, :]

        fit_w = min(curr_w, target_w)
        st_w = (target_w - fit_w) // 2
        new_V[st_w : st_w + fit_w, :] = Vh.mT[:fit_w, :]
        
        projected = new_U @ torch.diag(S) @ new_V.T
        
        source_norm = torch.norm(t_f, p='fro')
        target_norm = torch.norm(projected, p='fro')
        if target_norm > 1e-9:
            projected *= (source_norm / target_norm)
            
        return projected

    def _process_bridge(self, config: MorphConfig, ref: ModelReference, output_target: str):
        yield 0.0, "Starting Neural Bridge..."
        
        DIT_HIDDEN, DIT_MLP, DIT_BLOCKS = 3840, 10240, 30
        
        try:
            with self._resolve_source(ref) as io:
                groups = FormatHandler.group_keys(io.keys)
                if not groups: raise ValueError("Source LoRA is empty.")
                
                mapping = {"STRUCT": [], "CONCEPT": [], "STYLE": []}
                for g in groups:
                    name = g.base_name.lower()
                    if re.search(r'input_blocks\.[78]|output_blocks\.0', name): mapping["STRUCT"].append(g)
                    elif re.search(r'output_blocks\.[1-5]', name): mapping["STYLE"].append(g)
                    else: mapping["CONCEPT"].append(g)

                if not mapping["STRUCT"]: mapping["STRUCT"] = [g for g in groups if "down" in g.base_name.lower()]
                if not mapping["STYLE"]: mapping["STYLE"] = [g for g in groups if "up" in g.base_name.lower()]
                for k in mapping: 
                    if not mapping[k]: mapping[k] = groups

                output_tensors = {}
                
                for b_idx in range(DIT_BLOCKS):
                    if b_idx < 6: pool = mapping["STRUCT"]
                    elif b_idx > 24: pool = mapping["STYLE"]
                    else: pool = mapping["CONCEPT"]
                    
                    donor = pool[b_idx % len(pool)]
                    ld = io.get_tensor(donor.down_key).to(self.device).float()
                    lu = io.get_tensor(donor.up_key).to(self.device).float()
                    
                    delta_src = lu @ ld
                    
                    proj_attn = self._align_neural_space(delta_src, DIT_HIDDEN, DIT_HIDDEN)
                    proj_mlp = self._align_neural_space(delta_src, DIT_MLP, DIT_HIDDEN)
                    
                    target_mag = 0.0010 * config.strength
                    for p in [proj_attn, proj_mlp]:
                        m = torch.mean(torch.abs(p))
                        if m > 1e-11: p *= (target_mag / m)

                    nd_a, nu_a, _ = MathKernel.svd_decomposition(proj_attn, rank=64, clamp_threshold=1e-10)
                    nd_m, nu_m, _ = MathKernel.svd_decomposition(proj_mlp, rank=64, clamp_threshold=1e-10)
                    
                    prefix = f"transformer.blocks.{b_idx}"
                    
                    for comp in ["q_proj", "k_proj", "v_proj"]:
                        output_tensors[f"{prefix}.attn.{comp}.lora_down.weight"] = nd_a.to("cpu", dtype=torch.bfloat16)
                        output_tensors[f"{prefix}.attn.{comp}.lora_up.weight"] = nu_a.to("cpu", dtype=torch.bfloat16)
                        output_tensors[f"{prefix}.attn.{comp}.alpha"] = torch.tensor(64.0, dtype=torch.bfloat16)
                    
                    output_tensors[f"{prefix}.mlp.gate_proj.lora_down.weight"] = nd_m.to("cpu", dtype=torch.bfloat16)
                    output_tensors[f"{prefix}.mlp.gate_proj.lora_up.weight"] = nu_m.to("cpu", dtype=torch.bfloat16)
                    output_tensors[f"{prefix}.mlp.gate_proj.alpha"] = torch.tensor(64.0, dtype=torch.bfloat16)

                    if b_idx % 4 == 0:
                        yield (b_idx / DIT_BLOCKS), f"Projecting manifold {b_idx}/30..."
                    
                    del delta_src, proj_attn, proj_mlp, nd_a, nu_a, nd_m, nu_m

                meta = {
                    "ss_network_dim": "64", 
                    "ss_network_alpha": "64", 
                    "modelspec.title": "Converted Concept LoRA (SDXL->Turbo)"
                }
                
                if config.save_to_workspace:
                    self.workspace.add_model(output_target, output_tensors, meta)
                else:
                    out_p = NamingManager.resolve_output_path(ref.path, output_target, "_CasTurbo_v3")
                    SafeStreamer.save_tensors(output_tensors, out_p, meta)

                yield 1.0, "Conversion finished."
                
        except Exception as e:
            Logger.error(f"Bridge Failed: {e}")
            yield 0.0, f"Error: {str(e)}"