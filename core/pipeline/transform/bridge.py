# core/pipeline/transform/bridge.py

import torch
import torch.nn.functional as F
import numpy as np
import os
import re
from typing import List, Generator, Tuple, Union, Dict
from ...io_manager import SafeStreamer
from ...math import MathKernel
from ...format_handler import FormatHandler
from ...naming import NamingManager
from ...logger import Logger

class BridgeMixin:
    """
    Experimental Neural Bridge (SDXL -> Z-Image Turbo).
    Version 3.1: Fixed dimension mismatch for MLP layers (10240).
    Implements Dynamic Neural Projection with Frobenius-Optimal Scaling.
    """

    def _align_neural_space(self, tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Projects weight matrix into target dimensions while preserving 
        semantic feature distribution across heads or neurons.
        """
        curr_h, curr_w = tensor.shape
        t_f = tensor.float()
        
        # 1. Singular Value Decomposition
        U, S, Vh = torch.linalg.svd(t_f, full_matrices=False)
        rank = S.shape[0]
        
        # 2. Target Basis Initialization
        new_U = torch.zeros((target_h, rank), device=tensor.device, dtype=torch.float32)
        new_V = torch.zeros((target_w, rank), device=tensor.device, dtype=torch.float32)
        
        # 3. Intelligent Mapping
        if target_h == 3840:
            # Attention Path: Map to 32 heads x 120 dim
            HEADS = 32
            HEAD_DIM = 120
            fit_h_per_head = min(curr_h // HEADS, HEAD_DIM)
            
            # Reshape for head-wise processing
            u_reshaped = new_U.view(HEADS, HEAD_DIM, rank)
            for h in range(HEADS):
                src_start = h * (curr_h // HEADS)
                u_reshaped[h, :fit_h_per_head, :] = U[src_start : src_start + fit_h_per_head, :]
        else:
            # MLP/General Path: Uniform distribution of features
            # Calculate stride to spread source features across the larger target space
            stride = target_h / curr_h
            for i in range(min(curr_h, target_h)):
                idx = int(i * stride)
                if idx < target_h:
                    new_U[idx, :] = U[i, :]

        # V-matrix (Input projection) alignment (centered)
        fit_w = min(curr_w, target_w)
        st_w = (target_w - fit_w) // 2
        new_V[st_w : st_w + fit_w, :] = Vh.mT[:fit_w, :]
        
        # 4. Reconstruction with Frobenius Norm Preservation
        # This ensures the 'impact' of the weights is consistent with the source LoRA
        projected = new_U @ torch.diag(S) @ new_V.T
        
        source_norm = torch.norm(t_f, p='fro')
        target_norm = torch.norm(projected, p='fro')
        if target_norm > 1e-9:
            projected *= (source_norm / target_norm)
            
        return projected

    def convert_sdxl_to_turbo_gen(
        self, 
        sdxl_lora_path: str, 
        output_name: str, 
        strength: float = 0.5,
        save_to_workspace: bool = False
    ) -> Generator[Tuple[float, str], None, None]:
        
        yield 0.0, "Starting neural space migration..."
        source = self._resolve_source(sdxl_lora_path)
        
        DIT_HIDDEN, DIT_MLP, DIT_BLOCKS = 3840, 10240, 30

        try:
            with SafeStreamer(source, device=self.device) as io:
                groups = FormatHandler.group_keys(io.keys)
                if not groups: 
                    raise ValueError("Source LoRA is empty or unsupported.")
                
                # 1. Semantic Categorization
                mapping = {"STRUCT": [], "CONCEPT": [], "STYLE": []}
                for g in groups:
                    name = g.base_name.lower()
                    if re.search(r'input_blocks\.[78]|output_blocks\.0', name):
                        mapping["STRUCT"].append(g)
                    elif re.search(r'output_blocks\.[1-5]', name):
                        mapping["STYLE"].append(g)
                    else:
                        mapping["CONCEPT"].append(g)

                # Robust Fallbacks
                if not mapping["STRUCT"]: mapping["STRUCT"] = [g for g in groups if "down" in g.base_name.lower()]
                if not mapping["STYLE"]: mapping["STYLE"] = [g for g in groups if "up" in g.base_name.lower()]
                for k in mapping: 
                    if not mapping[k]: mapping[k] = groups

                output_tensors = {}
                
                # 2. Main Conversion Cycle
                for b_idx in range(DIT_BLOCKS):
                    # Region logic based on S3-DiT block depth
                    if b_idx < 6: pool = mapping["STRUCT"]
                    elif b_idx > 24: pool = mapping["STYLE"]
                    else: pool = mapping["CONCEPT"]
                    
                    donor = pool[b_idx % len(pool)]
                    ld = io.get_tensor(donor.down_key).to(self.device).float()
                    lu = io.get_tensor(donor.up_key).to(self.device).float()
                    
                    # Reconstruct source Concept-DNA
                    delta_src = lu @ ld
                    
                    # Project to Attention and MLP spaces
                    proj_attn = self._align_neural_space(delta_src, DIT_HIDDEN, DIT_HIDDEN)
                    proj_mlp = self._align_neural_space(delta_src, DIT_MLP, DIT_HIDDEN)
                    
                    # Apply Turbo-specific amplitude normalization (Target ~0.001 L1)
                    target_mag = 0.0010 * strength
                    for p in [proj_attn, proj_mlp]:
                        m = torch.mean(torch.abs(p))
                        if m > 1e-11: p *= (target_mag / m)

                    # Decompose back to LoRA format (Rank 64)
                    nd_a, nu_a, _ = MathKernel.svd_decomposition(proj_attn, rank=64, clamp_threshold=1e-10)
                    nd_m, nu_m, _ = MathKernel.svd_decomposition(proj_mlp, rank=64, clamp_threshold=1e-10)
                    
                    prefix = f"transformer.blocks.{b_idx}"
                    
                    # Inject into Q, K, V projections
                    for comp in ["q_proj", "k_proj", "v_proj"]:
                        output_tensors[f"{prefix}.attn.{comp}.lora_down.weight"] = nd_a.to("cpu", dtype=torch.bfloat16)
                        output_tensors[f"{prefix}.attn.{comp}.lora_up.weight"] = nu_a.to("cpu", dtype=torch.bfloat16)
                        output_tensors[f"{prefix}.attn.{comp}.alpha"] = torch.tensor(64.0, dtype=torch.bfloat16)
                    
                    # Inject into MLP gate for conceptual persistence
                    output_tensors[f"{prefix}.mlp.gate_proj.lora_down.weight"] = nd_m.to("cpu", dtype=torch.bfloat16)
                    output_tensors[f"{prefix}.mlp.gate_proj.lora_up.weight"] = nu_m.to("cpu", dtype=torch.bfloat16)
                    output_tensors[f"{prefix}.mlp.gate_proj.alpha"] = torch.tensor(64.0, dtype=torch.bfloat16)

                    if b_idx % 4 == 0:
                        yield (b_idx / DIT_BLOCKS), f"Projecting manifold {b_idx}/30..."
                    
                    del delta_src, proj_attn, proj_mlp, nd_a, nu_a, nd_m, nu_m

                # 3. Metadata and Finalization
                meta = {
                    "ss_network_dim": "64", 
                    "ss_network_alpha": "64", 
                    "modelspec.title": "Converted Concept LoRA (SDXL->Turbo)",
                    "modelspec.description": "Neural Bridge migration with Frobenius-Optimal Scaling"
                }
                
                if save_to_workspace:
                    self.workspace.add_model(output_name, output_tensors, meta)
                else:
                    out_p = NamingManager.resolve_output_path(sdxl_lora_path, output_name, "_CasTurbo_v3")
                    SafeStreamer.save_tensors(output_tensors, out_p, meta)

                yield 1.0, "Conversion finished successfully."
                
        except Exception as e:
            Logger.error(f"Bridge Failed: {e}")
            yield 0.0, f"Error: {str(e)}"