# core/services/merger.py

import torch
import time
from typing import List, Generator, Tuple, Dict
from collections import defaultdict
from .base import BaseService
from ..configs import MergeConfig, CheckpointMergeConfig
from ..structs import ModelReference
from ..io_manager import SafeStreamer
from ..math import MathKernel
from ..format_handler import FormatHandler
from ..naming import NamingManager
from ..checkpoint_merger import CheckpointMerger
from ..logger import Logger

class MergerService(BaseService):
    def merge_loras(
        self, 
        config: MergeConfig, 
        inputs: List[ModelReference], 
        output_target: str
    ) -> Generator[Tuple[float, str], None, None]:
        
        yield 0.0, "Initializing Merge..."
        
        try:
            streamers = [self._resolve_source(ref) for ref in inputs]
            
            block_map = defaultdict(list)
            for idx, s in enumerate(streamers):
                for grp in FormatHandler.group_keys(s.keys):
                    block_map[grp.base_name].append((idx, grp))
            
            unique_blocks = sorted(list(block_map.keys()))
            output_tensors = {}
            max_rank = 0
            
            for i, block_name in enumerate(unique_blocks):
                deltas = []
                weights = []
                
                for s_idx, grp in block_map[block_name]:
                    ratio = config.ratios[s_idx]
                    if ratio == 0: continue
                    
                    s = streamers[s_idx]
                    ld = s.get_tensor(grp.down_key).to(self.device).float()
                    lu = s.get_tensor(grp.up_key).to(self.device).float()
                    
                    alpha = ld.shape[0]
                    if grp.alpha_key:
                        at = s.get_tensor(grp.alpha_key)
                        alpha = float(at.item())
                    
                    scale = (alpha / ld.shape[0]) * ratio
                    delta = (lu @ ld) * scale
                    deltas.append(delta)
                    weights.append(ratio)
                
                if not deltas: continue
                
                final_delta = None
                if config.algorithm == "SVD":
                    final_delta = sum(deltas)
                elif config.algorithm == "TIES":
                    final_delta = MathKernel.ties_trim_and_elect_streaming(deltas, weights, config.ties_density)
                elif config.algorithm == "Median":
                    final_delta = MathKernel.median_merge(deltas)
                else:
                    final_delta = sum(deltas)
                
                if config.global_strength != 1.0:
                    final_delta *= config.global_strength
                
                nd, nu, nr = MathKernel.svd_decomposition(final_delta, config.rank, config.auto_rank_threshold)
                max_rank = max(max_rank, nr)
                
                base_grp = block_map[block_name][0][1]
                output_tensors[base_grp.down_key] = nd.to("cpu", dtype=torch.bfloat16)
                output_tensors[base_grp.up_key] = nu.to("cpu", dtype=torch.bfloat16)
                if base_grp.alpha_key:
                    output_tensors[base_grp.alpha_key] = torch.tensor(float(nr), dtype=torch.bfloat16)
                
                if i % 10 == 0:
                    yield i / len(unique_blocks), "Merging..."

            meta = {"ss_network_dim": str(max_rank)}
            if config.save_to_workspace:
                self.workspace.add_model(output_target, output_tensors, meta)
            else:
                out_path = NamingManager.resolve_merge_path(output_target)
                SafeStreamer.save_tensors(output_tensors, out_path, meta)
                
            for s in streamers: s.__exit__(None, None, None)
            yield 1.0, "Merge Complete"

        except Exception as e:
            Logger.error(f"Merge Error: {e}")
            yield 0.0, f"Error: {e}"

    def merge_checkpoints(
        self,
        config: CheckpointMergeConfig,
        inputs: List[ModelReference],
        output_target: str
    ) -> Generator[Tuple[float, str], None, None]:
        
        paths = {
            'A': inputs[0].path if len(inputs) > 0 else None,
            'B': inputs[1].path if len(inputs) > 1 else None,
            'C': inputs[2].path if len(inputs) > 2 else None
        }
        
        params_dict = {
            "mode": config.mode,
            "te_policy": config.te_policy,
            "vae_policy": config.vae_policy,
            "precision": config.precision,
            "lora_strength": config.lora_strength
        }
        
        gen = CheckpointMerger.merge_streamed(
            paths, 
            config.lora_path, 
            config.weights, 
            params_dict
        )
        
        output_sd = None
        for prog, msg, res in gen:
            if res: output_sd = res
            yield prog, msg
            
        if output_sd:
            out_path = NamingManager.resolve_merge_path(output_target, "merged_ckpt.safetensors")
            SafeStreamer.save_tensors(output_sd, out_path)
            yield 1.0, "Saved"