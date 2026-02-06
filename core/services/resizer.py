# core/services/resizer.py

import torch
import os
from typing import List, Generator, Tuple
from .base import BaseService
from ..configs import ResizeConfig
from ..structs import ModelReference
from ..io_manager import SafeStreamer
from ..math import MathKernel
from ..format_handler import FormatHandler
from ..naming import NamingManager
from ..logger import Logger

class ResizerService(BaseService):
    def process(
        self, 
        config: ResizeConfig, 
        inputs: List[ModelReference], 
        output_target: str
    ) -> Generator[Tuple[float, str], None, None]:
        
        is_batch = len(inputs) > 1
        total = len(inputs)
        
        for idx, ref in enumerate(inputs):
            yield idx / total, f"Resizing {ref.name}..."
            
            try:
                with self._resolve_source(ref) as io:
                    groups = FormatHandler.group_keys(io.keys)
                    output_tensors = {}
                    
                    for i, grp in enumerate(groups):
                        ld = io.get_tensor(grp.down_key).to(self.device).float()
                        lu = io.get_tensor(grp.up_key).to(self.device).float()
                        
                        rank = ld.shape[0]
                        alpha = rank
                        if grp.alpha_key:
                            at = io.get_tensor(grp.alpha_key)
                            alpha = float(at.item())
                        
                        scale = alpha / rank if rank > 0 else 1.0
                        delta = (lu @ ld) * scale
                        
                        nd, nu, nr = MathKernel.svd_decomposition(delta, config.rank, config.auto_rank_threshold)
                        
                        output_tensors[grp.down_key] = nd.to("cpu", dtype=torch.bfloat16)
                        output_tensors[grp.up_key] = nu.to("cpu", dtype=torch.bfloat16)
                        if grp.alpha_key:
                            output_tensors[grp.alpha_key] = torch.tensor(float(nr), dtype=torch.bfloat16)
                            
                        if i % 20 == 0:
                            yield (idx + (i/len(groups))) / total, f"Resizing blocks..."

                    meta = io.metadata.copy()
                    meta["ss_network_dim"] = str(config.rank)
                    
                    if config.save_to_workspace:
                        out_name = output_target if not is_batch else f"{output_target}_{ref.name}"
                        self.workspace.add_model(out_name, output_tensors, meta)
                    else:
                        out_path = NamingManager.resolve_output_path(ref.path, output_target, f"_rank{config.rank}", is_batch)
                        SafeStreamer.save_tensors(output_tensors, out_path, meta)
                    
                    del output_tensors
                    self.garbage_collect()
                    
            except Exception as e:
                Logger.error(f"Resize failed for {ref.name}: {e}")
                
        yield 1.0, "Resize Complete"