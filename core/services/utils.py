# core/services/utils.py

import torch
from typing import List, Generator, Tuple
from .base import BaseService
from ..configs import UtilsConfig
from ..structs import ModelReference
from ..io_manager import SafeStreamer
from ..format_handler import FormatHandler
from ..naming import NamingManager
from ..logger import Logger

class UtilsService(BaseService):
    def process(
        self, 
        config: UtilsConfig, 
        inputs: List[ModelReference], 
        output_target: str
    ) -> Generator[Tuple[float, str], None, None]:
        
        is_batch = len(inputs) > 1
        total = len(inputs)
        
        dtype_map = {"FP32": torch.float32, "FP16": torch.float16, "BF16": torch.bfloat16}
        target_dtype = dtype_map.get(config.precision)

        for idx, ref in enumerate(inputs):
            yield idx / total, f"Processing {ref.name}..."
            
            try:
                with self._resolve_source(ref) as io:
                    groups = FormatHandler.group_keys(io.keys)
                    output_tensors = {}
                    processed_keys = set()
                    
                    for grp in groups:
                        processed_keys.add(grp.down_key)
                        processed_keys.add(grp.up_key)
                        if grp.alpha_key: processed_keys.add(grp.alpha_key)
                        
                        ld = io.get_tensor(grp.down_key)
                        lu = io.get_tensor(grp.up_key)
                        la = io.get_tensor(grp.alpha_key) if grp.alpha_key else None
                        
                        rank = float(ld.shape[0])
                        alpha = float(la.item()) if la else rank
                        
                        final_alpha = alpha
                        if config.alpha_equals_rank: final_alpha = rank
                        elif config.target_alpha: final_alpha = config.target_alpha
                        
                        if final_alpha != alpha and final_alpha > 0:
                            scale = alpha / final_alpha
                            lu = lu.float() * scale
                        
                        nk_d = FormatHandler.fix_key_name(grp.down_key) if config.normalize_keys else grp.down_key
                        nk_u = FormatHandler.fix_key_name(grp.up_key) if config.normalize_keys else grp.up_key
                        nk_a = FormatHandler.fix_key_name(grp.alpha_key) if config.normalize_keys and grp.alpha_key else (grp.alpha_key or f"{nk_d.split('.')[0]}.alpha")
                        
                        if target_dtype:
                            ld = ld.to(dtype=target_dtype)
                            lu = lu.to(dtype=target_dtype)
                            
                        output_tensors[nk_d] = ld
                        output_tensors[nk_u] = lu
                        output_tensors[nk_a] = torch.tensor(final_alpha, dtype=target_dtype or ld.dtype)

                    for k in io.keys:
                        if k not in processed_keys:
                            t = io.get_tensor(k)
                            if target_dtype and t.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                                t = t.to(dtype=target_dtype)
                            output_tensors[k] = t

                    meta = io.metadata.copy()
                    if config.save_to_workspace:
                        out_name = output_target if not is_batch else f"{output_target}_{ref.name}"
                        self.workspace.add_model(out_name, output_tensors, meta)
                    else:
                        out_path = NamingManager.resolve_output_path(ref.path, output_target, "_opt", is_batch)
                        SafeStreamer.save_tensors(output_tensors, out_path, meta)
                    
                    del output_tensors
                    self.garbage_collect()

            except Exception as e:
                Logger.error(f"Utils error: {e}")
        
        yield 1.0, "Complete"