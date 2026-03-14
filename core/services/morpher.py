
import torch
import numpy as np
from typing import List, Generator, Tuple
from .base import BaseService
from ..configs import MorphConfig
from ..structs import ModelReference
from ..io_manager import SafeStreamer
from ..math import MathKernel
from ..format_handler import FormatHandler
from ..model_specs import ModelRegistry
from ..architectures.base import UnknownArchitecture
from ..tensor_processor import TensorProcessor
from ..naming import NamingManager
from ..logger import Logger
from ..structs_assembly import LoRAAssembly

class MorpherService(BaseService):
    def process(
        self, 
        config: MorphConfig, 
        inputs: List[ModelReference], 
        output_target: str
    ) -> Generator[Tuple[float, str], None, None]:
        
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
                assembly = None
                if self.workspace.exists(ref.path):
                    assembly = self.workspace.get_model(ref.path).assembly.clone()
                else:
                    with self._resolve_source(ref) as io:
                        tensors = io.load_state_dict()
                        assembly = LoRAAssembly.from_state_dict(tensors, io.metadata)

                # CRITICAL FIX: Use get_raw_keys()
                spec = ModelRegistry.get_spec(assembly.get_raw_keys())
                
                if isinstance(spec, UnknownArchitecture):
                    Logger.error(f"Morph failed: Architecture not supported for {ref.name}")
                    yield 1.0, f"Error: Architecture not supported for {ref.name}"
                    continue
                
                count = spec.block_count if spec.block_count > 0 else 30
                multipliers = np.ones(count)
                
                if config.eq_interpolate:
                    half = count // 2
                    multipliers[:half] = np.linspace(config.eq_in, config.eq_mid, half)
                    multipliers[half:] = np.linspace(config.eq_mid, config.eq_out, count - half)
                else:
                    for i in range(count):
                        reg = spec.get_region(i)
                        multipliers[i] = getattr(config, f"eq_{reg.lower()}", 1.0)

                keys_to_remove = []
                for name, mod in assembly.modules.items():
                    b_idx = spec.get_block_number(name)
                    eq_val = multipliers[b_idx] if 0 <= b_idx < len(multipliers) else 1.0
                    total_scale = eq_val * config.eq_global
                    
                    if mod.is_decomposed: mod.compose()
                    delta = mod.up.float() @ mod.down.float()
                    if config.fix_alpha and mod.alpha != mod.rank:
                        delta = MathKernel.rescale_alpha(delta, mod.alpha, mod.rank)
                    if config.temperature != 1.0:
                         delta = MathKernel.apply_eigen_temperature(delta, config.temperature)

                    delta = TensorProcessor.apply_filters(delta, tp_params, eq_factor=total_scale, b_idx=b_idx)
                    if delta is None: 
                        keys_to_remove.append(name)
                        continue
                    
                    nd, nu, nr = MathKernel.svd_decomposition(delta, mod.rank)
                    mod.down, mod.up, mod.alpha = nd.to(dtype=torch.bfloat16), nu.to(dtype=torch.bfloat16), float(nr)
                    mod.is_decomposed = False 

                for k in keys_to_remove: del assembly.modules[k]

                if config.save_to_workspace:
                    out_name = output_target if not is_batch else f"{output_target}_{ref.name}"
                    self.workspace.add_assembly(out_name, assembly)
                else:
                    out_path = NamingManager.resolve_output_path(ref.path, output_target, "_morphed", is_batch)
                    SafeStreamer.save_tensors(assembly.to_state_dict(), out_path, assembly.metadata)
                self.garbage_collect()
            except Exception as e: Logger.error(f"Morph failed: {e}")
        yield 1.0, "Morph Complete"