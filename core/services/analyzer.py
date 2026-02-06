# core/services/analyzer.py

import torch
import numpy as np
from typing import Generator, Tuple, Dict
from .base import BaseService
from ..structs import ModelReference
from ..io_manager import SafeStreamer
from ..format_handler import FormatHandler
from ..math import MathKernel
from ..model_specs import ModelRegistry, S3DiTSpec

class AnalyzerService(BaseService):
    def analyze(self, ref: ModelReference) -> Generator[Tuple[float, str, Dict], None, None]:
        yield 0.1, "Loading...", {}
        
        with self._resolve_source(ref) as io:
            tensors = io.load_state_dict()
            
            spec = ModelRegistry.get_spec(list(tensors.keys()))
            groups = FormatHandler.group_keys(list(tensors.keys()))
            
            results = []
            # Only create heatmap for S3-DiT architecture
            heatmap = np.zeros((30, 7)) if isinstance(spec, S3DiTSpec) else None
            
            total = len(groups)
            for i, grp in enumerate(groups):
                ld = tensors[grp.down_key].float()
                lu = tensors[grp.up_key].float()
                
                rank = ld.shape[0]
                alpha = rank
                if grp.alpha_key: alpha = tensors[grp.alpha_key].item()
                
                scale = alpha / rank if rank > 0 else 1.0
                
                s, e = MathKernel.get_spectrum_fast(ld, lu, scale)
                k, m = MathKernel.calculate_stats_estimated(ld, lu, scale)
                
                b_idx = spec.get_block_number(grp.base_name)
                c_idx = spec.get_component_idx(grp.base_name)
                
                if heatmap is not None and 0 <= b_idx < 30:
                    heatmap[b_idx, c_idx] = e
                
                results.append({
                    "spectrum": s, "energy": e, "kurtosis": k.item(), "magnitude": m.item(),
                    "rank": rank, "alpha": alpha, "region": spec.get_region(b_idx)
                })
                
                if i % 50 == 0: yield 0.2 + (0.7 * i/total), "Analyzing...", {}

            avg_rank = sum(r["rank"] for r in results) / len(results)
            avg_alpha = sum(r["alpha"] for r in results) / len(results)
            
            data = {
                "avg_rank": int(avg_rank),
                "avg_alpha": avg_alpha,
                "model_name": spec.name,
                "heatmap": heatmap.tolist() if heatmap is not None else None,
                "kurtosis": sum(r["kurtosis"] for r in results) / len(results),
                "magnitude": sum(r["magnitude"] for r in results) / len(results),
                "block_energy": {}, 
                "knee_rank": 0,
                "spectrum": []
            }
            
            yield 1.0, "Done", data