# core/pipeline/operations.py

import torch
import time
import os
import concurrent.futures
import threading
import numpy as np
from typing import List, Generator, Tuple, Dict, Union
from collections import defaultdict

from ..io_manager import SafeStreamer
from ..math import MathKernel
from ..format_handler import FormatHandler
from ..model_specs import ModelRegistry, ZImageTurboSpec
from ..naming import NamingManager
from ..logger import Logger
from .transform import TransformMixin
from .ops_combine import CombineMixin

class OperationsMixin(TransformMixin, CombineMixin):
    """
    Aggregates all operation mixins.
    Implements High-Performance Analysis Pipeline with Heatmap support.
    """
    
    def analyze_spectrum_gen(self, lora_path: str) -> Generator[Tuple[float, str, Dict], None, None]:
        yield 0.0, "Initializing Analysis...", {}
        
        try:
            source = self._resolve_source(lora_path)
            io = SafeStreamer(source, device="cpu")
            if io.load_error:
                yield 1.0, f"Error: {io.load_error}", {}
                return

            yield 0.1, "Loading tensors to RAM...", {}
            tensors = io.load_state_dict()

            spec = ModelRegistry.get_spec(list(tensors.keys()))
            groups = FormatHandler.group_keys(list(tensors.keys()))
            if not groups:
                yield 1.0, "No LoRA blocks found.", {}
                return

            tasks = []
            for grp in groups:
                ld, lu = tensors.get(grp.down_key), tensors.get(grp.up_key)
                if ld is None or lu is None: continue
                
                block_idx = spec.get_block_number(grp.base_name)
                comp_idx = spec.get_component_idx(grp.base_name)
                region = spec.get_region(block_idx)
                
                alpha_val = float(ld.shape[0])
                if grp.alpha_key and grp.alpha_key in tensors:
                    alpha_val = float(tensors[grp.alpha_key].item())
                
                tasks.append({
                    "ld": ld, "lu": lu, "alpha": alpha_val, 
                    "block_idx": block_idx, "comp_idx": comp_idx,
                    "region": region, "rank": ld.shape[0]
                })

            results = []
            heatmap_data = np.zeros((30, 7)) if isinstance(spec, ZImageTurboSpec) else None
            
            # FIX: Lock for shared resource (heatmap_data)
            heatmap_lock = threading.Lock()
            
            max_workers = min(os.cpu_count() or 4, 16)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {executor.submit(self._analyze_block_cpu, t): t for t in tasks}
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_task):
                    res = future.result()
                    results.append(res)
                    
                    if heatmap_data is not None:
                        b_idx, c_idx = res["block_idx"], res["comp_idx"]
                        if 0 <= b_idx < 30 and 0 <= c_idx < 7:
                            with heatmap_lock:
                                heatmap_data[b_idx, c_idx] = res["energy"]
                                
                    completed_count += 1
                    if completed_count % 20 == 0:
                        yield 0.2 + (0.7 * (completed_count / len(tasks))), f"Analyzing layers...", {}

            energy_by_region = defaultdict(list)
            for r in results:
                energy_by_region[r["region"]].append(r["energy"])
            
            avg_energy = {reg: sum(vals)/len(vals) for reg, vals in energy_by_region.items()}
            
            e_mid = avg_energy.get("MID", 0.0)
            e_rest = avg_energy.get("IN", 0.0) + avg_energy.get("OUT", 0.0)
            mid_dom = e_mid / e_rest if e_rest > 0 else 1.0

            spectra = [r["spectrum"] for r in results]
            padded_spectra = torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True, padding_value=0.0)
            final_spectrum = torch.mean(padded_spectra, dim=0).tolist()
            
            knee_rank = MathKernel.find_knee_point(final_spectrum)
            intrinsic_rank = MathKernel.calculate_intrinsic_rank_from_spectrum(final_spectrum, threshold=0.95)

            result_data = {
                "spectrum": final_spectrum,
                "knee_rank": knee_rank,
                "intrinsic_rank": intrinsic_rank,
                "block_energy": dict(avg_energy),
                "mid_dominance": mid_dom,
                "heatmap": heatmap_data.tolist() if heatmap_data is not None else None,
                "kurtosis": sum(r["kurtosis"] for r in results) / len(results),
                "magnitude": sum(r["magnitude"] for r in results) / len(results),
                "avg_rank": int(sum(r["rank"] for r in results) / len(results)),
                "avg_alpha": sum(r["alpha"] for r in results) / len(results),
                "model_name": spec.name
            }
            
            self.garbage_collect()
            yield 1.0, "Done", result_data
            
        except Exception as e:
            Logger.error(f"Analysis pipeline crash: {e}")
            yield 0.0, f"Error: {e}", {}

    def _analyze_block_cpu(self, task: Dict) -> Dict:
        ld, lu, alpha, rank = task["ld"], task["lu"], task["alpha"], task["rank"]
        scale = alpha / rank if rank > 0 else 1.0
        s, e = MathKernel.get_spectrum_fast(ld, lu, scale)
        k, m = MathKernel.calculate_stats_estimated(ld, lu, scale)
        return {
            "spectrum": s, "energy": e, "kurtosis": k.item(), "magnitude": m.item(),
            "region": task["region"], "block_idx": task["block_idx"], "comp_idx": task["comp_idx"],
            "rank": rank, "alpha": alpha
        }