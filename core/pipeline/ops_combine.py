# core/pipeline/ops_combine.py

import torch
import time
import os
import queue
import threading
from typing import List, Generator, Tuple
from collections import defaultdict
from contextlib import ExitStack

from ..io_manager import SafeStreamer
from ..math import MathKernel
from ..format_handler import FormatHandler
from ..naming import NamingManager
from ..logger import Logger

class CombineMixin:
    def merge_lora_gen(self, lora_paths: List[str], ratios: List[float], output_path: str, target_rank: int = 64, algorithm: str = "SVD", global_strength: float = 1.0, auto_rank_threshold: float = 0.0, pruning_threshold: float = 0.0, ties_density: float = 0.3, save_to_workspace: bool = False) -> Generator[Tuple[float, str], None, None]:
        
        start_time = time.time()
        yield 0.0, "Initializing Merge..."
        
        try:
            with ExitStack() as stack:
                streamers = [stack.enter_context(SafeStreamer(self._resolve_source(p), device="cpu", metadata=self._resolve_metadata(p))) for p in lora_paths]
                block_map = defaultdict(list)
                for idx, s in enumerate(streamers):
                    for g in FormatHandler.group_keys(s.keys):
                        block_map[g.base_name].append((idx, g))

                unique_blocks = sorted(list(block_map.keys()))
                total_blocks, output_tensors, max_final_rank = len(unique_blocks), {}, 0
                data_queue = queue.Queue(maxsize=2)
                stop_event = threading.Event()

                def _loader():
                    try:
                        for block_name in unique_blocks:
                            if stop_event.is_set(): break
                            block_data = []
                            for s_idx, grp in block_map[block_name]:
                                if ratios[s_idx] == 0: continue
                                s = streamers[s_idx]
                                ld, lu = s.get_tensor(grp.down_key), s.get_tensor(grp.up_key)
                                if ld is None or lu is None: continue
                                at = s.get_tensor(grp.alpha_key) if grp.alpha_key else None
                                block_data.append((ld, lu, float(at.item()) if at is not None else float(ld.shape[0]), ratios[s_idx]))
                            if block_data: data_queue.put((block_name, block_data))
                    finally: data_queue.put(None)

                threading.Thread(target=_loader, daemon=True).start()

                completed = 0
                while True:
                    item = data_queue.get()
                    if item is None: break
                    block_name, loaded_data = item
                    
                    if not loaded_data: 
                        completed += 1
                        continue

                    # Process
                    final_rank = 0
                    if algorithm == "Concat":
                        t_concat = [(ld.to(self.device), lu.to(self.device), (alpha/ld.shape[0]) * r * global_strength) for ld, lu, alpha, r in loaded_data]
                        nd, nu, final_rank = MathKernel.merge_concat(t_concat)
                        output_tensors[f"{block_name}.lora_down.weight"] = nd.to("cpu", dtype=torch.bfloat16)
                        output_tensors[f"{block_name}.lora_up.weight"] = nu.to("cpu", dtype=torch.bfloat16)
                        output_tensors[f"{block_name}.alpha"] = torch.tensor(float(final_rank), dtype=torch.bfloat16)
                        del t_concat
                        
                    elif algorithm == "SLERP":
                        # Support ONLY 2 models for standard SLERP, fallback to Weighted Sum for >2
                        if len(loaded_data) == 2:
                            (ld1, lu1, a1, r1), (ld2, lu2, a2, r2) = loaded_data
                            delta1 = (lu1.to(self.device).float() @ ld1.to(self.device).float()) * (a1/ld1.shape[0])
                            delta2 = (lu2.to(self.device).float() @ ld2.to(self.device).float()) * (a2/ld2.shape[0])
                            
                            # Normalized interpolation factor t based on ratios
                            total_ratio = r1 + r2
                            t = r2 / total_ratio if total_ratio > 0 else 0.5
                            
                            total_delta = MathKernel.slerp(delta1, delta2, t)
                            if global_strength != 1.0: total_delta *= global_strength
                            
                            nd, nu, final_rank = MathKernel.svd_decomposition(total_delta, target_rank, auto_rank_threshold)
                            output_tensors[f"{block_name}.lora_down.weight"], output_tensors[f"{block_name}.lora_up.weight"] = nd.to("cpu", dtype=torch.bfloat16), nu.to("cpu", dtype=torch.bfloat16)
                            output_tensors[f"{block_name}.alpha"] = torch.tensor(float(final_rank), dtype=torch.bfloat16)
                            del delta1, delta2, total_delta, nd, nu
                        else:
                            # Fallback to SVD sum for > 2 models
                            total_delta = None
                            for ld, lu, alpha, r in loaded_data:
                                curr = (lu.to(self.device).float() @ ld.to(self.device).float()) * ((alpha / ld.shape[0]) * r)
                                if total_delta is None: total_delta = curr
                                else: total_delta.add_(curr)
                                del curr
                            if total_delta is not None:
                                if global_strength != 1.0: total_delta *= global_strength
                                nd, nu, final_rank = MathKernel.svd_decomposition(total_delta, target_rank, auto_rank_threshold)
                                output_tensors[f"{block_name}.lora_down.weight"], output_tensors[f"{block_name}.lora_up.weight"] = nd.to("cpu", dtype=torch.bfloat16), nu.to("cpu", dtype=torch.bfloat16)
                                output_tensors[f"{block_name}.alpha"] = torch.tensor(float(final_rank), dtype=torch.bfloat16)
                                del total_delta, nd, nu
                    
                    elif algorithm == "TIES":
                        deltas, ws = [], []
                        for ld, lu, alpha, r in loaded_data:
                            deltas.append((lu.to(self.device).float() @ ld.to(self.device).float()) * (alpha / ld.shape[0]))
                            ws.append(r)
                        total_delta = MathKernel.ties_trim_and_elect_streaming(deltas, ws, density=ties_density)
                        if global_strength != 1.0: total_delta *= global_strength
                        nd, nu, final_rank = MathKernel.svd_decomposition(total_delta, target_rank, auto_rank_threshold)
                        output_tensors[f"{block_name}.lora_down.weight"], output_tensors[f"{block_name}.lora_up.weight"] = nd.to("cpu", dtype=torch.bfloat16), nu.to("cpu", dtype=torch.bfloat16)
                        output_tensors[f"{block_name}.alpha"] = torch.tensor(float(final_rank), dtype=torch.bfloat16)
                        del deltas, total_delta, nd, nu
                        
                    else:
                        # Standard SVD / Orthogonal
                        total_delta = None
                        for ld, lu, alpha, r in loaded_data:
                            curr = (lu.to(self.device).float() @ ld.to(self.device).float()) * ((alpha / ld.shape[0]) * r)
                            if total_delta is None: total_delta = curr
                            else:
                                if algorithm == "Orthogonal": curr = MathKernel.orthogonalize_update(total_delta, curr)
                                total_delta.add_(curr)
                            del curr
                        if total_delta is not None:
                            if global_strength != 1.0: total_delta *= global_strength
                            nd, nu, final_rank = MathKernel.svd_decomposition(total_delta, target_rank, auto_rank_threshold)
                            output_tensors[f"{block_name}.lora_down.weight"], output_tensors[f"{block_name}.lora_up.weight"] = nd.to("cpu", dtype=torch.bfloat16), nu.to("cpu", dtype=torch.bfloat16)
                            output_tensors[f"{block_name}.alpha"] = torch.tensor(float(final_rank), dtype=torch.bfloat16)
                            del total_delta, nd, nu

                    max_final_rank = max(max_final_rank, final_rank)
                    completed += 1
                    for d in loaded_data: del d
                    if completed % 5 == 0: yield min(completed / total_blocks, 0.9), f"Merging blocks ({completed}/{total_blocks})..."

                yield 0.95, "Saving..."
                meta = {"ss_network_dim": str(max_final_rank), "ss_network_alpha": str(max_final_rank)}
                if save_to_workspace: self.workspace.add_model(output_path, output_tensors, meta)
                else: SafeStreamer.save_tensors(output_tensors, NamingManager.resolve_merge_path(output_path), meta)
            
        except Exception as e:
            yield 0.0, f"Merge Error: {e}"
        self.garbage_collect()
        yield 1.0, "Done"