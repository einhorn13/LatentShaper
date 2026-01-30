# core/pipeline/transform/resize.py

import torch
import os
import queue
import threading
from typing import List, Generator, Tuple, Union
from ...io_manager import SafeStreamer
from ...math import MathKernel
from ...format_handler import FormatHandler
from ...naming import NamingManager
from ...logger import Logger

class ResizeMixin:
    """Logic for resizing (Downscaling) LoRA ranks using Producer-Consumer pattern."""

    def resize_lora_gen(
        self, 
        lora_paths: Union[str, List[str]], 
        output_target: str,
        new_rank: int, 
        auto_rank_threshold: float = 0.0,
        save_to_workspace: bool = False
    ) -> Generator[Tuple[float, str], None, None]:
        
        inputs = [lora_paths] if isinstance(lora_paths, str) else lora_paths
        is_batch = len(inputs) > 1
        total_files = len(inputs)
        max_workers = min(os.cpu_count(), 4) # Use multiple threads for CPU-bound tasks if needed

        for idx, lora_path_or_name in enumerate(inputs):
            filename = os.path.basename(lora_path_or_name)
            source = self._resolve_source(lora_path_or_name)
            source_meta = self._resolve_metadata(lora_path_or_name)

            yield idx / total_files, f"[{idx+1}/{total_files}] {filename}: Reading..."
            
            try:
                with SafeStreamer(source, device=self.device, metadata=source_meta) as io:
                    groups = FormatHandler.group_keys(io.keys)
                    if not groups: continue

                    output_tensors = {}
                    max_actual_rank = 0
                    total_groups = len(groups)
                    
                    data_queue = queue.Queue(maxsize=10)
                    stop_event = threading.Event()
                    
                    # --- Loader Thread ---
                    def _loader():
                        try:
                            for grp in groups:
                                if stop_event.is_set(): break
                                ld = io.get_tensor(grp.down_key)
                                lu = io.get_tensor(grp.up_key)
                                if ld is None or lu is None: continue
                                
                                at = io.get_tensor(grp.alpha_key) if grp.alpha_key else None
                                data_queue.put((grp, ld, lu, at))
                        except Exception as e:
                            Logger.error(f"Loader error: {e}")
                        finally:
                            data_queue.put(None)

                    t = threading.Thread(target=_loader, daemon=True)
                    t.start()

                    # --- Main Compute Loop ---
                    completed = 0
                    while True:
                        try:
                            item = data_queue.get(timeout=2)
                            if item is None: break
                            
                            grp, ld, lu, at = item
                            
                            # Move to Device
                            d_ld = ld.to(self.device, non_blocking=True).float()
                            d_lu = lu.to(self.device, non_blocking=True).float()
                            
                            current_rank = d_ld.shape[0]
                            current_alpha = float(current_rank)
                            if at is not None:
                                current_alpha = float(at.item())
                            
                            input_scale = current_alpha / current_rank if current_rank > 0 else 1.0
                            delta = (d_lu @ d_ld) * input_scale
                            
                            # FIX: Safety check for NaN/Inf before SVD
                            if torch.isnan(delta).any() or torch.isinf(delta).any():
                                delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
                                
                            nd, nu, ar = MathKernel.svd_decomposition(delta, new_rank, auto_rank_threshold)
                            
                            output_tensors[grp.down_key] = nd.to("cpu", dtype=torch.bfloat16)
                            output_tensors[grp.up_key] = nu.to("cpu", dtype=torch.bfloat16)
                            if grp.alpha_key:
                                output_tensors[grp.alpha_key] = torch.tensor(float(ar), dtype=torch.bfloat16)
                            
                            max_actual_rank = max(max_actual_rank, ar)
                            completed += 1
                            
                            del delta, nd, nu, d_ld, d_lu
                            
                            if completed % 5 == 0:
                                local_p = completed / total_groups
                                global_p = (idx + local_p) / total_files
                                yield global_p, f"[{idx+1}/{total_files}] {filename}: Resizing ({int(local_p*100)}%)..."
                                
                        except queue.Empty:
                            if not t.is_alive(): break
                            continue
                        except Exception as e:
                            Logger.error(f"Compute error: {e}")
                            stop_event.set()
                            break
                    
                    t.join()

                    # --- Save ---
                    yield (idx + 0.95) / total_files, f"[{idx+1}/{total_files}] {filename}: Saving..."
                    meta = io.metadata.copy()
                    meta.update({"ss_network_dim": str(max_actual_rank), "ss_network_alpha": str(max_actual_rank), "modelspec.title": "Z-Image Resized LoRA"})
                    
                    if save_to_workspace:
                        out_name = output_target if not is_batch else f"{output_target}_{filename}"
                        self.workspace.add_model(out_name, output_tensors, meta, {"source": filename, "op": "resize"})
                    else:
                        current_out = NamingManager.resolve_output_path(lora_path_or_name, output_target, suffix=f"_rank{new_rank}", is_batch=is_batch)
                        SafeStreamer.save_tensors(output_tensors, current_out, metadata=meta)
            
            except Exception as e:
                Logger.error(f"Error resizing {filename}: {e}")
                continue
        
        # Explicit GC after batch processing
        self.garbage_collect()
        yield 1.0, f"Batch Complete."