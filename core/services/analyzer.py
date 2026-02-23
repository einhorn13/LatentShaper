
import torch
import numpy as np
import concurrent.futures
from typing import Generator, Tuple, Dict, Union, List
from .base import BaseService
from ..structs import ModelReference
from ..io_manager import SafeStreamer
from ..format_handler import FormatHandler
from ..math import MathKernel
from ..model_specs import ModelRegistry, S3DiTSpec
from ..logger import Logger
from ..structs_assembly import LoRAAssembly

class AnalyzerService(BaseService):
    def _analyze_block_task(self, args: Dict) -> Dict:
        """
        Atomic task for a single thread.
        """
        device = args["device"]
        # If we have decomposed state (S), we can skip SVD!
        if args.get("s") is not None:
            s = args["s"].to(device, non_blocking=True)
            e = torch.sqrt(torch.sum(s**2)).item()
            # We can't easily calc kurtosis/magnitude from S alone without reconstructing,
            # but for optimization we might skip or approximate.
            # For now, let's assume we need full stats, so we need Down/Up.
            # If we strictly want SVD spectrum, S is enough.
            # But Analyzer reports kurtosis/magnitude of the weights.
            
            # Let's stick to standard path for full stats, but use S for spectrum if available.
            pass

        ld = args["ld"].to(device, non_blocking=True)
        lu = args["lu"].to(device, non_blocking=True)
        
        alpha = args["alpha"]
        rank = args["rank"]
        
        scale = alpha / rank if rank > 0 else 1.0
        
        # Spectrum
        # If S is provided (pre-calculated), use it
        if args.get("s") is not None:
            s = args["s"].cpu() # Already calculated
            e = torch.sqrt(torch.sum(s**2)).item()
        else:
            s, e = MathKernel.get_spectrum_fast(ld, lu, scale)
        
        # Stats
        k, m = MathKernel.calculate_stats_estimated(ld, lu, scale)
        
        del ld, lu
        
        return {
            "spectrum": s, 
            "energy": e, 
            "kurtosis": k.item(), 
            "magnitude": m.item(),
            "rank": rank, 
            "alpha": alpha, 
            "block_idx": args["block_idx"],
            "comp_idx": args["comp_idx"],
            "region": args["region"]
        }

    def analyze(self, ref: Union[str, ModelReference]) -> Generator[Tuple[float, str, Dict], None, None]:
        yield 0.1, "Loading headers...", {}
        
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_gpu = compute_device.type == "cuda"
        
        assembly = None
        
        # 1. Load Data (Optimized Path)
        path = ref.path if hasattr(ref, 'path') else ref
        
        if self.workspace.exists(path):
            # Fast path: Get Assembly directly
            assembly = self.workspace.get_model(path).assembly
        else:
            # Slow path: Load from disk
            with self._resolve_source(ref) as io:
                if io.load_error:
                    yield 1.0, f"Error: {io.load_error}", {}
                    return
                tensors = io.load_state_dict()
                assembly = LoRAAssembly.from_state_dict(tensors)
                del tensors

        spec = ModelRegistry.get_spec(list(assembly.modules.keys()))
        
        if not assembly.modules:
            yield 1.0, "No LoRA blocks detected.", {}
            return

        # 2. Prepare Tasks
        tasks = []
        for name, mod in assembly.modules.items():
            # Ensure we have Down/Up
            if mod.is_decomposed:
                # We can pass S directly to save SVD time!
                # But we need Down/Up for Kurtosis/Mag stats.
                # So we compose temporarily or use cached S.
                mod.compose() 
            
            ld = mod.down
            lu = mod.up
            
            b_idx = spec.get_block_number(name)
            c_idx = spec.get_component_idx(name)
            region = spec.get_region(b_idx)

            tasks.append({
                "ld": ld, "lu": lu, 
                "s": mod.s if mod.is_decomposed else None, # Pass cached S if available
                "alpha": mod.alpha, "rank": mod.rank,
                "block_idx": b_idx, "comp_idx": c_idx, "region": region,
                "device": compute_device
            })

        # 3. Configure Concurrency
        results = []
        heatmap = np.zeros((30, 7)) if isinstance(spec, S3DiTSpec) else None
        
        if is_gpu:
            max_workers = 6 
            desc_device = f"GPU ({torch.cuda.get_device_name(0)})"
        else:
            max_workers = min(32, (torch.get_num_threads() or 4))
            desc_device = "CPU"
        
        yield 0.2, f"Starting Analysis on {desc_device}...", {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_grp = {executor.submit(self._analyze_block_task, t): t for t in tasks}
            
            total_tasks = len(tasks)
            completed = 0
            
            for future in concurrent.futures.as_completed(future_to_grp):
                try:
                    res = future.result()
                    results.append(res)
                    
                    if heatmap is not None:
                        b, c = res["block_idx"], res["comp_idx"]
                        if 0 <= b < 30 and 0 <= c < 7:
                            heatmap[b, c] = res["energy"]

                except Exception as exc:
                    Logger.error(f"Task generated an exception: {exc}")
                
                completed += 1
                if completed % 10 == 0:
                    prog = 0.2 + (0.7 * (completed / total_tasks))
                    yield prog, f"Analyzing... ({completed}/{total_tasks})", {}

        # 4. Cleanup & Aggregation
        if is_gpu:
            torch.cuda.empty_cache()

        if not results:
            yield 1.0, "Analysis failed to produce results.", {}
            return

        avg_rank = sum(r["rank"] for r in results) / len(results)
        avg_alpha = sum(r["alpha"] for r in results) / len(results)
        
        max_len = 0
        for r in results:
            if len(r["spectrum"]) > max_len: max_len = len(r["spectrum"])
        
        final_spectrum = np.zeros(max_len, dtype=np.float32)
        for r in results:
            s = r["spectrum"].numpy() if isinstance(r["spectrum"], torch.Tensor) else r["spectrum"]
            if len(s) < max_len:
                final_spectrum[:len(s)] += s
            else:
                final_spectrum += s
        
        final_spectrum /= len(results)

        block_energy = {"IN": [], "MID": [], "OUT": []}
        for r in results:
            reg = r["region"]
            if reg in block_energy:
                block_energy[reg].append(r["energy"])
        
        avg_block_energy = {k: float(np.mean(v)) if v else 0.0 for k, v in block_energy.items()}

        data = {
            "avg_rank": int(avg_rank),
            "avg_alpha": avg_alpha,
            "model_name": spec.name,
            "heatmap": heatmap.tolist() if heatmap is not None else None,
            "kurtosis": sum(r["kurtosis"] for r in results) / len(results),
            "magnitude": sum(r["magnitude"] for r in results) / len(results),
            "block_energy": avg_block_energy,
            "knee_rank": MathKernel.find_knee_point(final_spectrum.tolist()),
            "spectrum": final_spectrum.tolist()
        }
        
        del tasks, results
        self.garbage_collect()
        
        yield 1.0, "Done", data