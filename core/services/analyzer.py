
import torch
import numpy as np
import concurrent.futures
from typing import Generator, Tuple, Dict, Union, List
from .base import BaseService
from ..structs import ModelReference
from ..io_manager import SafeStreamer
from ..math import MathKernel
from ..model_specs import ModelRegistry
from ..architectures.base import UnknownArchitecture
from ..logger import Logger
from ..structs_assembly import LoRAAssembly

class AnalyzerService(BaseService):
    def _analyze_block_task(self, args: Dict) -> Dict:
        device = args["device"]
        ld = args["ld"].to(device, non_blocking=True)
        lu = args["lu"].to(device, non_blocking=True)
        alpha, rank = args["alpha"], args["rank"]
        scale = alpha / rank if rank > 0 else 1.0
        
        if args.get("s") is not None:
            s = args["s"].cpu() 
            e = torch.sqrt(torch.sum(s**2)).item()
        else:
            s, e = MathKernel.get_spectrum_fast(ld, lu, scale)
        
        k, m = MathKernel.calculate_stats_estimated(ld, lu, scale)
        del ld, lu
        
        return {
            "spectrum": s, "energy": e, "kurtosis": k.item(), "magnitude": m.item(),
            "rank": rank, "alpha": alpha, "block_idx": args["block_idx"],
            "comp_idx": args["comp_idx"], "region": args["region"]
        }

    def analyze(self, ref: Union[str, ModelReference]) -> Generator[Tuple[float, str, Dict], None, None]:
        yield 0.1, "Loading headers...", {}
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_gpu = compute_device.type == "cuda"
        
        assembly = None
        path = ref.path if hasattr(ref, 'path') else ref
        
        if self.workspace.exists(path):
            assembly = self.workspace.get_model(path).assembly.clone()
        else:
            with self._resolve_source(ref) as io:
                if io.load_error:
                    yield 1.0, f"Error: {io.load_error}", {}
                    return
                tensors = io.load_state_dict()
                assembly = LoRAAssembly.from_state_dict(tensors, io.metadata)
                del tensors

        spec = ModelRegistry.get_spec(assembly.get_raw_keys())
        
        if isinstance(spec, UnknownArchitecture):
            msg = f"Error: Architecture not supported for {path}."
            Logger.error(msg)
            yield 1.0, msg, {}
            return
            
        # ПРОВЕРКА НА ПУСТУЮ МОДЕЛЬ ИЛИ ОШИБКУ ПАРСИНГА
        if not assembly.modules:
            msg = f"Error: No valid LoRA modules found for architecture '{spec.name}'. The file might be empty or uses an unsupported PEFT format."
            Logger.error(msg)
            yield 1.0, msg, {}
            return
        
        tasks =[]
        for name, mod in assembly.modules.items():
            if mod.is_decomposed: mod.compose() 
            b_idx = spec.get_block_number(name)
            c_idx = spec.get_component_idx(name)
            region = spec.get_region(b_idx)

            tasks.append({
                "ld": mod.down, "lu": mod.up, "s": mod.s if mod.is_decomposed else None,
                "alpha": mod.alpha, "rank": mod.rank,
                "block_idx": b_idx, "comp_idx": c_idx, "region": region,
                "device": compute_device
            })

        results =[]
        h_dim = spec.get_heatmap_dimensions()
        heatmap = np.zeros(h_dim) if h_dim != (0, 0) else None
        max_workers = 6 if is_gpu else min(32, (torch.get_num_threads() or 4))
        
        yield 0.2, f"Starting Analysis...", {}

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
                        if 0 <= b < h_dim[0] and 0 <= c < h_dim[1]: heatmap[b, c] = res["energy"]
                except Exception as exc: Logger.error(f"Task error: {exc}")
                completed += 1
                if completed % 10 == 0: yield 0.2 + (0.7 * (completed / total_tasks)), f"Analyzing...", {}

        if is_gpu: torch.cuda.empty_cache()
        if not results:
            yield 1.0, "Error: Analysis failed during computation.", {}
            return

        avg_rank = sum(r["rank"] for r in results) / len(results)
        avg_alpha = sum(r["alpha"] for r in results) / len(results)
        max_len = max([len(r["spectrum"]) for r in results] + [0])
        final_spectrum = np.zeros(max_len, dtype=np.float32)
        for r in results:
            s = r["spectrum"].numpy() if isinstance(r["spectrum"], torch.Tensor) else r["spectrum"]
            if len(s) < max_len: final_spectrum[:len(s)] += s
            else: final_spectrum += s
        final_spectrum /= len(results)

        block_energy = {reg: [] for reg in spec.get_regions()}
        for r in results:
            reg = r["region"]
            if reg in block_energy: block_energy[reg].append(r["energy"])
        
        avg_block_energy = {k: float(np.mean(v)) if v else 0.0 for k, v in block_energy.items()}

        data = {
            "avg_rank": int(avg_rank), "avg_alpha": avg_alpha, "model_name": spec.name,
            "heatmap": heatmap.tolist() if heatmap is not None else None,
            "kurtosis": sum(r["kurtosis"] for r in results) / len(results),
            "magnitude": sum(r["magnitude"] for r in results) / len(results),
            "block_energy": avg_block_energy,
            "knee_rank": MathKernel.find_knee_point(final_spectrum.tolist()),
            "spectrum": final_spectrum.tolist()
        }
        self.garbage_collect()
        yield 1.0, "Done", data