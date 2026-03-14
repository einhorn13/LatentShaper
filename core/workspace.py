
import torch
import os
import gc
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from .io_manager import SafeStreamer
from .logger import Logger
from .structs_assembly import LoRAAssembly

@dataclass
class VirtualModel:
    name: str
    assembly: LoRAAssembly
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        total = 0
        for mod in self.assembly.modules.values():
            if mod.down is not None: total += mod.down.numel() * mod.down.element_size()
            if mod.up is not None: total += mod.up.numel() * mod.up.element_size()
            if mod.s is not None: total += mod.s.numel() * mod.s.element_size()
            if mod.u is not None: total += mod.u.numel() * mod.u.element_size()
            if mod.vh is not None: total += mod.vh.numel() * mod.vh.element_size()
            
        for t in self.assembly.others.values():
            total += t.numel() * t.element_size()
        return total

class WorkspaceManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(WorkspaceManager, cls).__new__(cls)
                cls._instance._models: Dict[str, VirtualModel] = {}
        return cls._instance

    def list_models(self) -> List[str]:
        with self._lock:
            return list(self._models.keys())

    def get_model(self, name: str) -> Optional[VirtualModel]:
        with self._lock:
            return self._models.get(name)

    def exists(self, name: str) -> bool:
        with self._lock:
            return name in self._models

    def add_model(self, name: str, tensors: Dict[str, torch.Tensor], metadata: Dict[str, str] = None, info: Dict[str, Any] = None):
        with self._lock:
            final_name = name
            counter = 1
            while final_name in self._models:
                final_name = f"{name}_{counter}"
                counter += 1
            
            info = info or {}
            assembly = LoRAAssembly.from_state_dict(tensors, metadata)
            
            if "arch" not in info:
                from .model_specs import ModelRegistry
                spec = ModelRegistry.get_spec(assembly.get_raw_keys())
                info["arch"] = spec.name

            # СТАТИСТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ РАНГА (100% надежность)
            if info.get("rank", 0) == 0 and assembly.modules:
                ranks =[m.rank for m in assembly.modules.values() if m.rank > 0]
                if ranks:
                    # Находим моду (самый частый ранг)
                    info["rank"] = max(set(ranks), key=ranks.count)
                else:
                    info["rank"] = 0
            
            self._models[final_name] = VirtualModel(
                name=final_name,
                assembly=assembly,
                info=info
            )
            Logger.info(f"Workspace: Added '{final_name}' (Arch: {info['arch']}, Rank: {info.get('rank', 0)})")
            return final_name

    def add_assembly(self, name: str, assembly: LoRAAssembly, info: Dict[str, Any] = None):
        with self._lock:
            final_name = name
            counter = 1
            while final_name in self._models:
                final_name = f"{name}_{counter}"
                counter += 1
            
            info = info or {}
            if "arch" not in info:
                from .model_specs import ModelRegistry
                spec = ModelRegistry.get_spec(assembly.get_raw_keys())
                info["arch"] = spec.name

            if info.get("rank", 0) == 0 and assembly.modules:
                ranks = [m.rank for m in assembly.modules.values() if m.rank > 0]
                if ranks:
                    info["rank"] = max(set(ranks), key=ranks.count)
                else:
                    info["rank"] = 0

            self._models[final_name] = VirtualModel(
                name=final_name,
                assembly=assembly,
                info=info
            )
            Logger.info(f"Workspace: Added '{final_name}' (Assembly - Arch: {info['arch']})")
            return final_name

    def load_from_disk(self, path: str, alias: str = None) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")

        name = alias or os.path.splitext(os.path.basename(path))[0]
        io = SafeStreamer(path, device="cpu")
        tensors = io.load_state_dict()
        
        if not tensors:
            raise ValueError(f"File '{name}' is empty.")
            
        # Ранг и архитектура будут автоматически определены в add_model
        info = {"source_path": path, "rank": 0}
        return self.add_model(name, tensors, io.metadata, info)

    def save_to_disk(self, name: str, path: str):
        with self._lock:
            model = self._models.get(name)
            if not model:
                raise ValueError(f"Model '{name}' not found in workspace.")
            
            try:
                tensors = model.assembly.to_state_dict()
                SafeStreamer.save_tensors(tensors, path, model.assembly.metadata)
                Logger.info(f"Workspace: Saved '{name}' to '{path}'")
                del tensors
                gc.collect()
            except Exception as e:
                Logger.error(f"Workspace Save Error: {e}")
                raise e

    def delete_model(self, name: str):
        with self._lock:
            if name in self._models:
                self._models[name].assembly.modules.clear()
                self._models[name].assembly.others.clear()
                del self._models[name]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def clear_all(self):
        with self._lock:
            for m in self._models.values():
                m.assembly.modules.clear()
                m.assembly.others.clear()
            self._models.clear()
            gc.collect()

    def get_total_memory_usage(self) -> int:
        with self._lock:
            return sum(m.size_bytes for m in self._models.values())