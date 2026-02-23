
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
        # Estimate size from modules
        for mod in self.assembly.modules.values():
            if mod.down is not None: total += mod.down.numel() * mod.down.element_size()
            if mod.up is not None: total += mod.up.numel() * mod.up.element_size()
            if mod.s is not None: total += mod.s.numel() * mod.s.element_size()
            # U and Vh are usually transient or replace Down/Up, but count if exist
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
        """
        Accepts either a raw state_dict (legacy) or creates an Assembly.
        """
        with self._lock:
            final_name = name
            counter = 1
            while final_name in self._models:
                final_name = f"{name}_{counter}"
                counter += 1
            
            # Convert to Assembly immediately
            assembly = LoRAAssembly.from_state_dict(tensors, metadata)
            
            self._models[final_name] = VirtualModel(
                name=final_name,
                assembly=assembly,
                info=info or {}
            )
            Logger.info(f"Workspace: Added '{final_name}'")
            return final_name

    def add_assembly(self, name: str, assembly: LoRAAssembly, info: Dict[str, Any] = None):
        """Directly adds an assembly object."""
        with self._lock:
            final_name = name
            counter = 1
            while final_name in self._models:
                final_name = f"{name}_{counter}"
                counter += 1
            
            self._models[final_name] = VirtualModel(
                name=final_name,
                assembly=assembly,
                info=info or {}
            )
            Logger.info(f"Workspace: Added '{final_name}' (Assembly)")
            return final_name

    def load_from_disk(self, path: str, alias: str = None) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")

        name = alias or os.path.splitext(os.path.basename(path))[0]
        io = SafeStreamer(path, device="cpu")
        tensors = io.load_state_dict()
        
        if not tensors:
            raise ValueError(f"File '{name}' is empty.")
            
        # Quick rank check
        rank = 0
        for k in tensors.keys():
            if "lora_down" in k:
                rank = tensors[k].shape[0]
                break
        
        info = {"source_path": path, "rank": rank}
        return self.add_model(name, tensors, io.metadata, info)

    def save_to_disk(self, name: str, path: str):
        """Saves a workspace model to disk."""
        with self._lock:
            model = self._models.get(name)
            if not model:
                raise ValueError(f"Model '{name}' not found in workspace.")
            
            try:
                # Convert Assembly back to Dict
                tensors = model.assembly.to_state_dict()
                SafeStreamer.save_tensors(tensors, path, model.assembly.metadata)
                Logger.info(f"Workspace: Saved '{name}' to '{path}'")
                
                # Cleanup temporary dict
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