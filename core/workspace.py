# core/workspace.py

import torch
import os
import gc
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from .io_manager import SafeStreamer
from .logger import Logger

@dataclass
class VirtualModel:
    name: str
    tensors: Dict[str, torch.Tensor]
    metadata: Dict[str, str]
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        total = 0
        for t in self.tensors.values():
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
                
            cpu_tensors = {k: v.to("cpu") for k, v in tensors.items()}
            self._models[final_name] = VirtualModel(
                name=final_name,
                tensors=cpu_tensors,
                metadata=metadata or {},
                info=info or {}
            )
            Logger.info(f"Workspace: Added '{final_name}'")
            return final_name

    def load_from_disk(self, path: str, alias: str = None) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")

        name = alias or os.path.splitext(os.path.basename(path))[0]
        io = SafeStreamer(path, device="cpu")
        tensors = io.load_state_dict()
        
        if not tensors:
            raise ValueError(f"File '{name}' is empty.")
            
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
                SafeStreamer.save_tensors(model.tensors, path, model.metadata)
                Logger.info(f"Workspace: Saved '{name}' to '{path}'")
            except Exception as e:
                Logger.error(f"Workspace Save Error: {e}")
                raise e

    def delete_model(self, name: str):
        with self._lock:
            if name in self._models:
                self._models[name].tensors.clear()
                del self._models[name]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def clear_all(self):
        with self._lock:
            for m in self._models.values():
                m.tensors.clear()
            self._models.clear()
            gc.collect()

    def get_total_memory_usage(self) -> int:
        with self._lock:
            return sum(m.size_bytes for m in self._models.values())