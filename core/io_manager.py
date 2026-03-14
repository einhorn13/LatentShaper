
import torch
import os
import time
import gc
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, List, Optional, Union, Any
from .logger import Logger

class SafeStreamer:
    """
    Unified I/O handler with robust resource management.
    """
    
    _SCAN_CACHE: Dict[str, Any] = {}
    _CACHE_TTL: float = 10.0

    def __init__(self, source: Union[str, Dict[str, torch.Tensor]], device: str = "cpu", metadata: Dict[str, str] = None):
        self.target_device = device
        self._keys: List[str] = []
        self._metadata: Dict[str, str] = metadata or {}
        self.load_error: Optional[str] = None
        
        self._source_type = "file" if isinstance(source, str) else "memory"
        self._path = source if self._source_type == "file" else None
        self._memory_data = source if self._source_type == "memory" else None
        self._handle = None 

        if self._source_type == "memory":
            self._keys = list(self._memory_data.keys())
        else:
            self._init_file_mode()

    def _init_file_mode(self) -> None:
        try:
            if not os.path.exists(self._path):
                self.load_error = f"File not found: {self._path}"
                return
            
            with safe_open(self._path, framework="pt", device="cpu") as f:
                self._keys = list(f.keys())
                self._metadata = f.metadata() or {}
        except Exception as e:
            self.load_error = str(e)

    def __enter__(self):
        if self._source_type == "file" and not self.load_error:
            try:
                self._handle = safe_open(self._path, framework="pt", device="cpu")
            except Exception as e:
                self.load_error = str(e)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle:
            try:
                if hasattr(self._handle, 'close'): self._handle.close()
            except: pass
            self._handle = None

    @property
    def keys(self) -> List[str]: return self._keys
    
    @property
    def metadata(self) -> Dict[str, str]: return self._metadata

    def get_tensor(self, key: str) -> Optional[torch.Tensor]:
        if self.load_error: return None
        t = None
        if self._source_type == "memory":
            t = self._memory_data.get(key)
        elif self._handle:
            try:
                t = self._handle.get_tensor(key)
            except: return None
        
        return t.to(self.target_device) if t is not None else None

    def load_state_dict(self) -> Dict[str, torch.Tensor]:
        if self._source_type == "memory": return self._memory_data.copy()
        if self.load_error: return {}
        
        tensors = {}
        try:
            with safe_open(self._path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key).to(self.target_device)
            return tensors
        except Exception as e:
            Logger.error(f"SafeStreamer: Load failed: {e}")
            return {}

    @staticmethod
    def save_tensors(tensors: Dict[str, torch.Tensor], path: str, metadata: Dict[str, str] = None) -> None:
        if not tensors: return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        clean_dict = {}
        for k, v in tensors.items():
            if not isinstance(v, torch.Tensor): continue
            # Clone to ensure memory is contiguous and stripped of large storage buffers
            clean_dict[k] = v.detach().clone().contiguous().to("cpu")
        
        try:
            save_file(clean_dict, path, metadata=metadata)
        except Exception as e:
            Logger.error(f"SafeStreamer: Save failed: {e}")
            raise e
        finally:
            clean_dict.clear()
            gc.collect()

    @staticmethod
    def scan_directory(directory: str, extensions: List[str] = None, force_refresh: bool = False) -> List[str]:
        if not directory or not os.path.exists(directory): return []
        now = time.time()
        if not force_refresh and directory in SafeStreamer._SCAN_CACHE:
            ts, files = SafeStreamer._SCAN_CACHE[directory]
            if now - ts < SafeStreamer._CACHE_TTL: return files

        found_files = []
        extensions = extensions or [".safetensors", ".ckpt"]
        try:
            for root, _, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        found_files.append(os.path.abspath(os.path.join(root, file)))
        except: pass
        found_files.sort()
        SafeStreamer._SCAN_CACHE[directory] = (now, found_files)
        return found_files