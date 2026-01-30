# core/io_manager.py

import torch
import os
import time
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, List, Optional, Union

class SafeStreamer:
    """
    Unified I/O handler with robust retry logic for OS file locks.
    Forces CPU loading for file safety, moves to GPU only on demand.
    """
    
    # Cache for directory scanning: {path: (timestamp, [files])}
    _SCAN_CACHE = {}
    _CACHE_TTL = 10.0 # Seconds

    def __init__(self, source: Union[str, Dict[str, torch.Tensor]], device: str = "cpu", metadata: Dict[str, str] = None):
        self.target_device = device  # Device where tensors will be moved to
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

    def _init_file_mode(self):
        max_retries = 10
        for attempt in range(max_retries):
            try:
                # Always open on CPU to avoid VRAM fragmentation and locking issues
                with safe_open(self._path, framework="pt", device="cpu") as f:
                    self._keys = list(f.keys())
                    self._metadata = f.metadata()
                self.load_error = None
                return
            except Exception as e:
                err_msg = str(e).lower()
                is_transient = any(x in err_msg for x in ["process cannot", "access is denied", "header", "locking"])
                if is_transient and attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                self.load_error = str(e)
                break

    def __enter__(self):
        if self._source_type == "file" and not self.load_error:
            # Open handle on CPU
            self._handle = safe_open(self._path, framework="pt", device="cpu")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._handle = None

    @property
    def keys(self) -> List[str]:
        return self._keys
    
    @property
    def metadata(self) -> Dict[str, str]:
        return self._metadata

    def get_tensor(self, key: str) -> Optional[torch.Tensor]:
        if self.load_error: return None
        t = None
        if self._source_type == "memory":
            t = self._memory_data.get(key)
        elif self._handle:
            try:
                t = self._handle.get_tensor(key)
            except: 
                return None
        
        # Move to target device on demand
        if t is not None:
            return t.to(self.target_device)
        return None

    def load_state_dict(self) -> Dict[str, torch.Tensor]:
        if self._source_type == "memory": return self._memory_data.copy()
        if self.load_error: return {}
        
        if self._handle:
            try:
                tensors = {}
                for key in self._handle.keys():
                    t = self._handle.get_tensor(key)
                    if self.target_device != "cpu":
                        t = t.to(self.target_device)
                    tensors[key] = t
                return tensors
            except Exception:
                pass

        for attempt in range(5):
            try:
                tensors = {}
                with safe_open(self._path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        t = f.get_tensor(key)
                        if self.target_device != "cpu":
                            t = t.to(self.target_device)
                        tensors[key] = t
                return tensors
            except: time.sleep(0.5)
        return {}

    @staticmethod
    def save_tensors(tensors: Dict[str, torch.Tensor], path: str, metadata: Dict[str, str] = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cpu_tensors = {k: v.to("cpu").contiguous() for k, v in tensors.items()}
        save_file(cpu_tensors, path, metadata)

    @staticmethod
    def scan_directory(directory: str, extensions: List[str] = [".safetensors", ".ckpt"], force_refresh: bool = False) -> List[str]:
        """
        Scans a directory recursively with caching.
        """
        if not directory or not os.path.exists(directory):
            return []
            
        now = time.time()
        if not force_refresh and directory in SafeStreamer._SCAN_CACHE:
            ts, files = SafeStreamer._SCAN_CACHE[directory]
            if now - ts < SafeStreamer._CACHE_TTL:
                return files

        found_files = []
        try:
            for root, _, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        found_files.append(os.path.join(root, file))
        except Exception as e:
            print(f"Error scanning directory {directory}: {e}")
            
        found_files.sort()
        SafeStreamer._SCAN_CACHE[directory] = (now, found_files)
        return found_files