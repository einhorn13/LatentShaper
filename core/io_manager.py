# core/io_manager.py

import torch
import os
import time
import gc
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, List, Optional, Union, Any

class SafeStreamer:
    """
    Unified I/O handler with robust retry logic for OS file locks.
    Optimized for streaming large checkpoints and LoRA files.
    Implements strict storage management to prevent file size bloat.
    """
    
    # Cache for directory scanning: {path: (timestamp, [files])}
    _SCAN_CACHE: Dict[str, Any] = {}
    _CACHE_TTL: float = 10.0  # Seconds

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
        """Initializes file access with retry logic for locked files."""
        max_retries = 10
        for attempt in range(max_retries):
            try:
                if not os.path.exists(self._path):
                    self.load_error = f"File not found: {self._path}"
                    return
                
                with safe_open(self._path, framework="pt", device="cpu") as f:
                    self._keys = list(f.keys())
                    self._metadata = f.metadata() or {}
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
            try:
                self._handle = safe_open(self._path, framework="pt", device="cpu")
            except Exception as e:
                self.load_error = str(e)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle:
            self._handle = None

    @property
    def keys(self) -> List[str]:
        return self._keys
    
    @property
    def metadata(self) -> Dict[str, str]:
        return self._metadata

    def get_tensor(self, key: str) -> Optional[torch.Tensor]:
        """Retrieves a single tensor from the source."""
        if self.load_error: 
            return None
        t = None
        if self._source_type == "memory":
            t = self._memory_data.get(key)
        elif self._handle:
            try:
                t = self._handle.get_tensor(key)
            except Exception: 
                return None
        
        if t is not None:
            return t.to(self.target_device)
        return None

    def load_state_dict(self) -> Dict[str, torch.Tensor]:
        """Loads all tensors into a dictionary."""
        if self._source_type == "memory": 
            return self._memory_data.copy()
        if self.load_error: 
            return {}
        
        tensors = {}
        try:
            with safe_open(self._path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    if self.target_device != "cpu":
                        t = t.to(self.target_device)
                    tensors[key] = t
            return tensors
        except Exception as e:
            print(f"[SafeStreamer] Load failed: {e}")
            return {}

    @staticmethod
    def save_tensors(tensors: Dict[str, torch.Tensor], path: str, metadata: Dict[str, str] = None) -> None:
        """
        Saves a dictionary of tensors to a safetensors file.
        Strictly ensures no shared storage or oversized buffers to prevent file size bloat.
        """
        if not tensors:
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Final safety pass: ensure every tensor has its own dedicated storage
        # and is exactly the size it reports to be. This prevents 24GB vs 12GB issues.
        clean_dict = {}
        for k, v in tensors.items():
            if not isinstance(v, torch.Tensor):
                continue
            
            # If physical storage is larger than logical size (common after slicing or casting),
            # clone() creates a new storage buffer that matches the logical size exactly.
            logical_size = v.nelement() * v.element_size()
            try:
                physical_size = v.untyped_storage().size()
            except (AttributeError, RuntimeError):
                physical_size = logical_size

            if physical_size > logical_size:
                clean_dict[k] = v.detach().clone().contiguous()
            else:
                clean_dict[k] = v.detach().contiguous()
        
        try:
            save_file(clean_dict, path, metadata=metadata)
        except Exception as e:
            print(f"[SafeStreamer] Save failed: {e}")
            raise e
        finally:
            # Aggressive memory cleanup
            clean_dict.clear()
            tensors.clear()
            del clean_dict
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def scan_directory(directory: str, extensions: List[str] = None, force_refresh: bool = False) -> List[str]:
        """
        Scans a directory recursively for model files with caching.
        """
        if extensions is None:
            extensions = [".safetensors", ".ckpt"]
            
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
                        found_files.append(os.path.abspath(os.path.join(root, file)))
        except Exception as e:
            print(f"[SafeStreamer] Scan error in {directory}: {e}")
            
        found_files.sort()
        SafeStreamer._SCAN_CACHE[directory] = (now, found_files)
        return found_files