# core/pipeline/base.py

import torch
import os
import gc
from typing import Union, Dict, Any
from ..io_manager import SafeStreamer
from ..format_handler import FormatHandler
from ..workspace import WorkspaceManager

class PipelineBase:
    """
    Base class containing initialization and shared utilities.
    """
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.workspace = WorkspaceManager()
        print(f"Pipeline initialized on {self.device}")

    def _resolve_source(self, source: str) -> Union[str, Dict[str, torch.Tensor]]:
        """
        Resolves a string input to either a file path or a tensor dictionary (from Workspace).
        """
        if self.workspace.exists(source):
            return self.workspace.get_model(source).tensors
        
        if os.path.exists(source):
            return source
            
        raise ValueError(f"Source '{source}' not found in Workspace or Disk.")

    def _resolve_metadata(self, source: str) -> Dict[str, str]:
        if self.workspace.exists(source):
            return self.workspace.get_model(source).metadata
        return {}

    def get_lora_info(self, path: str) -> int:
        try:
            source = self._resolve_source(path)
            with SafeStreamer(source, device="cpu") as s:
                groups = FormatHandler.group_keys(s.keys[:100]) # Scan first 100 keys is enough
                if groups:
                    tensor = s.get_tensor(groups[0].down_key)
                    if tensor is not None:
                        return tensor.shape[0]
        except Exception:
            pass
        return -1

    def garbage_collect(self):
        """Forces memory cleanup for both RAM and VRAM."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()