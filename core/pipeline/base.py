
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
        # Check Workspace first
        if hasattr(source, 'path'): # Handle ModelReference object
            path = source.path
        else:
            path = source

        if self.workspace.exists(path):
            # Return SafeStreamer directly initialized with dict
            # This matches the expectation of services using `with self._resolve_source(ref) as io:`
            # But wait, services expect _resolve_source to return something that can be passed to SafeStreamer constructor OR be a SafeStreamer itself?
            # Looking at services code: `with self._resolve_source(ref) as io:`
            # BaseService._resolve_source calls ModelLoader.load which returns a SafeStreamer instance.
            # So PipelineBase._resolve_source is actually redundant/conflicting with BaseService._resolve_source?
            
            # Let's look at BaseService in core/services/base.py
            # It imports ModelLoader. So services use ModelLoader.
            
            # However, PipelineBase is used by OperationsMixin (legacy pipeline).
            # Let's fix it to return SafeStreamer compatible input.
            
            model = self.workspace.get_model(path)
            return model.assembly.to_state_dict()
        
        if os.path.exists(path):
            return path
            
        raise ValueError(f"Source '{path}' not found in Workspace or Disk.")

    def _resolve_metadata(self, source: str) -> Dict[str, str]:
        path = source.path if hasattr(source, 'path') else source
        if self.workspace.exists(path):
            return self.workspace.get_model(path).assembly.metadata
        return {}

    def get_lora_info(self, path: str) -> int:
        try:
            # Use ModelLoader to handle both cases transparently
            from ..loader import ModelLoader
            with ModelLoader.load(path, device="cpu") as s:
                groups = FormatHandler.group_keys(s.keys[:100])
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