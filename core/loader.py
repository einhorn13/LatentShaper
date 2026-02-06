# core/loader.py

import os
import torch
from typing import Dict, Union
from .structs import ModelReference, ModelSourceType
from .io_manager import SafeStreamer
from .workspace import WorkspaceManager

class ModelLoader:
    """
    Factory that provides a unified interface (SafeStreamer) 
    for models coming from Disk or RAM.
    """
    
    @staticmethod
    def load(ref: Union[str, ModelReference], device: str = "cpu") -> SafeStreamer:
        # Backward compatibility for string paths
        if isinstance(ref, str):
            ws = WorkspaceManager()
            if ws.exists(ref):
                ref = ModelReference(ref, ModelSourceType.WORKSPACE)
            else:
                ref = ModelReference(ref, ModelSourceType.DISK)

        if ref.source_type == ModelSourceType.WORKSPACE:
            ws = WorkspaceManager()
            model = ws.get_model(ref.path)
            if not model:
                raise ValueError(f"Model '{ref.path}' not found in Workspace.")
            # SafeStreamer supports memory dict source
            return SafeStreamer(model.tensors, device=device, metadata=model.metadata)
        
        else:
            if not os.path.exists(ref.path):
                raise FileNotFoundError(f"File not found: {ref.path}")
            return SafeStreamer(ref.path, device=device)

    @staticmethod
    def resolve_metadata(ref: Union[str, ModelReference]) -> Dict[str, str]:
        try:
            with ModelLoader.load(ref, device="cpu") as io:
                return io.metadata
        except Exception:
            return {}