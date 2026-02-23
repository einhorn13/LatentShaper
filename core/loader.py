
import os
import torch
from typing import Dict, Union, Any
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
        ws = WorkspaceManager()

        # Handle string input
        if isinstance(ref, str):
            source_type = ModelSourceType.WORKSPACE if ws.exists(ref) else ModelSourceType.DISK
            ref = ModelReference(ref, source_type)
        
        # Defensive check: if ModelReference is marked as DISK but file doesn't exist,
        # check if it's actually a model in the Workspace (RAM).
        if ref.source_type == ModelSourceType.DISK and not os.path.exists(ref.path):
            if ws.exists(ref.path):
                ref.source_type = ModelSourceType.WORKSPACE

        if ref.source_type == ModelSourceType.WORKSPACE:
            model = ws.get_model(ref.path)
            if not model:
                raise ValueError(f"Model '{ref.path}' not found in Workspace.")
            
            # Convert Assembly to Dict for SafeStreamer compatibility
            # This creates a temporary copy of tensors, which is acceptable for streaming
            tensors = model.assembly.to_state_dict()
            return SafeStreamer(tensors, device=device, metadata=model.assembly.metadata)
        
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