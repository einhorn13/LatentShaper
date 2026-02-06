# core/services/base.py

import torch
import gc
from ..workspace import WorkspaceManager
from ..loader import ModelLoader
from ..logger import Logger

class BaseService:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.workspace = WorkspaceManager()

    def garbage_collect(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_source(self, path):
        return ModelLoader.load(path, self.device)