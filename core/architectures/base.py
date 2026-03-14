
from abc import ABC, abstractmethod
from typing import List, Tuple
import torch

class BaseArchitecture(ABC):
    @property
    @abstractmethod
    def name(self) -> str: pass
    @property
    @abstractmethod
    def block_count(self) -> int: pass
    @property
    @abstractmethod
    def lora_prefixes(self) -> List[str]: pass
    @property
    @abstractmethod
    def model_prefixes(self) -> List[str]: pass
    @abstractmethod
    def detect(self, keys: List[str]) -> bool: pass
    
    def diagnose(self, keys: List[str]) -> List[str]:
        return ["No diagnostic information available."]
        
    @abstractmethod
    def get_block_number(self, key: str) -> int: pass
    @abstractmethod
    def get_component_idx(self, key: str) -> int: pass
    @abstractmethod
    def get_region(self, block_idx: int) -> str: pass
    @abstractmethod
    def is_lora_target(self, key: str) -> bool: pass
    @abstractmethod
    def get_heatmap_dimensions(self) -> Tuple[int, int]: pass
    @abstractmethod
    def get_regions(self) -> List[str]: pass
    def preprocess_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor: return tensor

class UnknownArchitecture(BaseArchitecture):
    @property
    def name(self) -> str: return "Unknown Architecture"
    @property
    def block_count(self) -> int: return 0
    @property
    def lora_prefixes(self) -> List[str]: return []
    @property
    def model_prefixes(self) -> List[str]: return []
    def detect(self, keys: List[str]) -> bool: return False
    def get_block_number(self, key: str) -> int: return -1
    def get_component_idx(self, key: str) -> int: return -1
    def get_region(self, block_idx: int) -> str: return "OTHER"
    def is_lora_target(self, key: str) -> bool: return False
    def get_heatmap_dimensions(self) -> Tuple[int, int]: return (0, 0)
    def get_regions(self) -> List[str]: return ["OTHER"]