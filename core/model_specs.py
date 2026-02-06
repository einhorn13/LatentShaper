# core/model_specs.py

import re
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List

class ModuleType(Enum):
    TRANSFORMER = auto()
    TEXT_ENCODER = auto()
    VAE = auto()
    OTHER = auto()

class ModelSpec(ABC):
    @property
    @abstractmethod
    def name(self) -> str: pass
    @property
    @abstractmethod
    def block_count(self) -> int: pass
    @abstractmethod
    def detect(self, keys: List[str]) -> bool: pass
    @abstractmethod
    def get_block_number(self, key: str) -> int: pass
    @abstractmethod
    def get_component_idx(self, key: str) -> int: pass
    @abstractmethod
    def get_region(self, block_idx: int) -> str: pass
    @abstractmethod
    def is_lora_target(self, key: str) -> bool: pass

class S3DiTSpec(ModelSpec):
    """
    Specification for S3-DiT Architecture (Z-Image Turbo, etc.)
    """
    name = "S3-DiT (Z-Image Turbo)"
    block_count = 30
    
    _BLOCK_PATTERN = re.compile(r"(?:blocks|layers|input_blocks|output_blocks|middle_block)[\._]?(\d+)")
    
    _COMPONENT_MAP = {
        "attn.q_proj": 0, "to_q": 0, "q_proj": 0,
        "attn.k_proj": 1, "to_k": 1, "k_proj": 1,
        "attn.v_proj": 2, "to_v": 2, "v_proj": 2,
        "attn.o_proj": 3, "to_out.0": 3, "out_proj": 3,
        "attn.qkv": 0, "qkv_proj": 0,
        "mlp.gate_proj": 4, "gate_proj": 4,
        "mlp.up_proj": 5, "up_proj": 5,
        "mlp.down_proj": 6, "down_proj": 6,
        "ff.net": 4, "linear": 4
    }

    def detect(self, keys: List[str]) -> bool:
        k_str = "".join(keys).lower()
        is_dit = "transformer" in k_str or "diffusion_model" in k_str
        is_not_unet = "input_blocks" not in k_str or "double_blocks" in k_str
        return is_dit and is_not_unet

    def get_block_number(self, key: str) -> int:
        match = self._BLOCK_PATTERN.search(key)
        if match:
            return int(match.group(1))
        return -1

    def get_component_idx(self, key: str) -> int:
        for k, v in self._COMPONENT_MAP.items():
            if k in key: return v
        return -1

    def get_region(self, block_idx: int) -> str:
        if block_idx == -1: return "OTHER"
        if block_idx < 10: return "IN"
        if block_idx < 20: return "MID"
        if block_idx < 30: return "OUT"
        return "OTHER"

    def is_lora_target(self, key: str) -> bool:
        if not key.endswith(".weight"): return False
        return any(k in key for k in self._COMPONENT_MAP.keys())

class UnknownSpec(ModelSpec):
    name = "Unknown Architecture"
    block_count = 0
    def detect(self, keys): return True
    def get_block_number(self, key): return -1
    def get_component_idx(self, key): return -1
    def get_region(self, block_idx): return "OTHER"
    def is_lora_target(self, key): return key.endswith(".weight")

class ModelRegistry:
    @staticmethod
    def get_spec(keys: List[str]) -> ModelSpec:
        # Future: Add FluxSpec, SDXLSpec here
        for spec_cls in [S3DiTSpec]:
            spec = spec_cls()
            if spec.detect(keys): return spec
        return UnknownSpec()