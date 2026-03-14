
import re
import torch
from typing import List, Tuple
from core.architectures.base import BaseArchitecture

class FluxArchitecture(BaseArchitecture):
    """
    Specification for FLUX Architecture.
    Covers FLUX 1.0, FLUX 2.0, and FLUX 2 Klein (Base & Distilled).
    """
    
    @property
    def name(self) -> str: return "FLUX (1.0 / 2.0 / Klein)"
    
    @property
    def block_count(self) -> int: return 57
    
    @property
    def lora_prefixes(self) -> List[str]: 
        return ["lora_unet_", "lora_te_", "lora_model_"]
        
    @property
    def model_prefixes(self) -> List[str]: 
        return["model.diffusion_model.", "transformer.", "model."]
    
    _BLOCK_PATTERN = re.compile(r"(?:double_blocks|single_blocks)[\._]?(\d+)")
    
    _COMPONENT_MAP = {
        "img_attn.qkv": 0, "txt_attn.qkv": 1, "qkv": 2,
        "img_attn.proj": 3, "txt_attn.proj": 4,
        "img_mlp.0": 5, "txt_mlp.0": 6, "linear1": 7,
        "img_mlp.2": 8, "txt_mlp.2": 9, "linear2": 10
    }

    def detect(self, keys: List[str]) -> bool:
        if not keys: return False
        k_str = "\n".join(keys[:500]).lower()
        
        # УЖЕСТОЧЕНИЕ: Требуем не только блоки, но и специфичные FLUX-проекции
        has_blocks = "double_blocks" in k_str or "single_blocks" in k_str
        has_flux_attn = any(x in k_str for x in["img_attn", "txt_attn", "linear1", "linear2"])
        
        return has_blocks and has_flux_attn

    def diagnose(self, keys: List[str]) -> List[str]:
        if not keys: return ["File contains no keys."]
        k_str = "\n".join(keys[:500]).lower()
        
        has_blocks = "double_blocks" in k_str or "single_blocks" in k_str
        has_flux_attn = any(x in k_str for x in ["img_attn", "txt_attn", "linear1", "linear2"])
        
        diag =[]
        if not has_blocks: diag.append("Missing FLUX blocks (double_blocks / single_blocks).")
        if not has_flux_attn: diag.append("Missing FLUX specific attention projections (img_attn / txt_attn).")
        
        return diag if diag else ["Unknown mismatch."]

    def get_block_number(self, key: str) -> int:
        match = self._BLOCK_PATTERN.search(key)
        if match:
            idx = int(match.group(1))
            if "single_blocks" in key:
                return idx + 19
            return idx
        return -1

    def get_component_idx(self, key: str) -> int:
        for k, v in self._COMPONENT_MAP.items():
            if k in key: return v
        return -1

    def get_region(self, block_idx: int) -> str:
        if block_idx == -1: return "OTHER"
        if block_idx < 19: return "DOUBLE"
        return "SINGLE"

    def is_lora_target(self, key: str) -> bool: 
        return any(k in key for k in self._COMPONENT_MAP.keys())

    def get_heatmap_dimensions(self) -> Tuple[int, int]: 
        return (57, 11)

    def get_regions(self) -> List[str]: 
        return ["DOUBLE", "SINGLE", "OTHER"]