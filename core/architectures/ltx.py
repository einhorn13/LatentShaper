
import re
import torch
from typing import List, Tuple
from core.architectures.base import BaseArchitecture

class LTXArchitecture(BaseArchitecture):
    """
    Specification for LTX-Video Architecture.
    Covers LTX-Video, LTX 2, and LTX 2.3.
    """
    
    @property
    def name(self) -> str: return "LTX-Video (LTX 2 / 2.3)"
    
    @property
    def block_count(self) -> int: return 48
    
    @property
    def lora_prefixes(self) -> List[str]: 
        return["lora_unet_", "lora_te_", "lora_model_"]
        
    @property
    def model_prefixes(self) -> List[str]: 
        return["model.diffusion_model.", "transformer.", "model.", "audio_connector."]
    
    _BLOCK_PATTERN = re.compile(r"transformer_blocks[\._]?(\d+)")
    
    _COMPONENT_MAP = {
        "attn1.to_q": 0, "attn1.to_k": 1, "attn1.to_v": 2, "attn1.to_out": 3,
        "attn2.to_q": 4, "attn2.to_k": 5, "attn2.to_v": 6, "attn2.to_out": 7,
        "ff.net.0": 8, "ff.net.2": 9
    }

    def detect(self, keys: List[str]) -> bool:
        if not keys: return False
        k_str = "\n".join(keys).lower()
        
        if "input_blocks" in k_str: return False
        if "double_blocks" in k_str: return False
        
        has_blocks = "transformer_blocks" in k_str
        has_attn = any(x in k_str for x in["attn1.to_q", "attn2.to_v", "caption_proj_before_connector"])
        
        return has_blocks and has_attn

    def diagnose(self, keys: List[str]) -> List[str]:
        if not keys: return ["File contains no keys."]
        k_str = "\n".join(keys).lower()
        
        if "input_blocks" in k_str: return ["Rejected: Found SD signatures."]
        if "double_blocks" in k_str: return ["Rejected: Found Flux signatures."]
        
        if "transformer_blocks" not in k_str:
            return ["Rejected: Missing LTX signatures (transformer_blocks)."]
            
        return ["Unknown mismatch."]

    def get_block_number(self, key: str) -> int:
        match = self._BLOCK_PATTERN.search(key)
        return int(match.group(1)) if match else -1

    def get_component_idx(self, key: str) -> int:
        for k, v in self._COMPONENT_MAP.items():
            if k in key: return v
        return -1

    def get_region(self, block_idx: int) -> str:
        if block_idx == -1: return "OTHER"
        if block_idx < 14: return "IN"
        if block_idx < 28: return "MID"
        return "OUT"

    def is_lora_target(self, key: str) -> bool: 
        return any(k in key for k in self._COMPONENT_MAP.keys())

    def get_heatmap_dimensions(self) -> Tuple[int, int]: 
        return (48, 10)

    def get_regions(self) -> List[str]: 
        return ["IN", "MID", "OUT", "OTHER"]