
import re
import torch
from typing import List, Tuple
from core.architectures.base import BaseArchitecture

class S3DiTArchitecture(BaseArchitecture):
    """
    Specification for S3-DiT Architecture (Z-Image Base, Z-Image Turbo).
    Strictly Single-Stream.
    """
    
    @property
    def name(self) -> str: return "S3-DiT (Z-Image)"
    
    @property
    def block_count(self) -> int: return 30
    
    @property
    def lora_prefixes(self) -> List[str]: 
        return ["lora_unet_", "lora_te_", "lora_model_"]
        
    @property
    def model_prefixes(self) -> List[str]: 
        return["model.diffusion_model.", "transformer.", "model."]
    
    _BLOCK_PATTERN = re.compile(r"(?:blocks|layers|input_blocks|output_blocks|middle_block|context_refiner|noise_refiner)[\._]?(\d+)")
    
    _COMPONENT_MAP = {
        "attn.q_proj": 0, "to_q": 0, "q_proj": 0, "attn1.to_q": 0, "attn2.to_q": 0,
        "attn.k_proj": 1, "to_k": 1, "k_proj": 1, "attn1.to_k": 1, "attn2.to_k": 1,
        "attn.v_proj": 2, "to_v": 2, "v_proj": 2, "attn1.to_v": 2, "attn2.to_v": 2,
        "attn.o_proj": 3, "to_out": 3, "out_proj": 3, "attn1.to_out": 3, "attn2.to_out": 3, "attention.out": 3,
        "attn.qkv": 0, "qkv_proj": 0, "qkv": 0, "attention.qkv": 0,
        "mlp.gate_proj": 4, "gate_proj": 4, "mlp.0": 4, "ff.net.0": 4,
        "mlp.up_proj": 5, "up_proj": 5, "ff.net.2": 5,
        "mlp.down_proj": 6, "down_proj": 6, "mlp.2": 6, "linear": 6,
        "adaln_modulation": 7, "adaln": 7
    }

    def detect(self, keys: List[str]) -> bool:
        if not keys: return False
        k_str = "\n".join(keys[:500]).lower()
        
        # 1. Anti-Patterns
        if "double_blocks" in k_str or "single_blocks" in k_str: return False # Flux
        if "joint_blocks" in k_str: return False # SD3
        if "input_blocks" in k_str: return False # SD1.5 / SDXL
        if "self_attn" in k_str or "cross_attn" in k_str: return False # Wan / Anima
        
        # 2. Signatures
        is_base = any(x in k_str for x in["context_refiner", "noise_refiner", "cap_embedder", "x_embedder"])
        has_blocks = "blocks." in k_str or "layers." in k_str
        has_projs = any(x in k_str for x in[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", 
            "qkv", "to_q", "to_k", "to_v", "to_out", "mlp", "ff_net", "linear"
        ])
        
        return is_base or (has_blocks and has_projs)

    def diagnose(self, keys: List[str]) -> List[str]:
        if not keys: return ["File contains no keys."]
        k_str = "\n".join(keys[:500]).lower()
        
        if "double_blocks" in k_str: return["Rejected: Found Flux signatures (double_blocks)."]
        if "joint_blocks" in k_str: return["Rejected: Found SD3 signatures (joint_blocks)."]
        if "input_blocks" in k_str: return ["Rejected: Found SD1.5/SDXL signatures (input_blocks)."]
        if "self_attn" in k_str or "cross_attn" in k_str: return["Rejected: Found Dual-Stream signatures (self_attn/cross_attn). S3-DiT is Single-Stream."]
        
        is_base = any(x in k_str for x in["context_refiner", "noise_refiner", "cap_embedder", "x_embedder"])
        has_blocks = "blocks." in k_str or "layers." in k_str
        has_projs = any(x in k_str for x in[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", 
            "qkv", "to_q", "to_k", "to_v", "to_out", "mlp", "ff_net", "linear"
        ])
        
        if not is_base and not (has_blocks and has_projs):
            return ["Rejected: Missing S3-DiT structural blocks or linear projections."]
            
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
        if block_idx < 10: return "IN"
        if block_idx < 20: return "MID"
        if block_idx < 30: return "OUT"
        return "OTHER"

    def is_lora_target(self, key: str) -> bool: 
        return any(k in key for k in self._COMPONENT_MAP.keys())

    def get_heatmap_dimensions(self) -> Tuple[int, int]: 
        return (30, 8)

    def get_regions(self) -> List[str]: 
        return["IN", "MID", "OUT", "OTHER"]