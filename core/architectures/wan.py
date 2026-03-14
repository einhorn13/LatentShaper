
import re
import torch
from typing import List, Tuple
from core.architectures.base import BaseArchitecture

class WanArchitecture(BaseArchitecture):
    """
    Specification for Wan Architecture.
    Covers Wan 1.0, Wan 2.1, and Anima (MiniTrainDIT).
    Supports Checkpoints and PEFTs (LoRA, DoRA, LoHa, LoKr).
    """
    
    @property
    def name(self) -> str: return "WAN 2.1 / Anima"
        
    @property
    def block_count(self) -> int: return 40

    @property
    def lora_prefixes(self) -> List[str]: 
        return["lora_unet_", "lora_te_", "lora_model_"]

    @property
    def model_prefixes(self) -> List[str]: 
        return["model.diffusion_model.", "transformer.", "model.", "diffusion_model.", "llm_adapter."]
    
    _BLOCK_PATTERN = re.compile(r"(?:transformer[\._]|diffusion_model[\._])?blocks[\._](\d+)")
    
    _COMPONENT_MAP = {
        "self_attn.q_proj": 0, "self_attn.to_q": 0,
        "self_attn.k_proj": 1, "self_attn.to_k": 1,
        "self_attn.v_proj": 2, "self_attn.to_v": 2,
        "self_attn.o_proj": 3, "self_attn.to_out": 3,
        "cross_attn.q_proj": 4, "cross_attn.to_q": 4,
        "cross_attn.k_proj": 5, "cross_attn.to_k": 5,
        "cross_attn.v_proj": 6, "cross_attn.to_v": 6,
        "cross_attn.o_proj": 7, "cross_attn.to_out": 7,
        "mlp.0": 8, "mlp.2": 9, "mlp.layer1": 8, "mlp.layer2": 9,
        "llm_adapter": 10
    }

    def detect(self, keys: List[str]) -> bool:
        if not keys: return False
        # Join all keys to ensure we don't miss DiT blocks hidden behind huge Text Encoders
        k_str = "\n".join(keys).lower()
        
        # 1. Anti-Patterns
        if "double_blocks" in k_str or "single_blocks" in k_str: return False # Flux
        if "input_blocks" in k_str: return False # SD
        if "joint_blocks" in k_str: return False # SD3
        
        # 2. Signatures
        is_base = "llm_adapter" in k_str or "x_embedder.proj.1" in k_str or "patch_embedding" in k_str
        has_blocks = "blocks." in k_str or "layers." in k_str
        has_dual_attn = "self_attn" in k_str and "cross_attn" in k_str
        
        return is_base or (has_blocks and has_dual_attn)

    def diagnose(self, keys: List[str]) -> List[str]:
        if not keys: return ["File contains no keys."]
        k_str = "\n".join(keys).lower()
        
        if "double_blocks" in k_str: return ["Rejected: Found Flux signatures."]
        if "input_blocks" in k_str: return ["Rejected: Found SD signatures."]
        if "joint_blocks" in k_str: return ["Rejected: Found SD3 signatures."]
        
        is_base = "llm_adapter" in k_str or "x_embedder.proj.1" in k_str or "patch_embedding" in k_str
        has_blocks = "blocks." in k_str or "layers." in k_str
        has_dual_attn = "self_attn" in k_str and "cross_attn" in k_str
        
        if not is_base and not (has_blocks and has_dual_attn):
            return["Rejected: Missing Wan/Anima Dual-Stream signatures (self_attn + cross_attn)."]
            
        return ["Unknown mismatch."]

    def get_block_number(self, key: str) -> int:
        match = self._BLOCK_PATTERN.search(key)
        # Anima specific adapter offset
        if "llm_adapter" in key and match:
            return 28 + int(match.group(1))
        return int(match.group(1)) if match else -1

    def get_component_idx(self, key: str) -> int:
        for k, v in self._COMPONENT_MAP.items():
            if k in key: return v
        return -1

    def get_region(self, block_idx: int) -> str:
        if block_idx == -1: return "OTHER"
        if block_idx < 10: return "IN"          
        if block_idx < 20: return "MID"        
        if block_idx < 40: return "OUT"        
        return "OTHER"

    def is_lora_target(self, key: str) -> bool: 
        return any(k in key for k in self._COMPONENT_MAP.keys())

    def get_heatmap_dimensions(self) -> Tuple[int, int]: 
        return (40, 11)

    def get_regions(self) -> List[str]: 
        return ["IN", "MID", "OUT", "OTHER"]