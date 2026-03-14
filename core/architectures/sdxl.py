
import re
import torch
from typing import List, Tuple
from core.architectures.base import BaseArchitecture

class SDXLArchitecture(BaseArchitecture):
    """
    Specification for SDXL and SD 1.5 Architecture.
    """
    
    @property
    def name(self) -> str: return "SDXL / SD 1.5"
    
    @property
    def block_count(self) -> int: return 20
    
    @property
    def lora_prefixes(self) -> List[str]: 
        return["lora_unet_", "lora_te_", "lora_te1_", "lora_te2_"]
        
    @property
    def model_prefixes(self) -> List[str]: 
        return ["model.diffusion_model.", "model."]
    
    def detect(self, keys: List[str]) -> bool:
        if not keys: return False
        k_str = "\n".join(keys).lower()
        
        if "double_blocks" in k_str: return False
        
        return "input_blocks" in k_str or "output_blocks" in k_str or "middle_block" in k_str

    def diagnose(self, keys: List[str]) -> List[str]:
        if not keys: return ["File contains no keys."]
        k_str = "\n".join(keys).lower()
        
        if "double_blocks" in k_str: return ["Rejected: Found Flux signatures."]
        if "input_blocks" not in k_str and "output_blocks" not in k_str:
            return["Rejected: Missing SD signatures (input_blocks / output_blocks)."]
            
        return ["Unknown mismatch."]

    def get_block_number(self, key: str) -> int:
        if "input_blocks" in key:
            match = re.search(r"input_blocks[\._]?(\d+)", key)
            return int(match.group(1)) if match else -1
        elif "middle_block" in key:
            return 9
        elif "output_blocks" in key:
            match = re.search(r"output_blocks[\._]?(\d+)", key)
            return 10 + int(match.group(1)) if match else -1
        return -1

    def get_component_idx(self, key: str) -> int:
        if "to_q" in key: return 0
        if "to_k" in key: return 1
        if "to_v" in key: return 2
        if "to_out" in key: return 3
        if "ff.net.0" in key: return 4
        if "ff.net.2" in key: return 5
        return -1

    def get_region(self, block_idx: int) -> str:
        if block_idx == -1: return "OTHER"
        if block_idx < 9: return "IN"
        if block_idx == 9: return "MID"
        return "OUT"

    def is_lora_target(self, key: str) -> bool: 
        return ".weight" in key

    def get_heatmap_dimensions(self) -> Tuple[int, int]: 
        return (20, 6)

    def get_regions(self) -> List[str]: 
        return ["IN", "MID", "OUT", "OTHER"]