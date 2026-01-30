# nodes_merging.py

import folder_paths
from .core.comfy_utils import process_merge_dict, load_lora_cached

class LS_Merger:
    """
    Universal Merger Node.
    """
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "algorithm": (["SVD (Sum)", "Median (Robust)", "SLERP (Chain)", "TIES (Density)", "Orthogonal (Accum)"],),
                # 0 = Auto (Max Input Rank)
                "target_rank": ("INT", {"default": 0, "min": 0, "max": 256, "step": 4, "label": "Target Rank (0=Auto)"}),
                "global_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
            },
            "optional": {}
        }
        
        for i in range(1, 7):
            inputs["optional"][f"z_lora_{i}"] = ("Z_LORA",)
            inputs["optional"][f"lora_name_{i}"] = (folder_paths.get_filename_list("loras"),)
            inputs["optional"][f"weight_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            
        return inputs

    RETURN_TYPES = ("Z_LORA",)
    RETURN_NAMES = ("z_lora",)
    FUNCTION = "merge"
    CATEGORY = "LoRA Studio/Merging"

    def merge(self, algorithm, target_rank, global_strength, **kwargs):
        active_loras = []
        
        for i in range(1, 7):
            z_lora = kwargs.get(f"z_lora_{i}")
            name = kwargs.get(f"lora_name_{i}")
            weight = kwargs.get(f"weight_{i}", 1.0)
            
            sd = None
            source_name = "Unknown"
            
            if z_lora is not None:
                sd = z_lora.get("sd")
                source_name = z_lora.get("name", f"Input_{i}")
            elif name and name != "None":
                path = folder_paths.get_full_path("loras", name)
                sd, _ = load_lora_cached(path)
                source_name = name
            
            if sd is not None and weight != 0:
                active_loras.append({
                    "sd": sd,
                    "ratio": weight,
                    "path": source_name
                })
        
        if not active_loras:
            print("[LoRA Studio] Warning: No valid inputs provided for merge.")
            return ({"sd": {}, "name": "Empty_Merge"},)

        merged_sd = process_merge_dict(active_loras, algorithm, target_rank, global_strength)
        return ({"sd": merged_sd, "name": "Merged_Result"},)