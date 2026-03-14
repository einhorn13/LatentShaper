
import folder_paths
from .core.comfy_utils import process_merge_dict, load_lora_cached
from .core.structs_assembly import LoRAAssembly
from .core.model_specs import ModelRegistry

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
            inputs["optional"][f"ls_lora_{i}"] = ("LS_LORA",)
            inputs["optional"][f"lora_name_{i}"] = (folder_paths.get_filename_list("loras"),)
            inputs["optional"][f"weight_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            
        return inputs

    RETURN_TYPES = ("LS_LORA",)
    RETURN_NAMES = ("ls_lora",)
    FUNCTION = "merge"
    CATEGORY = "Latent Shaper/Merging"

    def merge(self, algorithm, target_rank, global_strength, **kwargs):
        active_loras = []
        
        for i in range(1, 7):
            ls_lora = kwargs.get(f"ls_lora_{i}")
            name = kwargs.get(f"lora_name_{i}")
            weight = kwargs.get(f"weight_{i}", 1.0)
            
            assembly = None
            source_name = "Unknown"
            
            if ls_lora is not None:
                if "assembly" in ls_lora:
                    assembly = ls_lora["assembly"]
                elif "sd" in ls_lora:
                    assembly = LoRAAssembly.from_state_dict(ls_lora["sd"], ls_lora.get("metadata"))
                
                source_name = ls_lora.get("name", f"Input_{i}")
                
            elif name and name != "None":
                path = folder_paths.get_full_path("loras", name)
                assembly = load_lora_cached(path)
                source_name = name
            
            if assembly is not None and weight != 0:
                active_loras.append({
                    "assembly": assembly,
                    "ratio": weight,
                    "path": source_name
                })
        
        if not active_loras:
            print("[Latent Shaper] Warning: No valid inputs provided for merge.")
            return ({"assembly": LoRAAssembly(), "name": "Empty_Merge"},)

        # Architecture consistency check
        architectures = set()
        for item in active_loras:
            spec = ModelRegistry.get_spec(list(item["assembly"].modules.keys()))
            architectures.add(spec.name)
            
        if len(architectures) > 1:
            print(f"\n[Latent Shaper] ⚠️ WARNING: Merging LoRAs with DIFFERENT architectures: {architectures}")
            print("[Latent Shaper] This usually results in a broken or disjointed model.\n")

        merged_sd = process_merge_dict(active_loras, algorithm, target_rank, global_strength)
        merged_assembly = LoRAAssembly.from_state_dict(merged_sd)
        
        return ({"assembly": merged_assembly, "name": "Merged_Result"},)