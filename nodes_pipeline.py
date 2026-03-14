
import torch
import folder_paths
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import json
from PIL import Image

from .core.math import MathKernel
from .core.comfy_utils import load_lora_cached, apply_lora_assembly, save_ls_lora
from .core.tensor_processor import TensorProcessor
from .core.model_specs import ModelRegistry
from .core.structs_assembly import LoRAAssembly

# --- Helper for Dual Format Support ---
def get_assembly(lora_input) -> LoRAAssembly:
    """
    Robustly retrieves LoRAAssembly from input.
    Supports:
    1. New LS_LORA format: {"assembly": LoRAAssembly, ...}
    2. Old Z_LORA format: {"sd": dict, ...}
    """
    if lora_input is None:
        return LoRAAssembly()
        
    # Case 1: New Format
    if "assembly" in lora_input:
        return lora_input["assembly"]
    
    # Case 2: Old/Standard Format (dict with state_dict)
    if "sd" in lora_input:
        # Convert on the fly
        return LoRAAssembly.from_state_dict(lora_input["sd"], lora_input.get("metadata"))
        
    # Case 3: Raw state dict
    if isinstance(lora_input, dict):
        return LoRAAssembly.from_state_dict(lora_input)
        
    return LoRAAssembly()

class LS_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"lora_name": (folder_paths.get_filename_list("loras"),)}}
    
    RETURN_TYPES = ("LS_LORA",)
    RETURN_NAMES = ("ls_lora",)
    FUNCTION = "load"
    CATEGORY = "Latent Shaper/Pipeline"

    def load(self, lora_name):
        path = folder_paths.get_full_path("loras", lora_name)
        assembly = load_lora_cached(path)
        
        # Detect and log architecture using raw keys
        spec = ModelRegistry.get_spec(assembly.get_raw_keys())
        print(f"[Latent Shaper] Loaded '{lora_name}'. Detected Architecture: {spec.name}")
        
        return ({"assembly": assembly, "name": lora_name},)

class LS_EQ:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ls_lora": ("LS_LORA",),
                "eq_global": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "eq_in": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "eq_mid": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "eq_out": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "eq_adapter": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "eq_other": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("LS_LORA",)
    FUNCTION = "process"
    CATEGORY = "Latent Shaper/Pipeline"

    def process(self, ls_lora, eq_global, eq_in, eq_mid, eq_out, eq_adapter, eq_other):
        assembly = get_assembly(ls_lora).clone()
        # Use raw keys for accurate detection
        spec = ModelRegistry.get_spec(assembly.get_raw_keys())
        
        for name, mod in assembly.modules.items():
            b_idx = spec.get_block_number(name)
            region = spec.get_region(b_idx)
            
            m = eq_global
            if region == "IN": m *= eq_in
            elif region == "MID": m *= eq_mid
            elif region == "OUT": m *= eq_out
            elif region == "ADAPTER": m *= eq_adapter
            else: m *= eq_other
            
            mod.apply_scale(m)
            
        return ({"assembly": assembly, "name": ls_lora.get("name", "eq_result")},)

class LS_Filters:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ls_lora": ("LS_LORA",),
                "fft_cutoff": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                "band_stop_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "band_stop_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "homeostatic": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("LS_LORA",)
    FUNCTION = "process"
    CATEGORY = "Latent Shaper/Pipeline"

    def process(self, ls_lora, fft_cutoff, band_stop_start, band_stop_end, homeostatic):
        assembly = get_assembly(ls_lora).clone()
        spec = ModelRegistry.get_spec(assembly.get_raw_keys())
        
        params = {
            "fft_cutoff": fft_cutoff,
            "band_stop_enabled": band_stop_end > band_stop_start,
            "band_stop_start": band_stop_start,
            "band_stop_end": band_stop_end,
            "homeostatic": homeostatic
        }
        
        for name, mod in assembly.modules.items():
            if mod.is_decomposed: mod.compose()
            delta = mod.up.float() @ mod.down.float()
            
            b_idx = spec.get_block_number(name)
            filtered = TensorProcessor.apply_filters(delta, params, b_idx=b_idx)
            
            if filtered is not None:
                nd, nu, nr = MathKernel.svd_decomposition(filtered, mod.rank)
                mod.down = nd.to(dtype=torch.bfloat16)
                mod.up = nu.to(dtype=torch.bfloat16)
                mod.alpha = float(nr)
                mod.is_decomposed = False
            
        return ({"assembly": assembly, "name": ls_lora.get("name", "filter_result")},)

class LS_Dynamics:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ls_lora": ("LS_LORA",),
                "spectral_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "dare_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01}),
                "clamp": ("FLOAT", {"default": 1.0, "min": 0.8, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("LS_LORA",)
    FUNCTION = "process"
    CATEGORY = "Latent Shaper/Pipeline"

    def process(self, ls_lora, spectral_threshold, dare_rate, clamp):
        assembly = get_assembly(ls_lora).clone()
        
        params = {
            "spectral_enabled": spectral_threshold > 0,
            "spectral_threshold": spectral_threshold,
            "dare_enabled": dare_rate > 0,
            "dare_rate": dare_rate,
            "clamp_quantile": clamp
        }
        
        for mod in assembly.modules.values():
            # Optimization: if only spectral gate is used, we can work on S directly
            if params["spectral_enabled"] and not params["dare_enabled"] and params["clamp_quantile"] == 1.0:
                mod.decompose()
                if mod.s is not None:
                    mod.s = torch.where(mod.s < spectral_threshold, torch.tensor(0.0, device=mod.s.device), mod.s)
            else:
                if mod.is_decomposed: mod.compose()
                delta = mod.up.float() @ mod.down.float()
                filtered = TensorProcessor.apply_filters(delta, params)
                if filtered is not None:
                    nd, nu, nr = MathKernel.svd_decomposition(filtered, mod.rank)
                    mod.down = nd.to(dtype=torch.bfloat16)
                    mod.up = nu.to(dtype=torch.bfloat16)
                    mod.alpha = float(nr)
                    mod.is_decomposed = False

        return ({"assembly": assembly, "name": ls_lora.get("name", "dynamics_result")},)

class LS_Eraser:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ls_lora": ("LS_LORA",),
                "erase_blocks": ("STRING", {"default": "", "multiline": False}),
                "erase_concepts": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {"clip": ("CLIP",),}
        }
    
    RETURN_TYPES = ("LS_LORA",)
    FUNCTION = "process"
    CATEGORY = "Latent Shaper/Pipeline"

    def process(self, ls_lora, erase_blocks, erase_concepts, clip=None):
        assembly = get_assembly(ls_lora).clone()
        spec = ModelRegistry.get_spec(assembly.get_raw_keys())
        block_set = MathKernel.parse_block_string(erase_blocks)
        
        keys_to_remove = []
        for name in assembly.modules.keys():
            b_idx = spec.get_block_number(name)
            if b_idx in block_set:
                keys_to_remove.append(name)
        
        for k in keys_to_remove:
            del assembly.modules[k]

        if erase_concepts and erase_concepts.strip() and clip:
            concept_vectors = []
            for c in erase_concepts.split(","):
                c = c.strip()
                if not c: continue
                tokens = clip.tokenize(c)
                cond, _ = clip.encode_from_tokens(tokens, return_pooled=True)
                if cond.shape[1] > 0:
                    vec = torch.mean(cond[0], dim=0).float() 
                    vec = vec / (torch.norm(vec) + 1e-9)
                    concept_vectors.append(vec.to("cuda" if torch.cuda.is_available() else "cpu"))
            
            if concept_vectors:
                for mod in assembly.modules.values():
                    if mod.is_decomposed: mod.compose()
                    delta = mod.up.float() @ mod.down.float()
                    for vec in concept_vectors:
                        delta = MathKernel.orthogonalize_rows_against_vector(delta, vec)
                    
                    nd, nu, nr = MathKernel.svd_decomposition(delta, mod.rank)
                    mod.down = nd.to(dtype=torch.bfloat16)
                    mod.up = nu.to(dtype=torch.bfloat16)
                    mod.is_decomposed = False

        return ({"assembly": assembly, "name": ls_lora.get("name", "eraser_result")},)

class LS_Metadata:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ls_lora": ("LS_LORA",),
                "new_name": ("STRING", {"default": ""}),
                "trigger_words": ("STRING", {"default": ""}),
                "description": ("STRING", {"default": "", "multiline": True}),
                "merge_mode": (["Passthrough", "Replace", "Clear"],),
            }
        }
    
    RETURN_TYPES = ("LS_LORA",)
    FUNCTION = "edit"
    CATEGORY = "Latent Shaper/Pipeline"

    def edit(self, ls_lora, new_name, trigger_words, description, merge_mode):
        assembly = get_assembly(ls_lora).clone()
        meta = assembly.metadata
        
        if merge_mode == "Clear": meta.clear()
        if merge_mode in ["Replace", "Clear"]:
            if new_name: meta["ss_output_name"] = new_name
            if trigger_words: meta["ss_tag_frequency"] = json.dumps({t.strip(): 1 for t in trigger_words.split(",") if t.strip()})
            if description: meta["modelspec.description"] = description
            
        assembly.metadata = meta
        return ({"assembly": assembly, "name": ls_lora.get("name", "meta_result")},)

class LS_Analyzer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"ls_lora": ("LS_LORA",), "mode": (["Basic Stats", "Block Heatmap"],),}}
    
    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "analyze"
    CATEGORY = "Latent Shaper/Pipeline"
    
    def analyze(self, ls_lora, mode):
        assembly = get_assembly(ls_lora)
        spec = ModelRegistry.get_spec(assembly.get_raw_keys())
        
        total_mag = 0.0
        count = 0
        
        # Dynamic block count from architecture spec
        b_count = spec.block_count if spec.block_count > 0 else 30
        block_energy = [0.0] * b_count
        
        for name, mod in assembly.modules.items():
            if mod.down is not None:
                mag = torch.mean(torch.abs(mod.down.float())).item()
                total_mag += mag
                count += 1
                b_idx = spec.get_block_number(name)
                if 0 <= b_idx < b_count: 
                    block_energy[b_idx] += mag
            
        avg_mag = total_mag / count if count > 0 else 0
        
        plt.figure(figsize=(10, 4))
        if mode == "Block Heatmap":
            plt.bar(range(b_count), block_energy, color='skyblue')
            plt.title(f"Block Energy ({ls_lora.get('name', 'Unknown')}) - {spec.name}")
            plt.xlabel("Block Index")
            plt.ylabel("Magnitude")
        else:
            plt.text(0.1, 0.5, f"Arch: {spec.name}\nAvg Mag: {avg_mag:.5f}\nModules: {len(assembly.modules)}", fontsize=14)
            plt.axis('off')
            
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        return (img_tensor.unsqueeze(0), f"Mag: {avg_mag:.6f} | Arch: {spec.name}")

class LS_Save:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ls_lora": ("LS_LORA",),
                "filename_prefix": ("STRING", {"default": "latent_shaper/my_lora"}),
                "precision": (["FP16", "BF16", "FP32"],),
                "save_metadata": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "save"
    CATEGORY = "Latent Shaper/Pipeline"
    OUTPUT_NODE = True

    def save(self, ls_lora, filename_prefix, precision, save_metadata):
        wrapper = ls_lora
        if "assembly" not in wrapper:
            wrapper = {"assembly": get_assembly(ls_lora)}

        output_dir = folder_paths.get_output_directory()
        full_output_dir = os.path.dirname(os.path.join(output_dir, filename_prefix))
        base_name = os.path.basename(filename_prefix)
        
        os.makedirs(full_output_dir, exist_ok=True)
        
        counter = 1
        filename = f"{base_name}.safetensors"
        while os.path.exists(os.path.join(full_output_dir, filename)):
            filename = f"{base_name}_{counter:02d}.safetensors"
            counter += 1
            
        path = os.path.join(full_output_dir, filename)
        save_ls_lora(wrapper, path, precision, save_metadata)
        return (path,)

class LS_Apply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "ls_lora": ("LS_LORA",),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply"
    CATEGORY = "Latent Shaper/Pipeline"

    def apply(self, model, clip, ls_lora, strength):
        assembly = get_assembly(ls_lora)
        return apply_lora_assembly(model, clip, assembly, strength)