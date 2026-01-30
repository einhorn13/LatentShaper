# nodes_pipeline.py

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
from .core.comfy_utils import load_lora_cached, process_lora_dict, apply_lora_dict, save_z_lora
from .core.tensor_processor import TensorProcessor
from .core.model_specs import ModelRegistry

class LS_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"lora_name": (folder_paths.get_filename_list("loras"),)}}
    RETURN_TYPES = ("Z_LORA",)
    RETURN_NAMES = ("z_lora",)
    FUNCTION = "load"
    CATEGORY = "LoRA Studio/Pipeline"
    def load(self, lora_name):
        path = folder_paths.get_full_path("loras", lora_name)
        sd, meta = load_lora_cached(path)
        return ({"sd": sd, "name": lora_name, "metadata": meta},)

class LS_EQ:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "z_lora": ("Z_LORA",),
                "eq_global": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "eq_in": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "eq_mid": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "eq_out": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("Z_LORA",)
    FUNCTION = "process"
    CATEGORY = "LoRA Studio/Pipeline"
    def process(self, z_lora, eq_global, eq_in, eq_mid, eq_out):
        def callback(delta, b_idx, region, grp):
            m = eq_global
            if region == "IN": m *= eq_in
            elif region == "MID": m *= eq_mid
            elif region == "OUT": m *= eq_out
            return TensorProcessor.apply_filters(delta, {}, eq_factor=m, b_idx=b_idx)
        new_sd = process_lora_dict(z_lora["sd"], callback)
        return ({"sd": new_sd, "name": z_lora["name"], "metadata": z_lora.get("metadata", {})},)

class LS_Filters:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "z_lora": ("Z_LORA",),
                "fft_cutoff": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                "band_stop_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "band_stop_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "homeostatic": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("Z_LORA",)
    FUNCTION = "process"
    CATEGORY = "LoRA Studio/Pipeline"
    def process(self, z_lora, fft_cutoff, band_stop_start, band_stop_end, homeostatic):
        params = {
            "fft_cutoff": fft_cutoff,
            "band_stop_enabled": band_stop_end > band_stop_start,
            "band_stop_start": band_stop_start,
            "band_stop_end": band_stop_end,
            "homeostatic": homeostatic
        }
        def callback(delta, b_idx, region, grp):
            return TensorProcessor.apply_filters(delta, params, b_idx=b_idx)
        new_sd = process_lora_dict(z_lora["sd"], callback)
        return ({"sd": new_sd, "name": z_lora["name"], "metadata": z_lora.get("metadata", {})},)

class LS_Dynamics:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "z_lora": ("Z_LORA",),
                "spectral_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "dare_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01}),
                "clamp": ("FLOAT", {"default": 1.0, "min": 0.8, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("Z_LORA",)
    FUNCTION = "process"
    CATEGORY = "LoRA Studio/Pipeline"
    def process(self, z_lora, spectral_threshold, dare_rate, clamp):
        params = {
            "spectral_enabled": spectral_threshold > 0,
            "spectral_threshold": spectral_threshold,
            "dare_enabled": dare_rate > 0,
            "dare_rate": dare_rate,
            "clamp_quantile": clamp
        }
        def callback(delta, b_idx, region, grp):
            return TensorProcessor.apply_filters(delta, params, b_idx=b_idx)
        new_sd = process_lora_dict(z_lora["sd"], callback)
        return ({"sd": new_sd, "name": z_lora["name"], "metadata": z_lora.get("metadata", {})},)

class LS_Eraser:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "z_lora": ("Z_LORA",),
                "erase_blocks": ("STRING", {"default": "", "multiline": False}),
                "erase_concepts": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {"clip": ("CLIP",),}
        }
    RETURN_TYPES = ("Z_LORA",)
    FUNCTION = "process"
    CATEGORY = "LoRA Studio/Pipeline"
    def process(self, z_lora, erase_blocks, erase_concepts, clip=None):
        block_set = MathKernel.parse_block_string(erase_blocks)
        concept_vectors = []
        
        if erase_concepts and erase_concepts.strip():
            if clip is None:
                print("[LoRA Studio] Warning: 'erase_concepts' provided but CLIP input is missing. Concept erasure skipped.")
            else:
                for c in erase_concepts.split(","):
                    c = c.strip()
                    if not c: continue
                    tokens = clip.tokenize(c)
                    cond, _ = clip.encode_from_tokens(tokens, return_pooled=True)
                    if cond.shape[1] > 0:
                        vec = torch.mean(cond[0], dim=0).float() 
                        vec = vec / (torch.norm(vec) + 1e-9)
                        concept_vectors.append(vec.to("cuda" if torch.cuda.is_available() else "cpu"))
        
        params = {
            "erase_blocks_set": block_set,
            "concept_vectors": concept_vectors
        }
        def callback(delta, b_idx, region, grp):
            return TensorProcessor.apply_filters(delta, params, b_idx=b_idx)
        new_sd = process_lora_dict(z_lora["sd"], callback)
        return ({"sd": new_sd, "name": z_lora["name"], "metadata": z_lora.get("metadata", {})},)

class LS_Metadata:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "z_lora": ("Z_LORA",),
                "new_name": ("STRING", {"default": ""}),
                "trigger_words": ("STRING", {"default": ""}),
                "description": ("STRING", {"default": "", "multiline": True}),
                "merge_mode": (["Passthrough", "Replace", "Clear"],),
            }
        }
    RETURN_TYPES = ("Z_LORA",)
    FUNCTION = "edit"
    CATEGORY = "LoRA Studio/Pipeline"
    def edit(self, z_lora, new_name, trigger_words, description, merge_mode):
        meta = z_lora.get("metadata", {}).copy()
        if merge_mode == "Clear": meta = {}
        if merge_mode in ["Replace", "Clear"]:
            if new_name: meta["ss_output_name"] = new_name
            if trigger_words: meta["ss_tag_frequency"] = json.dumps({t.strip(): 1 for t in trigger_words.split(",") if t.strip()})
            if description: meta["modelspec.description"] = description
        return ({"sd": z_lora["sd"], "name": z_lora["name"], "metadata": meta},)

class LS_Analyzer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"z_lora": ("Z_LORA",), "mode": (["Basic Stats", "Block Heatmap"],),}}
    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "analyze"
    CATEGORY = "LoRA Studio/Pipeline"
    def analyze(self, z_lora, mode):
        sd = z_lora["sd"]
        spec = ModelRegistry.get_spec(list(sd.keys()))
        total_mag = 0.0
        count = 0
        block_energy = [0.0] * 30
        for k, v in sd.items():
            if "lora_down" in k:
                mag = torch.mean(torch.abs(v.float())).item()
                total_mag += mag
                count += 1
                b_idx = spec.get_block_number(k)
                if 0 <= b_idx < 30: block_energy[b_idx] += mag
        avg_mag = total_mag / count if count > 0 else 0
        
        plt.figure(figsize=(10, 4))
        if mode == "Block Heatmap":
            plt.bar(range(30), block_energy, color='skyblue')
            plt.title(f"Block Energy ({z_lora.get('name')})")
        else:
            plt.text(0.1, 0.5, f"Avg Mag: {avg_mag:.5f}\nKeys: {len(sd)}", fontsize=14)
            plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        return (img_tensor.unsqueeze(0), f"Mag: {avg_mag:.6f}")

class LS_Save:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "z_lora": ("Z_LORA",),
                "filename_prefix": ("STRING", {"default": "lora_studio/my_lora"}),
                "precision": (["FP16", "BF16", "FP32"],),
                "save_metadata": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "save"
    CATEGORY = "LoRA Studio/Pipeline"
    OUTPUT_NODE = True
    def save(self, z_lora, filename_prefix, precision, save_metadata):
        output_dir = folder_paths.get_output_directory()
        full_output_dir = os.path.dirname(os.path.join(output_dir, filename_prefix))
        base_name = os.path.basename(filename_prefix)
        counter = 1
        filename = f"{base_name}.safetensors"
        while os.path.exists(os.path.join(full_output_dir, filename)):
            filename = f"{base_name}_{counter:02d}.safetensors"
            counter += 1
        path = os.path.join(full_output_dir, filename)
        save_z_lora(z_lora, path, precision, save_metadata)
        return (path,)

class LS_Apply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "z_lora": ("Z_LORA",),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply"
    CATEGORY = "LoRA Studio/Pipeline"
    def apply(self, model, clip, z_lora, strength):
        return apply_lora_dict(model, clip, z_lora["sd"], strength)