# core/format_handler.py

import re
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from .model_specs import ModelSpec, UnknownSpec

@dataclass
class LoRAGroup:
    base_name: str
    structure_name: str
    prefix: str
    down_key: str
    up_key: str
    alpha_key: Optional[str] = None

class FormatHandler:
    """
    Parses raw keys from safetensors and groups them into logical LoRA units.
    """

    _LORA_DOWN_PATTERN = re.compile(r"(.*)\.(lora_down|down_proj|lora_A|lora\.down)\.weight")
    _LORA_UP_PATTERN = re.compile(r"(.*)\.(lora_up|up_proj|lora_B|lora\.up)\.weight")
    _ALPHA_PATTERN = re.compile(r"(.*)\.(alpha|lora_alpha|lora\.alpha)")

    _LORA_PREFIXES = [
        "lora_unet_", "lora_unet.", "lora_te_", "lora_te.",
        "lora_model_", "lora_model.", "lora_te1_", "lora_te2_"
    ]

    _MODEL_PREFIXES = [
        "model.diffusion_model.", "first_stage_model.", "cond_stage_model.",
        "transformer.", "model.", "diffusion_model.",
        "text_encoders.qwen.", "text_encoders.clip.", "text_encoder.", "te."
    ]

    @staticmethod
    def split_prefix(raw_base: str) -> Tuple[str, str]:
        for p in FormatHandler._LORA_PREFIXES:
            if raw_base.startswith(p):
                return p, raw_base[len(p):]
        return "", raw_base

    @staticmethod
    def get_base_name(structure_name: str) -> str:
        res = structure_name
        changed = True
        while changed:
            changed = False
            for p in FormatHandler._MODEL_PREFIXES:
                if res.startswith(p):
                    res = res[len(p):]
                    changed = True
        return res

    @staticmethod
    def fix_key_name(key: str) -> str:
        """
        Robust key fixer that handles standard ComfyUI formatting, 
        broken underscores, and specific layer naming conventions.
        Integrates logic from Z-Image community fix scripts.
        """
        # 0. Handle basic cleanup
        new_key = key.replace("lora_unet__", "") # Fix double underscores
        
        # Check if this is a Text Encoder key (TE) - usually preserve or standard cleanup
        if any(x in key for x in ["text_encoders", "text_encoder", "te.", "clip", "lora_te_"]):
            # Simple fallback for TE to ensure dot notation if preferred, 
            # or return standard kohya format if that's safer.
            # For now, let's use the standard Kohya converter for TE to be safe.
            return FormatHandler.convert_to_kohya_key(key) + ".weight"

        # --- UNET / DIT FIXES (Based on provided script) ---
        
        # 1. Strip known prefixes to isolate the block name
        # We need to find where the "layers" or structure begins
        # E.g. "lora_unet_layers_0..." -> "layers_0..."
        # But we must preserve the suffix (lora_down.weight)
        
        # Identify suffix
        suffix = ""
        if new_key.endswith(".weight"):
            # Try to split logical parts
            if ".lora_down.weight" in new_key: suffix = ".lora_down.weight"
            elif ".lora_up.weight" in new_key: suffix = ".lora_up.weight"
            elif ".alpha" in new_key: suffix = ".alpha"
            elif ".weight" in new_key: suffix = ".weight"
            
        core_part = new_key
        if suffix:
            core_part = new_key[:-len(suffix)]
            
        # Clean prefix from core_part
        for p in FormatHandler._LORA_PREFIXES + FormatHandler._MODEL_PREFIXES:
            if core_part.startswith(p):
                core_part = core_part[len(p):]
                break

        # 2. Apply Regex Fixes to the core block name
        
        # Fix "layers_X_" -> "layers.X."
        core_part = re.sub(r'layers_(\d+)_', r'layers.\1.', core_part)
        
        # Fix "context_refiner_X_" -> "context_refiner.X."
        core_part = re.sub(r'context_refiner_(\d+)_', r'context_refiner.\1.', core_part)
        
        # Fix "noise_refiner_X_" -> "noise_refiner.X."
        core_part = re.sub(r'noise_refiner_(\d+)_', r'noise_refiner.\1.', core_part)

        # Fix Attention blocks
        # "attention_to_k" -> "attention.to_k"
        for t in ["to_k", "to_q", "to_v"]:
             if f"_{t}" in core_part:
                core_part = core_part.replace(f"_{t}", f".{t}")
        
        # "attention_to_out_0" -> "attention.to_out.0"
        if "_to_out" in core_part:
            core_part = core_part.replace("_to_out", ".to_out")
            core_part = re.sub(r'\.to_out_(\d+)', r'.to_out.\1', core_part)

        # 3. Reconstruct
        # The script prepends "diffusion_model." for UNet weights.
        # This is the internal ComfyUI format (Dot notation).
        final_key = f"diffusion_model.{core_part}{suffix}"
        
        return final_key

    @staticmethod
    def convert_to_kohya_key(base_key: str) -> str:
        """
        Converts any key to Kohya-ss format (ComfyUI standard underscore format).
        """
        core_name = base_key
        if core_name.endswith(".weight"):
            core_name = core_name[:-7]
            
        # Strip Wrapper Prefixes
        # Sort by length to handle nested prefixes correctly
        sorted_wrappers = sorted(FormatHandler._MODEL_PREFIXES, key=len, reverse=True)
        for prefix in sorted_wrappers:
            if core_name.startswith(prefix):
                core_name = core_name[len(prefix):]
                break 
        
        # Determine LoRA Prefix
        if any(x in base_key for x in ["text_encoders", "text_encoder", "te.", "clip"]):
            lora_prefix = "lora_te_"
        else:
            lora_prefix = "lora_unet_"
            
        # Replace separators
        kohya_name = core_name.replace(".", "_").replace("__", "_")
        if kohya_name.startswith("_"): kohya_name = kohya_name.lstrip("_")
        
        return f"{lora_prefix}{kohya_name}"

    @staticmethod
    def group_keys(keys: list[str], normalize: bool = True) -> list[LoRAGroup]:
        groups = {}
        for key in keys:
            match_down = FormatHandler._LORA_DOWN_PATTERN.match(key)
            if match_down:
                raw_base = match_down.group(1)
                prefix, struct = FormatHandler.split_prefix(raw_base)
                base = FormatHandler.get_base_name(struct) if normalize else struct
                if base not in groups: groups[base] = {'struct': struct, 'prefix': prefix}
                groups[base]['down'] = key
                continue

            match_up = FormatHandler._LORA_UP_PATTERN.match(key)
            if match_up:
                raw_base = match_up.group(1)
                prefix, struct = FormatHandler.split_prefix(raw_base)
                base = FormatHandler.get_base_name(struct) if normalize else struct
                if base not in groups: groups[base] = {'struct': struct, 'prefix': prefix}
                groups[base]['up'] = key
                continue

            match_alpha = FormatHandler._ALPHA_PATTERN.match(key)
            if match_alpha:
                raw_base = match_alpha.group(1)
                prefix, struct = FormatHandler.split_prefix(raw_base)
                base = FormatHandler.get_base_name(struct) if normalize else struct
                if base not in groups: groups[base] = {'struct': struct, 'prefix': prefix}
                groups[base]['alpha'] = key
                continue

        result = []
        for base, parts in groups.items():
            if 'down' in parts and 'up' in parts:
                result.append(LoRAGroup(
                    base_name=base,
                    structure_name=parts['struct'],
                    prefix=parts['prefix'],
                    down_key=parts['down'],
                    up_key=parts['up'],
                    alpha_key=parts.get('alpha')
                ))
        return result

    @staticmethod
    def get_block_region(key_name: str, spec: ModelSpec = None) -> str:
        if spec is None: spec = UnknownSpec()
        idx = spec.get_block_number(key_name)
        if idx == -1: return "OTHER"
        return spec.get_region(idx)