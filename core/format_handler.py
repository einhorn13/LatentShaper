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
    def convert_to_kohya_key(base_key: str) -> str:
        """
        Converts any key to Kohya-ss format (ComfyUI standard).
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