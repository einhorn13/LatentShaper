
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from .architectures.base import BaseArchitecture

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
    Uses dynamic prefixes from the architecture specification.
    """

    _LORA_DOWN_PATTERN = re.compile(r"(.*)\.(lora_down|down_proj|lora_A|lora\.down)\.weight")
    _LORA_UP_PATTERN = re.compile(r"(.*)\.(lora_up|up_proj|lora_B|lora\.up)\.weight")
    _ALPHA_PATTERN = re.compile(r"(.*)\.(alpha|lora_alpha|lora\.alpha)")

    _DEFAULT_LORA_PREFIXES = ["lora_unet_", "lora_unet.", "lora_te_", "lora_te.", "lora_model_", "lora_model."]
    _DEFAULT_MODEL_PREFIXES = ["model.diffusion_model.", "first_stage_model.", "cond_stage_model.", "transformer.", "model.", "diffusion_model."]

    @staticmethod
    def _get_prefixes(spec: BaseArchitecture = None) -> Tuple[List[str], List[str]]:
        if spec: return spec.lora_prefixes, spec.model_prefixes
        return FormatHandler._DEFAULT_LORA_PREFIXES, FormatHandler._DEFAULT_MODEL_PREFIXES

    @staticmethod
    def split_prefix(raw_base: str, spec: BaseArchitecture = None) -> Tuple[str, str]:
        lora_prefixes, _ = FormatHandler._get_prefixes(spec)
        for p in lora_prefixes:
            # Check exact match or underscore version (e.g. transformer. -> transformer_)
            if raw_base.startswith(p): return p, raw_base[len(p):]
            p_alt = p.replace(".", "_")
            if raw_base.startswith(p_alt): return p_alt, raw_base[len(p_alt):]
        return "", raw_base

    @staticmethod
    def get_base_name(structure_name: str, spec: BaseArchitecture = None) -> str:
        _, model_prefixes = FormatHandler._get_prefixes(spec)
        res = structure_name
        changed = True
        while changed:
            changed = False
            for p in model_prefixes:
                if res.startswith(p):
                    res = res[len(p):]
                    changed = True
                    break
                p_alt = p.replace(".", "_")
                if res.startswith(p_alt):
                    res = res[len(p_alt):]
                    changed = True
                    break
        return res

    @staticmethod
    def fix_key_name(key: str, spec: BaseArchitecture = None) -> str:
        """
        Robust key fixer that handles standard ComfyUI formatting, 
        broken underscores, and specific layer naming conventions.
        """
        lora_prefixes, model_prefixes = FormatHandler._get_prefixes(spec)
        new_key = key.replace("lora_unet__", "") 
        
        if any(x in key for x in ["text_encoders", "text_encoder", "te.", "clip", "lora_te_"]):
            return FormatHandler.convert_to_kohya_key(key, spec) + ".weight"

        suffix = ""
        if new_key.endswith(".weight"):
            if ".lora_down.weight" in new_key: suffix = ".lora_down.weight"
            elif ".lora_up.weight" in new_key: suffix = ".lora_up.weight"
            elif ".alpha" in new_key: suffix = ".alpha"
            elif ".weight" in new_key: suffix = ".weight"
            
        core_part = new_key
        if suffix: core_part = new_key[:-len(suffix)]
            
        for p in lora_prefixes + model_prefixes:
            if core_part.startswith(p):
                core_part = core_part[len(p):]
                break
            p_alt = p.replace(".", "_")
            if core_part.startswith(p_alt):
                core_part = core_part[len(p_alt):]
                break

        core_part = re.sub(r'layers_(\d+)_', r'layers.\1.', core_part)
        core_part = re.sub(r'context_refiner_(\d+)_', r'context_refiner.\1.', core_part)
        core_part = re.sub(r'noise_refiner_(\d+)_', r'noise_refiner.\1.', core_part)

        for t in ["to_k", "to_q", "to_v"]:
             if f"_{t}" in core_part: core_part = core_part.replace(f"_{t}", f".{t}")
        
        if "_to_out" in core_part:
            core_part = core_part.replace("_to_out", ".to_out")
            core_part = re.sub(r'\.to_out_(\d+)', r'.to_out.\1', core_part)

        return f"diffusion_model.{core_part}{suffix}"

    @staticmethod
    def convert_to_kohya_key(base_key: str, spec: BaseArchitecture = None) -> str:
        """
        Converts any key to Kohya-ss format (ComfyUI standard underscore format).
        """
        _, model_prefixes = FormatHandler._get_prefixes(spec)
        core_name = base_key
        if core_name.endswith(".weight"): core_name = core_name[:-7]
            
        sorted_wrappers = sorted(model_prefixes, key=len, reverse=True)
        for prefix in sorted_wrappers:
            if core_name.startswith(prefix):
                core_name = core_name[len(prefix):]
                break 
            p_alt = prefix.replace(".", "_")
            if core_name.startswith(p_alt):
                core_name = core_name[len(p_alt):]
                break
        
        if any(x in base_key for x in ["text_encoders", "text_encoder", "te.", "clip"]): lora_prefix = "lora_te_"
        else: lora_prefix = "lora_unet_"
            
        kohya_name = core_name.replace(".", "_").replace("__", "_")
        if kohya_name.startswith("_"): kohya_name = kohya_name.lstrip("_")
        return f"{lora_prefix}{kohya_name}"

    @staticmethod
    def group_keys(keys: list[str], spec: BaseArchitecture = None, normalize: bool = True) -> list[LoRAGroup]:
        groups = {}
        for key in keys:
            match_down = FormatHandler._LORA_DOWN_PATTERN.match(key)
            if match_down:
                raw_base = match_down.group(1)
                prefix, struct = FormatHandler.split_prefix(raw_base, spec)
                base = FormatHandler.get_base_name(struct, spec) if normalize else struct
                if base not in groups: groups[base] = {'struct': struct, 'prefix': prefix}
                groups[base]['down'] = key
                continue

            match_up = FormatHandler._LORA_UP_PATTERN.match(key)
            if match_up:
                raw_base = match_up.group(1)
                prefix, struct = FormatHandler.split_prefix(raw_base, spec)
                base = FormatHandler.get_base_name(struct, spec) if normalize else struct
                if base not in groups: groups[base] = {'struct': struct, 'prefix': prefix}
                groups[base]['up'] = key
                continue

            match_alpha = FormatHandler._ALPHA_PATTERN.match(key)
            if match_alpha:
                raw_base = match_alpha.group(1)
                prefix, struct = FormatHandler.split_prefix(raw_base, spec)
                base = FormatHandler.get_base_name(struct, spec) if normalize else struct
                if base not in groups: groups[base] = {'struct': struct, 'prefix': prefix}
                groups[base]['alpha'] = key
                continue

        result = []
        for base, parts in groups.items():
            if 'down' in parts and 'up' in parts:
                result.append(LoRAGroup(
                    base_name=base, structure_name=parts['struct'], prefix=parts['prefix'],
                    down_key=parts['down'], up_key=parts['up'], alpha_key=parts.get('alpha')
                ))
        return result