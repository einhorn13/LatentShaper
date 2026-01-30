# core/naming.py

import os
from typing import Optional
from .config import ConfigManager

class NamingManager:
    """
    Handles file naming logic for batch and single operations.
    Ensures all relative paths are resolved against the global 'output_dir'.
    """

    @staticmethod
    def _get_base_dir() -> str:
        return ConfigManager().output_dir

    @staticmethod
    def _resolve_dir(target: str) -> str:
        """
        If target is absolute, return as is.
        If relative, join with global output_dir.
        """
        if os.path.isabs(target):
            return target
        return os.path.join(NamingManager._get_base_dir(), target)

    @staticmethod
    def resolve_output_path(
        input_path: str, 
        output_target: str, 
        suffix: str = "",
        is_batch: bool = False
    ) -> str:
        """
        Determines the final output filename for Transform operations (Extract, Resize, Morph).
        """
        # 1. Prepare Base Name from Input
        base_name = os.path.basename(input_path)
        name, ext = os.path.splitext(base_name)
        if suffix and not name.endswith(suffix):
            filename = f"{name}{suffix}{ext}"
        else:
            filename = f"{name}{ext}"

        # 2. Handle Empty Target -> Default Dir + Filename
        if not output_target or output_target.strip() == "":
            return os.path.join(NamingManager._get_base_dir(), filename)

        # 3. Resolve Target Path (Relative -> Absolute via Output Dir)
        resolved_target = NamingManager._resolve_dir(output_target)

        # 4. Handle Batch Mode (Target is strictly a folder)
        if is_batch:
            # Even if user typed "file.safetensors", in batch mode we treat it as folder or take parent
            if os.path.splitext(resolved_target)[1]:
                return os.path.join(os.path.dirname(resolved_target), filename)
            return os.path.join(resolved_target, filename)

        # 5. Handle Single Mode
        # Check if target looks like a directory (no extension)
        if not os.path.splitext(resolved_target)[1]:
            return os.path.join(resolved_target, filename)
        
        # Target is a specific filename
        return resolved_target

    @staticmethod
    def resolve_merge_path(output_target: str, default_name: str = "merged.safetensors") -> str:
        """
        Determines the output filename for Merge operations (Many-to-One).
        """
        # 1. Handle Empty -> Default Dir + Default Name
        if not output_target or output_target.strip() == "":
            return os.path.join(NamingManager._get_base_dir(), default_name)

        # 2. Resolve Path
        resolved_target = NamingManager._resolve_dir(output_target)

        # 3. Ensure Extension
        if not os.path.splitext(resolved_target)[1]:
            # User provided a folder name or filename without extension
            # We assume filename if it doesn't end with separator
            if output_target.endswith(os.sep):
                return os.path.join(resolved_target, default_name)
            else:
                return f"{resolved_target}.safetensors"
        
        return resolved_target

    @staticmethod
    def ensure_dir(path: str):
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)