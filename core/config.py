
import os
import json
from typing import Any, Dict
from .logger import Logger

class ConfigManager:
    """
    Singleton to manage application settings.
    Enhanced with Environment Variable support and robust path discovery.
    """
    _instance = None
    _config_path = "config.json"
    
    _defaults = {
        "output_dir": "output",
        "checkpoints_dir": "",
        "loras_dir": ""
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._data = cls._defaults.copy()
            cls._instance.load()
            cls._instance._auto_discover_paths()
        return cls._instance

    def load(self):
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # Support for legacy Windows paths in JSON: ensure they are handled safely
                    self._data.update(user_config)
            except Exception as e:
                Logger.error(f"Error loading config.json: {e}")
        else:
            self.save()

    def save(self):
        try:
            with open(self._config_path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=4)
        except Exception as e:
            Logger.error(f"Error saving config: {e}")

    def _auto_discover_paths(self):
        """
        Attempts to find ComfyUI model directories using ENV vars or relative structure.
        Priority: 1. User Config, 2. ENV Vars, 3. Relative Discovery.
        """
        # Check ENV variables first (standard for containers/installers)
        env_ckpt = os.environ.get("COMFYUI_CHECKPOINTS")
        env_loras = os.environ.get("COMFYUI_LORAS")
        env_root = os.environ.get("COMFYUI_PATH")

        # 1. Checkpoint Dir
        if not self._data.get("checkpoints_dir") or not os.path.exists(self._data["checkpoints_dir"]):
            if env_ckpt and os.path.exists(env_ckpt):
                self._data["checkpoints_dir"] = env_ckpt
            elif env_root:
                p = os.path.join(env_root, "models", "checkpoints")
                if os.path.exists(p): self._data["checkpoints_dir"] = p
            else:
                # Relative fallback from custom_nodes/LatentShaper/core/
                try:
                    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
                    p = os.path.join(base, "models", "checkpoints")
                    if os.path.exists(p): self._data["checkpoints_dir"] = p
                except: pass

        # 2. LoRA Dir
        if not self._data.get("loras_dir") or not os.path.exists(self._data["loras_dir"]):
            if env_loras and os.path.exists(env_loras):
                self._data["loras_dir"] = env_loras
            elif env_root:
                p = os.path.join(env_root, "models", "loras")
                if os.path.exists(p): self._data["loras_dir"] = p
            else:
                try:
                    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
                    p = os.path.join(base, "models", "loras")
                    if os.path.exists(p): self._data["loras_dir"] = p
                except: pass

        if not self._data["checkpoints_dir"]:
            Logger.warning("ConfigManager: Checkpoints directory not found. Please set it in Settings.")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        self._data[key] = value
        self.save()

    @property
    def output_dir(self) -> str:
        path = self._data.get("output_dir", "output")
        return path if os.path.isabs(path) else os.path.abspath(path)

    @property
    def checkpoints_dir(self) -> str:
        return self._data.get("checkpoints_dir", "")

    @property
    def loras_dir(self) -> str:
        return self._data.get("loras_dir", "")