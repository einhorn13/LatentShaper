# core/config.py

import os
import json
from typing import Any, Dict

class ConfigManager:
    """
    Singleton to manage application settings.
    Stores config in 'config.json' in the application root.
    """
    _instance = None
    _config_path = "config.json"
    
    # Default settings
    _defaults = {
        "output_dir": "output",
        "checkpoints_dir": "",  # Path to base checkpoints
        "loras_dir": ""         # Path to LoRAs
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._data = cls._defaults.copy()
            cls._instance.load()
            cls._instance._auto_discover_paths()
        return cls._instance

    def load(self):
        """Loads config from disk, creating default if missing."""
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    self._data.update(user_config)
            except Exception as e:
                print(f"Error loading config: {e}")
        else:
            self.save()

    def save(self):
        """Saves current config to disk."""
        try:
            with open(self._config_path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def _auto_discover_paths(self):
        """Attempts to find ComfyUI model directories if not set."""
        # Assuming script is in ComfyUI/custom_nodes/Z-Image-Turbo-Tool/
        # We go up 3 levels to reach ComfyUI root
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        models_path = os.path.join(base_path, "models")
        
        if os.path.exists(models_path):
            if not self._data["checkpoints_dir"]:
                ckpt = os.path.join(models_path, "checkpoints")
                if os.path.exists(ckpt): self._data["checkpoints_dir"] = ckpt
            
            if not self._data["loras_dir"]:
                loras = os.path.join(models_path, "loras")
                if os.path.exists(loras): self._data["loras_dir"] = loras

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