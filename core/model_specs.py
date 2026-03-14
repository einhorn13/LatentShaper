
import os
import sys
import importlib.util
import inspect
from typing import List
from core.architectures.base import BaseArchitecture, UnknownArchitecture
from core.logger import Logger

class ModelRegistry:
    _architectures: List[BaseArchitecture] = []
    _initialized = False

    @classmethod
    def _initialize(cls):
        if cls._initialized: return
        cls._architectures = []
        
        # Устанавливаем корень проекта в sys.path
        core_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(core_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
        arch_dir = os.path.join(core_dir, "architectures")
        
        for filename in os.listdir(arch_dir):
            if filename.endswith(".py") and filename != "base.py" and not filename.startswith("__"):
                file_path = os.path.join(arch_dir, filename)
                module_name = f"core.architectures.{filename[:-3]}"
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, BaseArchitecture) and obj not in [BaseArchitecture, UnknownArchitecture]:
                            instance = obj()
                            cls._architectures.append(instance)
                            Logger.info(f"Loaded architecture plugin: {instance.name}")
                except Exception as e:
                    Logger.error(f"Failed to load plugin {filename}: {e}")
        cls._initialized = True

    @classmethod
    def get_spec(cls, keys: List[str]) -> BaseArchitecture:
        cls._initialize()
        if not keys: return UnknownArchitecture()
        
        logs = {}
        for spec in cls._architectures:
            if spec.detect(keys): return spec
            logs[spec.name] = spec.diagnose(keys)
        
        Logger.error("--- MODEL DETECTION FAILED ---")
        for arch, issues in logs.items():
            Logger.error(f"  > [{arch}]: {', '.join(issues)}")
        return UnknownArchitecture()

ModelSpec = BaseArchitecture