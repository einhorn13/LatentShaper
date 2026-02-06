# core/configs.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union

@dataclass
class BaseConfig:
    save_to_workspace: bool = False

@dataclass
class ExtractConfig(BaseConfig):
    rank: int = 64
    threshold: float = 0.0
    baked_scale: float = 1.0
    manual_alpha: Optional[float] = None

@dataclass
class ResizeConfig(BaseConfig):
    rank: int = 32
    auto_rank_threshold: float = 0.0

@dataclass
class MorphConfig(BaseConfig):
    # EQ
    eq_global: float = 1.0
    eq_in: float = 1.0
    eq_mid: float = 1.0
    eq_out: float = 1.0
    eq_interpolate: bool = False
    # Dynamics
    temperature: float = 1.0
    fft_cutoff: float = 1.0
    clamp_quantile: float = 1.0
    fix_alpha: bool = False
    homeostatic: bool = False
    homeostatic_thr: float = 0.01
    # Filters
    spectral_enabled: bool = False
    spectral_threshold: float = 0.0
    spectral_remove_structure: bool = False
    spectral_adaptive: bool = False
    dare_enabled: bool = False
    dare_rate: float = 0.0
    band_stop_enabled: bool = False
    band_stop_start: float = 0.0
    band_stop_end: float = 0.0
    # Erasers
    eraser_start: int = 0
    eraser_end: int = 0
    erase_blocks: str = ""
    # Bridge
    is_bridge: bool = False
    strength: float = 0.5

@dataclass
class MergeConfig(BaseConfig):
    ratios: List[float] = field(default_factory=list)
    rank: int = 64
    algorithm: str = "SVD"
    global_strength: float = 1.0
    auto_rank_threshold: float = 0.0
    pruning_threshold: float = 0.0
    ties_density: float = 0.3

@dataclass
class CheckpointMergeConfig(BaseConfig):
    mode: str = "Weighted Sum"
    te_policy: str = "Copy A"
    vae_policy: str = "Copy A"
    precision: str = "BF16"
    lora_strength: float = 1.0
    weights: List[float] = field(default_factory=lambda: [0.5]*31)
    lora_path: Optional[str] = None

@dataclass
class UtilsConfig(BaseConfig):
    precision: str = "Keep"
    normalize_keys: bool = False
    target_alpha: Optional[float] = None
    alpha_equals_rank: bool = False