
import torch
import copy
from typing import Dict, Optional, Tuple, Any, List
from .math import MathKernel
from .format_handler import FormatHandler

class LoRAModule:
    """
    Represents a single LoRA block (Down + Up + Alpha).
    Capable of holding decomposed state (SVD) to avoid re-computation.
    """
    def __init__(self, down: torch.Tensor, up: torch.Tensor, alpha: Optional[float] = None):
        self.down = down
        self.up = up
        self.alpha = alpha if alpha is not None else float(down.shape[0])
        
        # SVD Cache
        self.u: Optional[torch.Tensor] = None
        self.s: Optional[torch.Tensor] = None
        self.vh: Optional[torch.Tensor] = None
        self.is_decomposed = False

    @property
    def rank(self) -> int:
        if self.is_decomposed and self.s is not None:
            return self.s.shape[0]
        return self.down.shape[0]

    def to(self, device: str = None, dtype: torch.dtype = None):
        """Moves tensors to device/dtype in place."""
        if self.down is not None: self.down = self.down.to(device=device, dtype=dtype)
        if self.up is not None: self.up = self.up.to(device=device, dtype=dtype)
        if self.u is not None: self.u = self.u.to(device=device, dtype=dtype)
        if self.s is not None: self.s = self.s.to(device=device, dtype=dtype)
        if self.vh is not None: self.vh = self.vh.to(device=device, dtype=dtype)
        return self

    def decompose(self):
        """Performs SVD if not already done."""
        if self.is_decomposed: return
        
        # Reconstruct delta
        # Cast to float for precision
        d_float = self.down.float()
        u_float = self.up.float()
        
        # Scale by alpha/rank if needed, but usually we decompose the raw weight product
        # and handle alpha separately. Here we decompose the raw matrix product.
        delta = u_float @ d_float
        
        # SVD
        # We use a safe rank estimation
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        
        self.u = u
        self.s = s
        self.vh = vh
        self.is_decomposed = True
        
        # Clear raw weights to save memory if needed, 
        # but usually we keep them until modification.
        # self.down = None
        # self.up = None

    def compose(self):
        """Reconstructs Down/Up from SVD state."""
        if not self.is_decomposed: return

        # Reconstruct from U, S, Vh
        # Down = sqrt(S) @ Vh
        # Up = U @ sqrt(S)
        
        sqrt_s = torch.diag(torch.sqrt(self.s))
        
        # Keep original dtype
        target_dtype = self.down.dtype if self.down is not None else torch.bfloat16
        
        self.down = (sqrt_s @ self.vh).to(dtype=target_dtype)
        self.up = (self.u @ sqrt_s).to(dtype=target_dtype)
        
        # Reset cache
        self.is_decomposed = False
        self.u = None
        self.s = None
        self.vh = None

    def apply_scale(self, factor: float):
        """Efficiently scales the module."""
        if factor == 1.0: return
        
        if self.is_decomposed:
            self.s *= factor
        else:
            # Scale Up matrix (arbitrary choice)
            self.up = (self.up.float() * factor).to(self.up.dtype)

    def clone(self):
        """Deep copy of the module."""
        new_mod = LoRAModule(
            self.down.clone() if self.down is not None else None,
            self.up.clone() if self.up is not None else None,
            self.alpha
        )
        if self.is_decomposed:
            new_mod.u = self.u.clone() if self.u is not None else None
            new_mod.s = self.s.clone() if self.s is not None else None
            new_mod.vh = self.vh.clone() if self.vh is not None else None
            new_mod.is_decomposed = True
        return new_mod

class LoRAAssembly:
    """
    Structured container for a LoRA model.
    Replaces the flat state_dict for internal processing.
    """
    def __init__(self):
        self.modules: Dict[str, LoRAModule] = {} # Key: Normalized Base Name
        self.others: Dict[str, torch.Tensor] = {} # Embeddings, biases, etc.
        self.metadata: Dict[str, str] = {}
        self.key_map: Dict[str, Tuple[str, str]] = {} # BaseName -> (DownKey, UpKey) for reconstruction

    @staticmethod
    def from_state_dict(state_dict: Dict[str, torch.Tensor], metadata: Dict[str, str] = None) -> 'LoRAAssembly':
        assembly = LoRAAssembly()
        assembly.metadata = metadata or {}
        
        keys = list(state_dict.keys())
        groups = FormatHandler.group_keys(keys)
        
        processed_keys = set()
        
        for grp in groups:
            processed_keys.add(grp.down_key)
            processed_keys.add(grp.up_key)
            if grp.alpha_key: processed_keys.add(grp.alpha_key)
            
            ld = state_dict[grp.down_key]
            lu = state_dict[grp.up_key]
            
            alpha = None
            if grp.alpha_key and grp.alpha_key in state_dict:
                alpha = float(state_dict[grp.alpha_key].item())
            
            mod = LoRAModule(ld, lu, alpha)
            assembly.modules[grp.base_name] = mod
            
            # Store naming convention for export
            assembly.key_map[grp.base_name] = (grp.down_key, grp.up_key, grp.alpha_key)

        # Store remaining keys (biases, norms, embeddings)
        for k, v in state_dict.items():
            if k not in processed_keys:
                assembly.others[k] = v
                
        return assembly

    def to_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = {}
        
        # 1. Modules
        for base_name, mod in self.modules.items():
            if mod.is_decomposed:
                mod.compose()
            
            # Retrieve original key names or generate standard ones
            if base_name in self.key_map:
                d_key, u_key, a_key = self.key_map[base_name]
                
                # FIX: If a_key was None (missing in source), generate a valid one
                if a_key is None:
                    if d_key.endswith("lora_down.weight"):
                        a_key = d_key.replace("lora_down.weight", "alpha")
                    elif d_key.endswith("down.weight"):
                        a_key = d_key.replace("down.weight", "alpha")
                    else:
                        # Fallback for weird keys
                        a_key = f"{d_key.rsplit('.', 1)[0]}.alpha"
            else:
                # Fallback naming (Kohya style)
                safe_name = FormatHandler.convert_to_kohya_key(base_name)
                d_key = f"{safe_name}.lora_down.weight"
                u_key = f"{safe_name}.lora_up.weight"
                a_key = f"{safe_name}.alpha"

            sd[d_key] = mod.down
            sd[u_key] = mod.up
            
            # Always save alpha
            sd[a_key] = torch.tensor(mod.alpha, dtype=mod.down.dtype)

        # 2. Others
        for k, v in self.others.items():
            sd[k] = v
            
        return sd

    def clone(self) -> 'LoRAAssembly':
        new_asm = LoRAAssembly()
        new_asm.metadata = copy.deepcopy(self.metadata)
        new_asm.key_map = copy.deepcopy(self.key_map)
        
        for k, v in self.modules.items():
            new_asm.modules[k] = v.clone()
            
        for k, v in self.others.items():
            new_asm.others[k] = v.clone()
            
        return new_asm