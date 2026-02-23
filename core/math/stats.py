
import torch
import numpy as np
from typing import Tuple, List

class MathStats:
    """Statistical analysis tools with safety guards."""

    @staticmethod
    def calculate_stats_estimated(ld: torch.Tensor, lu: torch.Tensor, scale: float = 1.0, sample_size: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimates statistics by sampling the matrix product.
        OPTIMIZED: Samples indices BEFORE casting to float to reduce memory bandwidth.
        """
        out_dim, rank = lu.shape
        in_dim = ld.shape[1]
        
        # Stride sampling directly on the device
        # Ensure we don't exceed dimensions
        step_u = max(1, out_dim // sample_size)
        step_d = max(1, in_dim // sample_size)
        
        # Slice raw tensors (BF16/FP16)
        # Using slicing [::step] is faster than index_select for strided access
        sub_lu = lu[0:out_dim:step_u]
        sub_ld = ld[:, 0:in_dim:step_d]
        
        # Now cast to float for the MatMul
        # This reconstruction is small (~1024x1024)
        delta = (sub_lu.float() @ sub_ld.float())
        
        if scale != 1.0: 
            delta.mul_(scale)
            
        mag = torch.mean(torch.abs(delta))
        
        mean = torch.mean(delta)
        diffs = delta - mean
        var = torch.mean(torch.square(diffs))
        m4 = torch.mean(torch.pow(diffs, 4))
        
        # Protection against zero variance
        kurt = m4 / (torch.square(var) + 1e-12)
        return kurt, mag

    @staticmethod
    def calculate_kurtosis(tensor: torch.Tensor) -> float:
        t = tensor.float()
        mean = torch.mean(t)
        std = torch.std(t)
        if std < 1e-12: return 0.0
        return (torch.mean((t - mean)**4) / (std**4 + 1e-12)).item()

    @staticmethod
    def find_knee_point(values: List[float]) -> int:
        if not values or len(values) < 2: return 1
        y = np.array(values)
        if np.all(y == 0): return 1
        x = np.arange(len(y))
        y_min, y_max = y.min(), y.max()
        if y_max == y_min: return len(values)
        y_norm = (y - y_min) / (y_max - y_min)
        x_norm = x / x.max()
        coords = np.vstack((x_norm, y_norm)).T
        line_vec = coords[-1] - coords[0]
        vec_from_first = coords - coords[0]
        scalar_prod = np.sum(vec_from_first * line_vec, axis=1)
        vec_proj = np.outer(scalar_prod, line_vec) / np.dot(line_vec, line_vec)
        dist = np.linalg.norm(vec_from_first - vec_proj, axis=1)
        return int(np.argmax(dist)) + 1

    @staticmethod
    def calculate_intrinsic_rank_from_spectrum(spectrum: List[float], threshold: float = 0.95) -> int:
        """
        Calculates rank based on Manifold Energy Capacity (PCA approach).
        Threshold 0.95 means keeping 95% of the information energy.
        """
        if not spectrum: return 1
        S = torch.tensor(spectrum, dtype=torch.float32)
        
        # Energy = squared singular values
        energy = S ** 2
        total_energy = torch.sum(energy)
        if total_energy == 0: return 1
        
        cumulative = torch.cumsum(energy, dim=0)
        target = total_energy * threshold
        
        # Find first index where cumulative energy >= target
        mask = cumulative >= target
        if mask.any():
            return torch.argmax(mask.int()).item() + 1
        return len(spectrum)