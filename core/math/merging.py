# core/math/merging.py

import torch
from typing import List

class MathMerging:
    """Algorithms for merging multiple LoRAs."""

    @staticmethod
    def merge_concat(tensors):
        down_list, up_list, total_rank = [], [], 0
        dtype = tensors[0][0].dtype
        for down, up, scale in tensors:
            down_list.append(down.to(dtype) * scale)
            up_list.append(up.to(dtype))
            total_rank += down.shape[0]
        return torch.cat(down_list, dim=0), torch.cat(up_list, dim=1), total_rank

    @staticmethod
    def ties_trim_and_elect_streaming(deltas: List[torch.Tensor], weights: List[float], density: float = 0.3) -> torch.Tensor:
        """
        Memory-efficient TIES implementation avoiding torch.stack.
        """
        # 1. Accumulate Weighted Sum to find Majority Sign
        sum_delta = torch.zeros_like(deltas[0], dtype=torch.float32)
        for d, w in zip(deltas, weights):
            sum_delta.add_(d.float(), alpha=w)
            
        majority_sign = torch.sign(sum_delta)
        del sum_delta 
        
        # 2. Accumulate values where sign matches majority
        final_delta = torch.zeros_like(deltas[0], dtype=torch.float32)
        for d, w in zip(deltas, weights):
            d_f = d.float()
            mask = (torch.sign(d_f) == majority_sign)
            final_delta.add_(d_f * mask, alpha=w)
            del d_f, mask
            
        if density < 1.0:
            # Calculate threshold for trimming
            # We want to keep the top 'density' fraction of values by magnitude
            flat = torch.abs(final_delta.view(-1))
            numel = flat.numel()
            
            # Optimization: Use sampling for large tensors to avoid OOM/RuntimeError in quantile
            if numel > 1_000_000:
                # Sample 1M elements to estimate the threshold
                # This is statistically sufficient for stable diffusion weights
                indices = torch.randint(0, numel, (1_000_000,), device=final_delta.device)
                sample = flat[indices]
                threshold = torch.quantile(sample, 1.0 - density)
            else:
                # Exact calculation for manageable sizes
                threshold = torch.quantile(flat, 1.0 - density)
            
            # Apply trim
            final_delta = torch.where(torch.abs(final_delta) < threshold, torch.tensor(0.0, device=final_delta.device), final_delta)
        
        return final_delta.to(deltas[0].dtype)

    @staticmethod
    def orthogonalize_update(base_delta: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
        b_flat, t_flat = base_delta.reshape(-1).float(), target_delta.reshape(-1).float()
        b_norm_sq = torch.dot(b_flat, b_flat)
        if b_norm_sq < 1e-12: return target_delta
        proj = (torch.dot(t_flat, b_flat) / b_norm_sq) * b_flat
        return (t_flat - proj).reshape_as(target_delta).to(target_delta.dtype)

    @staticmethod
    def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, dot_threshold: float = 0.9995) -> torch.Tensor:
        v0_f = torch.nan_to_num(v0.float())
        v1_f = torch.nan_to_num(v1.float())
        
        v0_norm = torch.norm(v0_f)
        v1_norm = torch.norm(v1_f)
        
        if v0_norm < 1e-9 or v1_norm < 1e-9:
            return (1 - t) * v0_f + t * v1_f

        v0_unit = v0_f / v0_norm
        v1_unit = v1_f / v1_norm
        
        dot = torch.sum(v0_unit * v1_unit)
        
        if dot < 0.0:
            v1_unit = -v1_unit
            dot = -dot
        
        dot = torch.clamp(dot, -1.0, 1.0)
        
        if torch.abs(dot) > dot_threshold:
            res = (1 - t) * v0_f + t * v1_f
            return res.to(v0.dtype)
            
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        
        if torch.abs(sin_theta) < 1e-9:
            return (1 - t) * v0_f + t * v1_f
        
        s0 = torch.sin((1 - t) * theta) / sin_theta
        s1 = torch.sin(t * theta) / sin_theta
        
        res_unit = s0 * v0_unit + s1 * v1_unit
        res_mag = (1 - t) * v0_norm + t * v1_norm
        
        result = res_unit * res_mag
        return result.to(v0.dtype)

    @staticmethod
    def median_merge(deltas: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes the element-wise median of a list of tensors.
        Robust against outliers.
        """
        if not deltas: return None
        if len(deltas) == 1: return deltas[0]
        
        # Stack: [N, Out, In]
        # Ensure float32 for precision during aggregation
        # WARNING: This consumes N * Size memory.
        try:
            stack = torch.stack([d.float() for d in deltas], dim=0)
            median_val, _ = torch.median(stack, dim=0)
            return median_val.to(deltas[0].dtype)
        except RuntimeError as e:
            # Fallback for OOM: Mean (Average) is much more memory efficient
            print(f"Median Merge OOM, falling back to Mean: {e}")
            sum_val = torch.zeros_like(deltas[0], dtype=torch.float32)
            for d in deltas:
                sum_val.add_(d.float())
            return (sum_val / len(deltas)).to(deltas[0].dtype)