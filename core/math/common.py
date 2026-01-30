# core/math/common.py

import torch
from typing import Set

class MathCommon:
    """Basic utility functions for tensor manipulation."""

    @staticmethod
    def calculate_l2_norm(tensor: torch.Tensor) -> float:
        return torch.norm(tensor.float()).item()

    @staticmethod
    def percentile_clamp(tensor: torch.Tensor, percentile: float = 0.999) -> torch.Tensor:
        if percentile >= 1.0: return tensor
        t = tensor.float()
        if t.numel() > 1_000_000:
            flat = t.view(-1)
            indices = torch.randint(0, flat.size(0), (1_000_000,), device=t.device)
            sample = flat[indices]
            k = torch.quantile(torch.abs(sample), percentile)
        else:
            k = torch.quantile(torch.abs(t), percentile)
        return torch.clamp(t, min=-k, max=k).to(tensor.dtype)

    @staticmethod
    def rescale_alpha(tensor: torch.Tensor, current_alpha: float, current_rank: float) -> torch.Tensor:
        if current_rank <= 0: return tensor
        scale = current_alpha / current_rank
        if scale == 1.0: return tensor
        return (tensor.float() * scale).to(tensor.dtype)

    @staticmethod
    def match_distribution(delta: torch.Tensor, target_mean: float = 0.0, target_std: float = None) -> torch.Tensor:
        curr_mean = delta.mean()
        curr_std = delta.std()
        
        if curr_std < 1e-9: return delta
        
        normalized = (delta - curr_mean) / curr_std
        final_std = target_std if target_std is not None else curr_std
        
        result = normalized * final_std + target_mean
        return result.to(delta.dtype)

    @staticmethod
    def parse_block_string(block_str: str) -> Set[int]:
        """
        Parses a string like "1, 4-6, 10" into a set of integers {1, 4, 5, 6, 10}.
        Handles spaces and invalid formats gracefully.
        """
        blocks = set()
        if not block_str or not block_str.strip():
            return blocks
            
        parts = block_str.split(',')
        for part in parts:
            part = part.strip()
            if not part: continue
            
            try:
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    # Support both 4-6 and 6-4
                    if start > end: start, end = end, start
                    blocks.update(range(start, end + 1))
                else:
                    blocks.add(int(part))
            except ValueError:
                continue # Skip malformed parts
                
        return blocks