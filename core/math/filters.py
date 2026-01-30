# core/math/filters.py

import torch
import torch.fft
import torch.nn.functional as F
import hashlib

class MathFilters:
    """
    Weight modification filters.
    Optimized with Real-to-Complex FFT (rfft2) for 50% memory savings.
    """

    @staticmethod
    def _get_rfft_mask(rows: int, cols: int, device: torch.device) -> torch.Tensor:
        """Generates a radial frequency mask for rfft2 output."""
        # rfft2 returns frequencies:
        # Y-axis: standard fftfreq (0 to 0.5, -0.5 to 0)
        # X-axis: rfftfreq (0 to 0.5) - only positive frequencies
        
        fy = torch.fft.fftfreq(rows, device=device)
        fx = torch.fft.rfftfreq(cols, device=device)
        
        # Create grid (Y, X)
        y, x = torch.meshgrid(fy, fx, indexing='ij')
        
        # Euclidean distance from DC component (0,0)
        # Normalized frequency distance (0.0 to ~0.707)
        return torch.sqrt(y**2 + x**2)

    @staticmethod
    def fft_filter(tensor: torch.Tensor, cutoff_freq: float = 0.5) -> torch.Tensor:
        if cutoff_freq >= 1.0: return tensor
        
        # Sanitize input
        t_float = torch.nan_to_num(tensor.float())
        rows, cols = t_float.shape
        
        # Real-to-Complex FFT (Faster, less memory)
        fft = torch.fft.rfft2(t_float)
        
        # Generate mask directly for rfft layout (no fftshift needed for X)
        dist = MathFilters._get_rfft_mask(rows, cols, tensor.device)
        
        # Apply Low-Pass Filter
        # Note: cutoff_freq is normalized 0..1, dist is 0..0.5 (Nyquist)
        # We adjust cutoff to match distance scale
        mask = torch.where(dist <= (cutoff_freq * 0.5), 1.0, 0.0)
        
        fft_filtered = fft * mask
        result = torch.fft.irfft2(fft_filtered, s=(rows, cols))
        
        return result.to(tensor.dtype)

    @staticmethod
    def apply_band_stop(tensor: torch.Tensor, freq_start: float, freq_end: float) -> torch.Tensor:
        if freq_start > freq_end: freq_start, freq_end = freq_end, freq_start
        if freq_start >= 1.0: return tensor
        
        t_float = torch.nan_to_num(tensor.float())
        rows, cols = t_float.shape
        
        fft = torch.fft.rfft2(t_float)
        dist = MathFilters._get_rfft_mask(rows, cols, tensor.device)
        
        # Band Stop Logic
        # Scale inputs (0..1) to Nyquist (0..0.5)
        fs, fe = freq_start * 0.5, freq_end * 0.5
        
        mask = torch.ones_like(dist)
        mask_band = (dist >= fs) & (dist <= fe)
        mask[mask_band] = 0.0
        
        fft_filtered = fft * mask
        result = torch.fft.irfft2(fft_filtered, s=(rows, cols))
        
        return result.to(tensor.dtype)

    @staticmethod
    def apply_dare(tensor, drop_rate):
        if drop_rate <= 0: return tensor
        if drop_rate >= 1.0: return torch.zeros_like(tensor)
        
        keep_prob = 1.0 - drop_rate
        # Create mask on same device
        mask = torch.bernoulli(torch.full_like(tensor, keep_prob))
        
        # Scale to maintain expected magnitude
        return tensor * mask * (1.0 / keep_prob)

    @staticmethod
    def apply_spectral_gate(tensor, threshold, is_adaptive=False):
        if threshold <= 0: return tensor
        t_float = torch.nan_to_num(tensor.float())
        
        try:
            U, S, V = torch.svd_lowrank(t_float, q=min(256, min(t_float.shape)), niter=2)
        except:
            return tensor # Fallback if SVD fails
        
        if is_adaptive:
            if threshold >= 1.0: return tensor
            total_energy = torch.sum(S**2)
            # Avoid div by zero
            if total_energy < 1e-9: return tensor
            
            sorted_S, _ = torch.sort(S, descending=True)
            cumulative = torch.cumsum(sorted_S**2, dim=0)
            target = total_energy * threshold
            
            mask = cumulative >= target
            if mask.any():
                cutoff_idx = torch.argmax(mask.int()).item() + 1
                S[cutoff_idx:] = 0.0
        else:
            S = torch.where(S < threshold, torch.tensor(0.0, device=S.device), S)
            
        return (U @ torch.diag(S) @ V.T).to(tensor.dtype)

    @staticmethod
    def apply_vector_eraser(tensor, start_idx, end_idx):
        if start_idx >= end_idx: return tensor
        t_float = torch.nan_to_num(tensor.float())
        min_dim = min(t_float.shape)
        q = min(256, min_dim)
        
        try:
            U, S, V = torch.svd_lowrank(t_float, q=q, niter=2)
            valid_start = max(0, start_idx)
            valid_end = min(len(S), end_idx)
            if valid_start < valid_end:
                S[valid_start:valid_end] = 0.0
            return (U @ torch.diag(S) @ V.T).to(tensor.dtype)
        except:
            return tensor

    @staticmethod
    def apply_homeostatic_scaling(tensor: torch.Tensor, target_magnitude: float = 0.01) -> torch.Tensor:
        if target_magnitude <= 0: return tensor
        t_float = tensor.float()
        current_mag = torch.mean(torch.abs(t_float))
        if current_mag < 1e-9: return tensor
        if current_mag > target_magnitude:
            scale = target_magnitude / current_mag
            return (t_float * scale).to(tensor.dtype)
        return tensor

    @staticmethod
    def apply_eigen_temperature(tensor: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature == 1.0: return tensor
        t_float = torch.nan_to_num(tensor.float())
        try:
            U, S, V = torch.svd_lowrank(t_float, q=min(256, min(t_float.shape)), niter=2)
            S_new = torch.pow(S + 1e-8, temperature)
            return (U @ torch.diag(S_new) @ V.T).to(tensor.dtype)
        except:
            return tensor

    @staticmethod
    def get_random_projection(tensor, seed_key, target_dim=256):
        seed = int(hashlib.sha256(seed_key.encode('utf-8')).hexdigest(), 16) % (2**32)
        torch.manual_seed(seed)
        flat = tensor.view(-1).float()
        if flat.shape[0] < target_dim: 
            return F.pad(flat, (0, target_dim - flat.shape[0]))
        indices = torch.randperm(flat.shape[0])[:target_dim]
        return flat[indices]