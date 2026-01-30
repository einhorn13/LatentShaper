# core/math/linalg.py

import torch
import torch.nn.functional as F
from typing import Tuple

class MathLinalg:
    """Linear Algebra operations: SVD, QR, Spectrum, Projections."""

    @staticmethod
    def get_spectrum_fast(ld: torch.Tensor, lu: torch.Tensor, scale: float = 1.0) -> Tuple[torch.Tensor, float]:
        """
        Computes Singular Values via QR decomposition.
        Robust against NaNs.
        """
        ld_f = torch.nan_to_num(ld.float())
        lu_f = torch.nan_to_num(lu.float())
        
        def get_R(res):
            if isinstance(res, (tuple, list)) and len(res) >= 2:
                return res[1]
            if hasattr(res, 'R'):
                return res.R
            return res

        try:
            res_u = torch.linalg.qr(lu_f, mode='r')
            R_u = get_R(res_u)
            
            res_d = torch.linalg.qr(ld_f.mT, mode='r')
            R_d = get_R(res_d)
            
            core = R_u @ R_d.mT
            
            if scale != 1.0:
                core.mul_(scale)
                
            S = torch.linalg.svdvals(core)
            energy = torch.sqrt(torch.sum(S**2))
            return S, energy.item()
        except Exception:
            # Fallback for degenerate matrices
            return torch.zeros(1), 0.0

    @staticmethod
    def svd_decomposition(delta_w, rank, auto_rank_threshold=0.0, clamp_threshold=1e-6):
        orig_dtype = delta_w.dtype
        # CRITICAL: Ensure no NaNs or Infs enter SVD
        d_float = torch.nan_to_num(delta_w.float(), nan=0.0, posinf=0.0, neginf=0.0)
        
        # Hard clamp to remove noise floor before SVD
        if clamp_threshold > 0:
            mask = torch.abs(d_float) >= clamp_threshold
            d_float = d_float * mask
        
        min_dim = min(d_float.shape)
        request_q = min(256 if auto_rank_threshold > 0 else rank + 16, min_dim)
        
        try:
            # Lowrank SVD is faster but can fail on edge cases
            U, S, V = torch.svd_lowrank(d_float, q=request_q, niter=4)
        except Exception:
            try:
                # Fallback to full SVD (slower but more robust)
                U, S, V = torch.linalg.svd(d_float, full_matrices=False)
            except Exception as e:
                print(f"SVD Failed: {e}")
                # Emergency fallback: return zeros
                return (torch.zeros((rank, d_float.shape[1]), dtype=orig_dtype), 
                        torch.zeros((d_float.shape[0], rank), dtype=orig_dtype), 
                        0)
        
        final_rank = rank
        if auto_rank_threshold > 0:
            total_energy = torch.sum(S)
            if total_energy > 1e-9:
                cumulative = torch.cumsum(S, dim=0)
                mask = cumulative >= (auto_rank_threshold * total_energy)
                if mask.any():
                    calc_rank = torch.argmax(mask.int()).item() + 1
                    final_rank = max(min(calc_rank, rank), 4)
        
        # Effective rank check (ignore singular values near zero)
        effective_rank = (S > 1e-5).sum().item()
        final_rank = max(min(final_rank, effective_rank), 1)
        final_rank = min(final_rank, U.shape[1])
        
        U = U[:, :final_rank]
        S = S[:final_rank]
        V = V[:, :final_rank]
        Vh = V.T 
        
        sqrt_S = torch.diag(torch.sqrt(S))
        Down = (sqrt_S @ Vh).to(orig_dtype)
        Up = (U @ sqrt_S).to(orig_dtype)
        
        return Down, Up, final_rank

    @staticmethod
    def resize_lora(ld, lu, new_rank, auto_rank_threshold=0.0):
        delta = lu.float() @ ld.float()
        return MathLinalg.svd_decomposition(delta, new_rank, auto_rank_threshold, clamp_threshold=1e-8)

    @staticmethod
    def get_spectrum(delta_w: torch.Tensor, rank_hint: int = None) -> torch.Tensor:
        d_float = torch.nan_to_num(delta_w.float())
        min_dim = min(d_float.shape)
        q = min(rank_hint, min_dim) if rank_hint else min(512, min_dim)
        try:
            _, S, _ = torch.svd_lowrank(d_float, q=q, niter=2)
            return S.cpu()
        except:
            return torch.zeros(q)

    @staticmethod
    def orthogonalize_rows_against_vector(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        m_f = torch.nan_to_num(matrix.float())
        v_f = torch.nan_to_num(vector.float())
        
        if m_f.shape[1] != v_f.shape[0]:
            return matrix 
            
        v_norm_sq = torch.dot(v_f, v_f)
        if v_norm_sq < 1e-9:
            return matrix
            
        dots = m_f @ v_f 
        scalars = dots / v_norm_sq
        projections = torch.outer(scalars, v_f)
        
        result = m_f - projections
        return result.to(matrix.dtype)