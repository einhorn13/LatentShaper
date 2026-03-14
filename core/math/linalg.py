
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from ..logger import Logger

class MathLinalg:
    """
    Linear Algebra operations: SVD, QR, Spectrum.
    Enhanced for numerical stability and cross-device compatibility.
    """

    @staticmethod
    def get_spectrum_fast(ld: torch.Tensor, lu: torch.Tensor, scale: float = 1.0) -> Tuple[torch.Tensor, float]:
        """
        Computes Singular Values via QR decomposition.
        Fixed: Explicitly casts to Float32 to avoid "geqrf_cuda" not implemented for BFloat16.
        """
        # 1. Защита от NaN и приведение к Float32 (обязательно для linalg на CUDA)
        ld_safe = torch.nan_to_num(ld.float(), nan=0.0)
        lu_safe = torch.nan_to_num(lu.float(), nan=0.0)
        
        def get_R(res):
            if isinstance(res, (tuple, list)) and len(res) >= 2: return res[1]
            if hasattr(res, 'R'): return res.R
            return res

        try:
            # Выполняем QR разложение. Результат всегда в Float32.
            # Даже если входные тензоры на CUDA, QR требует Float32/Float64.
            
            # Разложение матрицы Up (Out_dim x Rank)
            res_u = torch.linalg.qr(lu_safe, mode='r')
            R_u = get_R(res_u)
            
            # Разложение матрицы Down (Rank x In_dim) -> подаем транспонированной
            res_d = torch.linalg.qr(ld_safe.mT, mode='r')
            R_d = get_R(res_d)
            
            # Матрица ядра (Core): Rank x Rank (очень маленькая)
            core = R_u @ R_d.mT
            
            if scale != 1.0:
                core.mul_(scale)
                
            # SVD вычисляется на маленькой квадратной матрице ядра
            S = torch.linalg.svdvals(core)
            energy = torch.sqrt(torch.sum(S**2))
            
            return S.detach().cpu(), energy.item()
            
        except Exception as e:
            Logger.error(f"Fast Spectrum failed: {e}")
            # Возвращаем заглушку, чтобы не прерывать пайплайн анализа
            return torch.zeros(1), 0.0

    @staticmethod
    def svd_decomposition(delta_w: torch.Tensor, rank: int, auto_rank_threshold: float = 0.0, clamp_threshold: float = 1e-6):
        """
        Decomposes weight matrix into LoRA Down/Up.
        Uses fallback drivers and logic for numerical stability.
        """
        orig_dtype = delta_w.dtype
        # Всегда вычисляем SVD в Float32 для стабильности
        d_float = torch.nan_to_num(delta_w.float(), nan=0.0, posinf=0.0, neginf=0.0)
        
        if clamp_threshold > 0:
            mask = torch.abs(d_float) >= clamp_threshold
            d_float = d_float * mask
        
        rows, cols = d_float.shape
        min_dim = min(rows, cols)
        
        # Решаем, какой драйвер использовать
        use_lowrank = min_dim > 512 and rank < (min_dim // 2)
        
        try:
            if use_lowrank:
                q = min(rank + 32, min_dim)
                U, S, V = torch.svd_lowrank(d_float, q=q, niter=4)
            else:
                # На CPU 'gesvd' стабильнее, на CUDA используется дефолтный драйвер
                driver = 'gesvd' if d_float.device.type == 'cpu' else None
                U, S, Vh = torch.linalg.svd(d_float, full_matrices=False, driver=driver)
                V = Vh.mT
        except Exception as e:
            Logger.warning(f"SVD Primary Driver failed: {e}. Falling back to default.")
            try:
                U, S, Vh = torch.linalg.svd(d_float, full_matrices=False)
                V = Vh.mT
            except Exception as final_e:
                Logger.error(f"Critical SVD Failure: {final_e}")
                return (torch.zeros((rank, cols), dtype=orig_dtype), 
                        torch.zeros((rows, rank), dtype=orig_dtype), 0)
        
        # Логика выбора эффективного ранга
        final_rank = rank
        if auto_rank_threshold > 0:
            total_energy = torch.sum(S)
            if total_energy > 1e-9:
                cumulative = torch.cumsum(S, dim=0)
                mask = cumulative >= (auto_rank_threshold * total_energy)
                if mask.any():
                    calc_rank = torch.argmax(mask.int()).item() + 1
                    final_rank = max(min(calc_rank, rank), 1)
        
        final_rank = min(final_rank, U.shape[1], S.shape[0])
        
        U = U[:, :final_rank]
        S_sliced = S[:final_rank]
        V = V[:, :final_rank]
        
        sqrt_S = torch.diag(torch.sqrt(S_sliced))
        Down = (sqrt_S @ V.mT).to(orig_dtype)
        Up = (U @ sqrt_S).to(orig_dtype)
        
        return Down, Up, final_rank

    @staticmethod
    def resize_lora(ld, lu, new_rank, auto_rank_threshold=0.0):
        # Перемножаем в Float32
        delta = lu.float() @ ld.float()
        return MathLinalg.svd_decomposition(delta, new_rank, auto_rank_threshold)

    @staticmethod
    def get_spectrum(delta_w: torch.Tensor, rank_hint: int = None) -> torch.Tensor:
        d_float = torch.nan_to_num(delta_w.float(), nan=0.0)
        min_dim = min(d_float.shape)
        q = min(rank_hint, min_dim) if rank_hint else min(512, min_dim)
        try:
            # svd_lowrank обычно работает в Float32 даже на GPU
            _, S, _ = torch.svd_lowrank(d_float, q=q, niter=2)
            return S.detach().cpu()
        except:
            return torch.zeros(q)