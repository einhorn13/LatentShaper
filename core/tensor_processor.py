# core/tensor_processor.py

import torch
from typing import Dict, Optional
from .math import MathKernel

class TensorProcessor:
    """
    Centralized logic for tensor modification.
    Used by both GUI (MorphMixin) and ComfyUI Nodes (LS_Filters, etc).
    """

    @staticmethod
    def apply_filters(
        delta: torch.Tensor, 
        params: Dict[str, any], 
        eq_factor: float = 1.0,
        b_idx: int = -1
    ) -> Optional[torch.Tensor]:
        """
        Applies a chain of filters to the delta matrix.
        Returns None if the block should be erased.
        """
        
        # 1. Eraser (Block Level)
        # Check if block index is in the erase set
        erase_set = params.get("erase_blocks_set")
        if erase_set and b_idx in erase_set:
            return None

        # 2. Semantic Eraser (Projection)
        # Handled by caller usually, but if vectors passed in params:
        concept_vectors = params.get("concept_vectors")
        if concept_vectors and delta.shape[1] == concept_vectors[0].shape[0]:
            for vec in concept_vectors:
                delta = MathKernel.orthogonalize_rows_against_vector(delta, vec)

        # 3. DARE (Structure Random Drop)
        if params.get("dare_enabled"):
            delta = MathKernel.apply_dare(delta, params.get("dare_rate", 0.1))

        # 4. Frequency Filters (FFT)
        fft_cutoff = params.get("fft_cutoff", 1.0)
        if fft_cutoff < 1.0:
            delta = MathKernel.fft_filter(delta, fft_cutoff)
            
        if params.get("band_stop_enabled"):
            delta = MathKernel.apply_band_stop(
                delta, 
                params.get("band_stop_start", 0.0), 
                params.get("band_stop_end", 0.0)
            )

        # 5. Spectral Gate (SVD-based filtering)
        if params.get("spectral_enabled"):
            delta = MathKernel.apply_spectral_gate(
                delta, 
                params.get("spectral_threshold", 0.1), 
                is_adaptive=params.get("spectral_adaptive", False)
            )

        # 6. Vector Eraser (Specific Rank Removal)
        e_start = int(params.get("eraser_start", 0))
        e_end = int(params.get("eraser_end", 0))
        if e_end > e_start and b_idx != -1:
            delta = MathKernel.apply_vector_eraser(delta, e_start, e_end)

        # 7. EQ Application
        if eq_factor != 1.0:
            delta *= eq_factor

        # 8. Safety & Normalization
        if params.get("homeostatic", False):
            delta = MathKernel.apply_homeostatic_scaling(delta, params.get("homeostatic_thr", 0.01))

        # 9. Clamping
        clamp_q = params.get("clamp_quantile", 1.0)
        if clamp_q < 1.0:
            delta = MathKernel.percentile_clamp(delta, clamp_q)

        return delta