# core/math/__init__.py

from .common import MathCommon
from .linalg import MathLinalg
from .stats import MathStats
from .filters import MathFilters
from .merging import MathMerging

class MathKernel(MathCommon, MathLinalg, MathStats, MathFilters, MathMerging):
    """
    Unified MathKernel Facade.
    Inherits from sub-modules to provide a single access point:
    MathKernel.svd_decomposition(), MathKernel.fft_filter(), etc.
    """
    pass