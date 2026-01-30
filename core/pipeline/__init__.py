# core/pipeline/__init__.py

from .base import PipelineBase
from .operations import OperationsMixin

class ZPipeline(PipelineBase, OperationsMixin):
    """
    Unified Pipeline class that combines:
    - Base utilities (PipelineBase)
    - Heavy Operations (OperationsMixin)
    """
    pass