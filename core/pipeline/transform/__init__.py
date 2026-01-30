# core/pipeline/transform/__init__.py
from .extract import ExtractMixin
from .resize import ResizeMixin
from .morph import MorphMixin
from .bridge import BridgeMixin
from .utils import UtilsMixin

class TransformMixin(ExtractMixin, ResizeMixin, MorphMixin, BridgeMixin, UtilsMixin):
    pass