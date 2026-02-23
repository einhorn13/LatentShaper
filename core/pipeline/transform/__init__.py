from .extract import ExtractMixin
from .resize import ResizeMixin
from .morph import MorphMixin
from .utils import UtilsMixin

class TransformMixin(ExtractMixin, ResizeMixin, MorphMixin, UtilsMixin):
    pass