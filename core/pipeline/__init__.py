# core/pipeline/__init__.py

from ..services.extractor import ExtractorService
from ..services.resizer import ResizerService
from ..services.morpher import MorpherService
from ..configs import ExtractConfig, ResizeConfig, MorphConfig
from ..structs import ModelReference

class ZPipeline:
    """
    Legacy Facade. Redirects calls to new Services.
    """
    def __init__(self, device=None):
        self.device = device

    def extract_lora_gen(self, base, tuned, output, rank):
        cfg = ExtractConfig(rank=rank)
        svc = ExtractorService(self.device)
        tuned_list = [ModelReference(tuned)]
        return svc.process(cfg, ModelReference(base), tuned_list, output)

    def resize_lora_gen(self, lora, output, rank):
        cfg = ResizeConfig(rank=rank)
        svc = ResizerService(self.device)
        return svc.process(cfg, [ModelReference(lora)], output)

    def morph_lora_gen(self, lora, output, params):
        cfg = MorphConfig(
            eq_in=params.get("eq_in", 1.0),
            eq_mid=params.get("eq_mid", 1.0),
            eq_out=params.get("eq_out", 1.0),
            spectral_enabled=params.get("spectral_enabled", False),
            spectral_threshold=params.get("spectral_threshold", 0.0),
            spectral_remove_structure=params.get("spectral_remove_structure", False),
            dare_enabled=params.get("dare_enabled", False),
            dare_rate=params.get("dare_rate", 0.0)
        )
        svc = MorpherService(self.device)
        return svc.process(cfg, [ModelReference(lora)], output)