# core/jobs.py

from typing import List
from .structs import BaseJob, JobStatus, ModelReference, ModelSourceType
from .configs import ExtractConfig, ResizeConfig, MorphConfig, MergeConfig, UtilsConfig, CheckpointMergeConfig
from .services.extractor import ExtractorService
from .services.resizer import ResizerService
from .services.morpher import MorpherService
from .services.merger import MergerService
from .services.utils import UtilsService
from .workspace import WorkspaceManager

class ExtractJob(BaseJob):
    def __init__(self, config: ExtractConfig, base_path: str, tuned_paths: List[str], output_path: str):
        super().__init__(f"Extract {len(tuned_paths)} models")
        self.config = config
        self.base_ref = ModelReference(base_path)
        self.tuned_refs = [ModelReference(p) for p in tuned_paths]
        self.output_path = output_path
        self.service = ExtractorService()

    def run(self):
        for prog, msg in self.service.process(self.config, self.base_ref, self.tuned_refs, self.output_path):
            self.progress = prog
            self.message = msg
            if self._cancel_flag: break
        self.status = JobStatus.COMPLETED

class ResizeJob(BaseJob):
    def __init__(self, config: ResizeConfig, input_paths: List[str], output_path: str):
        super().__init__(f"Resize {len(input_paths)} models")
        self.config = config
        self.inputs = [ModelReference(p) for p in input_paths]
        self.output_path = output_path
        self.service = ResizerService()

    def run(self):
        for prog, msg in self.service.process(self.config, self.inputs, self.output_path):
            self.progress = prog
            self.message = msg
        self.status = JobStatus.COMPLETED

class MorphJob(BaseJob):
    def __init__(self, config: MorphConfig, input_paths: List[str], output_path: str):
        super().__init__(f"Morph {len(input_paths)} models")
        self.config = config
        self.inputs = [ModelReference(p) for p in input_paths]
        self.output_path = output_path
        self.service = MorpherService()

    def run(self):
        for prog, msg in self.service.process(self.config, self.inputs, self.output_path):
            self.progress = prog
            self.message = msg
        self.status = JobStatus.COMPLETED

class MergeJob(BaseJob):
    def __init__(self, config: MergeConfig, input_paths: List[str], output_path: str):
        super().__init__(f"Merge {len(input_paths)} models")
        self.config = config
        self.inputs = [ModelReference(p) for p in input_paths]
        self.output_path = output_path
        self.service = MergerService()

    def run(self):
        for prog, msg in self.service.merge_loras(self.config, self.inputs, self.output_path):
            self.progress = prog
            self.message = msg
        self.status = JobStatus.COMPLETED

class CheckpointMergeJob(BaseJob):
    def __init__(self, config: CheckpointMergeConfig, input_paths: List[str], output_path: str):
        super().__init__("Checkpoint Merge")
        self.config = config
        self.inputs = [ModelReference(p) for p in input_paths if p]
        self.output_path = output_path
        self.service = MergerService()

    def run(self):
        for prog, msg in self.service.merge_checkpoints(self.config, self.inputs, self.output_path):
            self.progress = prog
            self.message = msg
        self.status = JobStatus.COMPLETED

class UtilsJob(BaseJob):
    def __init__(self, config: UtilsConfig, input_paths: List[str], output_path: str):
        super().__init__("Utils Process")
        self.config = config
        self.inputs = [ModelReference(p) for p in input_paths]
        self.output_path = output_path
        self.service = UtilsService()

    def run(self):
        for prog, msg in self.service.process(self.config, self.inputs, self.output_path):
            self.progress = prog
            self.message = msg
        self.status = JobStatus.COMPLETED