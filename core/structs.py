
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
import time
import uuid
from abc import ABC, abstractmethod

class JobStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()

class ModelSourceType(Enum):
    DISK = auto()
    WORKSPACE = auto()

@dataclass
class ModelReference:
    path: str
    source_type: ModelSourceType = ModelSourceType.DISK

    @property
    def name(self) -> str:
        import os
        return os.path.basename(self.path)

class BaseJob(ABC):
    """
    Enhanced Job structure with timing, logs and parameter storage.
    """
    def __init__(self, description: str = "Task", config: Any = None):
        self.id: str = str(uuid.uuid4())[:8]
        self.created_at: float = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        
        self.status: JobStatus = JobStatus.PENDING
        self.progress: float = 0.0
        self.message: str = "Queued"
        self.description: str = description
        
        # Store configuration for "Recall Parameters"
        self.config_data: Dict[str, Any] = asdict(config) if config else {}
        self.logs: str = "" # Per-job stdout/stderr/traceback
        self._cancel_flag: bool = False

    @property
    def duration(self) -> float:
        if not self.started_at: return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at

    @abstractmethod
    def run(self):
        pass

    def cancel(self):
        self._cancel_flag = True
        self.status = JobStatus.CANCELLED
        self.message = "Cancelled by user"