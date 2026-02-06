# core/structs.py

from enum import Enum, auto
from dataclasses import dataclass, field
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
    path: str # File path or Workspace Key
    source_type: ModelSourceType = ModelSourceType.DISK

    @property
    def name(self) -> str:
        import os
        return os.path.basename(self.path)

class BaseJob(ABC):
    """
    Abstract Command Pattern for Jobs.
    Encapsulates the execution logic.
    """
    def __init__(self, description: str = "Task"):
        self.id: str = str(uuid.uuid4())[:8]
        self.created_at: float = time.time()
        self.status: JobStatus = JobStatus.PENDING
        self.progress: float = 0.0
        self.message: str = "Queued"
        self.description: str = description
        self.result: Any = None
        self.error: Optional[str] = None
        self._cancel_flag: bool = False

    @abstractmethod
    def run(self):
        """
        Execute the job logic. Should yield (progress, message) tuples if possible,
        or update self.progress/self.message directly.
        """
        pass

    def cancel(self):
        self._cancel_flag = True
        self.status = JobStatus.CANCELLED
        self.message = "Cancelled"