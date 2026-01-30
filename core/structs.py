# core/structs.py

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import uuid

class JobStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()

class JobType(Enum):
    EXTRACT = auto()
    RESIZE = auto()
    MERGE = auto()
    MORPH = auto()
    ANALYSIS = auto()
    UTILS = auto() # New

@dataclass
class Job:
    """
    Represents a single unit of work in the queue.
    """
    job_type: JobType
    input_paths: List[str]
    output_path: str
    params: Dict[str, Any]
    
    description: str = "Task"
    
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: float = field(default_factory=time.time)
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = "Queued"
    result: Any = None
    error: Optional[str] = None

    @property
    def is_batch(self) -> bool:
        return len(self.input_paths) > 1