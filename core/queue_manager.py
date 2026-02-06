# core/queue_manager.py

import threading
import queue
import traceback
import atexit
import sys
from typing import Dict, Optional, List
from .structs import BaseJob, JobStatus
from .logger import Logger

class QueueManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(QueueManager, cls).__new__(cls)
                cls._instance._init_queue()
        return cls._instance

    def _init_queue(self):
        self.queue: queue.Queue[Optional[BaseJob]] = queue.Queue()
        self.jobs: Dict[str, BaseJob] = {} 
        self.current_job: Optional[BaseJob] = None
        self.active: bool = True
        
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        atexit.register(self.shutdown)

    def shutdown(self):
        self.active = False
        self.queue.put(None)
        self.worker_thread.join(timeout=2.0)

    def submit_job(self, job: BaseJob) -> str:
        with self._lock:
            self.jobs[job.id] = job
        self.queue.put(job)
        return job.id

    def get_all_jobs(self) -> List[BaseJob]:
        with self._lock:
            return sorted(self.jobs.values(), key=lambda x: x.created_at, reverse=True)
            
    def clear_history(self):
        with self._lock:
            self.jobs = {k: v for k, v in self.jobs.items() if v.status in [JobStatus.PENDING, JobStatus.RUNNING]}

    def is_working(self) -> bool:
        return self.current_job is not None or not self.queue.empty()

    @property
    def pipeline(self):
        # Facade for backward compatibility if needed by GUI context
        from .services.analyzer import AnalyzerService
        return AnalyzerService()

    def _worker_loop(self):
        while self.active:
            try:
                job = self.queue.get(timeout=1.0)
                if job is None: break
                
                self.current_job = job
                job.status = JobStatus.RUNNING
                job.message = "Starting..."
                
                try:
                    job.run()
                except Exception as e:
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                    job.message = f"Error: {str(e)[:50]}"
                    Logger.error(f"Job failed: {e}")
                    traceback.print_exc()
                finally:
                    self.current_job = None
                    self.queue.task_done()
                    
            except queue.Empty:
                continue