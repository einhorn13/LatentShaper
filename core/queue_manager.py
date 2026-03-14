
import threading
import queue
import atexit
import hashlib
import traceback
import time
from typing import Dict, Optional, List
from core.structs import BaseJob, JobStatus
from core.logger import Logger

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
        self.queue = queue.Queue()
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

    def get_state_hash(self) -> str:
        with self._lock:
            state_str = "".join([f"{j.id}:{j.status.name}:{j.progress}" for j in self.jobs.values()])
            return hashlib.md5(state_str.encode()).hexdigest()

    def submit_job(self, job: BaseJob) -> str:
        with self._lock:
            self.jobs[job.id] = job
        self.queue.put(job)
        return job.id

    def cancel_job(self, job_id: str):
        with self._lock:
            if self.current_job and self.current_job.id == job_id:
                self.current_job.cancel()
            elif job_id in self.jobs:
                job = self.jobs[job_id]
                if job.status == JobStatus.PENDING:
                    job.status = JobStatus.CANCELLED
                    job.message = "Cancelled in queue"

    def cancel_all_pending(self):
        with self._lock:
            for job in self.jobs.values():
                if job.status == JobStatus.PENDING:
                    job.status = JobStatus.CANCELLED
                    job.message = "Mass Cancelled"

    def clear_finished(self):
        with self._lock:
            finished_ids = [k for k, v in self.jobs.items() 
                           if v.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]]
            for fid in finished_ids:
                del self.jobs[fid]

    def get_all_jobs(self) -> List[BaseJob]:
        with self._lock:
            return sorted(self.jobs.values(), key=lambda x: x.created_at, reverse=True)

    def _worker_loop(self):
        while self.active:
            try:
                job = self.queue.get(timeout=1.0)
                if job is None: break
                
                if job.status == JobStatus.CANCELLED:
                    self.queue.task_done()
                    continue

                self.current_job = job
                job.started_at = time.time()
                job.status = JobStatus.RUNNING
                
                try:
                    job.run()
                    if job.status == JobStatus.RUNNING:
                        job.status = JobStatus.COMPLETED
                        job.progress = 1.0
                        job.message = "Finished"
                except Exception as e:
                    job.status = JobStatus.FAILED
                    # Capture full traceback for individual job logs
                    job.logs = traceback.format_exc()
                    job.message = f"Error: {str(e)}"
                    Logger.error(f"Job {job.id} failed: {e}")
                finally:
                    job.finished_at = time.time()
                    self.current_job = None
                    self.queue.task_done()
            except queue.Empty:
                continue