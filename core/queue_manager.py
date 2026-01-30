# core/queue_manager.py

import threading
import time
import queue
import traceback
import atexit
import signal
import sys
from typing import Dict, Optional, List
from .structs import Job, JobStatus, JobType
from .pipeline import ZPipeline
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
        self.queue = queue.Queue()
        self.jobs: Dict[str, Job] = {} 
        self.current_job: Optional[Job] = None
        self.active = True
        
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = ZPipeline(device=device)
        
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        atexit.register(self.shutdown)
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError:
            pass

    def _signal_handler(self, sig, frame):
        Logger.info(f"System signal {sig} received. Performing graceful shutdown...")
        self.shutdown()
        sys.exit(0)

    def shutdown(self):
        if not self.active: return
        self.active = False
        self.queue.put(None) 
        self.worker_thread.join(timeout=2.0)
        Logger.info("QueueManager: Offline.")

    def submit_job(self, job: Job) -> str:
        with self._lock:
            self.jobs[job.id] = job
        self.queue.put(job)
        return job.id

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self.jobs.get(job_id)

    def get_all_jobs(self) -> List[Job]:
        with self._lock:
            return sorted(self.jobs.values(), key=lambda x: x.created_at, reverse=True)

    def clear_history(self):
        with self._lock:
            self.jobs = {
                k: v for k, v in self.jobs.items() 
                if v.status in [JobStatus.PENDING, JobStatus.RUNNING]
            }

    def is_working(self) -> bool:
        return self.current_job is not None or not self.queue.empty()

    def cancel_job(self, job_id: str):
        with self._lock:
            job = self.jobs.get(job_id)
            if job and job.status == JobStatus.PENDING:
                job.status = JobStatus.CANCELLED
                job.message = "Cancelled"

    def _worker_loop(self):
        while self.active:
            try:
                job = self.queue.get(timeout=1.0)
                if job is None: break
                    
                if job.status == JobStatus.CANCELLED:
                    self.queue.task_done()
                    continue
                    
                self._process_job(job)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                Logger.error(f"Queue Worker internal error: {e}")

    def _process_job(self, job: Job):
        self.current_job = job
        job.status = JobStatus.RUNNING
        job.message = "Initializing..."
        job.progress = 0.0
        
        generator = None
        try:
            if job.job_type == JobType.EXTRACT:
                generator = self.pipeline.extract_lora_gen(
                    base_path=job.params['base_path'],
                    tuned_paths=job.input_paths,
                    output_target=job.output_path,
                    rank=job.params['rank'],
                    threshold=job.params.get('threshold', 0.0),
                    save_to_workspace=job.params.get('save_to_workspace', False)
                )
            
            elif job.job_type == JobType.RESIZE:
                generator = self.pipeline.resize_lora_gen(
                    lora_paths=job.input_paths,
                    output_target=job.output_path,
                    new_rank=job.params['rank'],
                    auto_rank_threshold=job.params.get('auto_rank_threshold', 0.0),
                    save_to_workspace=job.params.get('save_to_workspace', False)
                )
            
            elif job.job_type == JobType.MORPH:
                if job.params.get('is_bridge'):
                    generator = self.pipeline.convert_sdxl_to_turbo_gen(
                        sdxl_lora_path=job.input_paths[0],
                        output_name=job.output_path,
                        strength=job.params.get('strength', 0.5),
                        save_to_workspace=job.params.get('save_to_workspace', False)
                    )
                else:
                    generator = self.pipeline.morph_lora_gen(
                        lora_paths=job.input_paths,
                        output_target=job.output_path,
                        params=job.params,
                        save_to_workspace=job.params.get('save_to_workspace', False)
                    )
            
            elif job.job_type == JobType.MERGE:
                generator = self.pipeline.merge_lora_gen(
                    lora_paths=job.input_paths,
                    ratios=job.params['ratios'],
                    output_path=job.output_path,
                    target_rank=job.params['rank'],
                    algorithm=job.params['algorithm'],
                    global_strength=job.params.get('global_strength', 1.0),
                    auto_rank_threshold=job.params.get('auto_rank_threshold', 0.0),
                    pruning_threshold=job.params.get('pruning_threshold', 0.0),
                    ties_density=job.params.get('ties_density', 0.3),
                    save_to_workspace=job.params.get('save_to_workspace', False)
                )
            
            elif job.job_type == JobType.UTILS:
                generator = self.pipeline.process_utils_gen(
                    lora_paths=job.input_paths,
                    output_target=job.output_path,
                    params=job.params,
                    save_to_workspace=job.params.get('save_to_workspace', False)
                )

            if generator:
                for progress, message in generator:
                    if not self.active: break 
                    job.progress = progress
                    job.message = message
                
                if self.active:
                    job.status = JobStatus.COMPLETED
                    job.progress = 1.0
                    job.message = "Finished"
            else:
                job.status = JobStatus.FAILED
                job.error = "Unsupported Job Type"

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.message = f"Error: {str(e)[:50]}..."
            Logger.error(f"Job {job.id} failed: {e}")
            traceback.print_exc()
        finally:
            self.current_job = None