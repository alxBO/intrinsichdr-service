"""In-process FIFO job queue with a single worker thread for GPU inference."""

import collections
import logging
import threading
import time
from dataclasses import dataclass

from .models import GenerateRequest

logger = logging.getLogger(__name__)

MAX_QUEUE_SIZE = 50
QUEUE_TIMEOUT_SECONDS = 1800  # 30 min max wait in queue


def _queue_message(position: int) -> str:
    """Human-readable queue message. position is 1-based."""
    if position == 1:
        return "Pending (next)"
    return f"Pending (position {position}, {position - 1} job{'s' if position > 2 else ''} ahead)"


@dataclass
class QueueEntry:
    job_id: str
    request: GenerateRequest


class JobQueue:
    """FIFO queue consumed by a single worker thread (GPU serialization)."""

    def __init__(self, app):
        self.app = app
        self._queue: collections.deque[QueueEntry] = collections.deque()
        self._work_available = threading.Event()
        self._shutdown = threading.Event()
        self._lock = threading.Lock()  # protects _queue mutations
        self._worker: threading.Thread | None = None

    def start(self):
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="job-queue-worker")
        self._worker.start()
        logger.info("Job queue worker started.")

    def stop(self):
        self._shutdown.set()
        self._work_available.set()
        if self._worker:
            self._worker.join(timeout=10)

    def enqueue(self, job_id: str, request: GenerateRequest) -> int:
        """Add job to queue. Returns queue position (1-based).
        Raises ValueError if queue is full."""
        with self._lock:
            if len(self._queue) >= MAX_QUEUE_SIZE:
                raise ValueError("Queue full")
            self._queue.append(QueueEntry(job_id=job_id, request=request))
            position = len(self._queue)

        job = self.app.state.jobs[job_id]
        job.stage = "queued"
        job.enqueued_at = time.time()
        job.queue_position = position
        job.message = _queue_message(position)

        self._update_positions()
        self._work_available.set()
        return position

    def cancel(self, job_id: str) -> bool:
        """Remove job from queue or flag running job for cancellation.
        Returns True if the job was found and acted on."""
        # Try to remove from queue first
        with self._lock:
            for i, entry in enumerate(self._queue):
                if entry.job_id == job_id:
                    del self._queue[i]
                    self._update_positions_unlocked()
                    job = self.app.state.jobs.get(job_id)
                    if job:
                        job.stage = "cancelled"
                        job.message = "Cancelled"
                        job.queue_position = 0
                    return True

        # Not in queue — maybe currently processing
        job = self.app.state.jobs.get(job_id)
        if job and job.stage not in ("pending", "complete", "error", "cancelled"):
            job.stage = "cancelled"
            job.message = "Cancelling..."

            return True

        return False

    @property
    def size(self) -> int:
        return len(self._queue)

    # --- internal ---

    def _update_positions(self):
        with self._lock:
            self._update_positions_unlocked()

    def _update_positions_unlocked(self):
        """Recompute queue_position for all queued jobs. Must hold self._lock."""
        for i, entry in enumerate(self._queue):
            job = self.app.state.jobs.get(entry.job_id)
            if job and job.stage == "queued":
                job.queue_position = i + 1
                job.message = _queue_message(i + 1)

    def _worker_loop(self):
        while not self._shutdown.is_set():
            try:
                self._work_available.wait(timeout=5)
                self._work_available.clear()

                while not self._shutdown.is_set():
                    with self._lock:
                        if not self._queue:
                            break
                        entry = self._queue.popleft()
                    self._update_positions()

                    job = self.app.state.jobs.get(entry.job_id)
                    if not job:
                        continue
                    if job.stage == "cancelled":
                        continue

                    # Queue timeout check
                    if time.time() - job.enqueued_at > QUEUE_TIMEOUT_SECONDS:
                        job.stage = "error"
                        job.error = "Timeout in queue"
                        job.message = "Timeout in queue"
                        continue

                    job.queue_position = 0
                    self._run_inference(entry.job_id, entry.request, job)
            except Exception as e:
                logger.exception("Worker loop error: %s", e)

    def _run_inference(self, job_id: str, req: GenerateRequest, job):
        import gc
        from .analysis import analyze_hdr

        pipeline = self.app.state.pipeline
        try:
            start_time = time.time()

            img_bytes = job.input_bytes
            if not img_bytes:
                raise FileNotFoundError("Input image not found")

            def progress_cb(stage, progress, message):
                # Check cancellation at each stage transition
                if job.stage == "cancelled":
                    raise InterruptedError("Job cancelled")
                job.stage = stage
                job.progress = progress
                job.message = message

            hdr_out = pipeline.run(
                img_bytes=img_bytes,
                progress_cb=progress_cb,
                max_res=req.max_res,
                img_scale=req.img_scale,
                proc_scale=req.proc_scale,
            )

            # Check cancellation before saving
            if job.stage == "cancelled":
                return

            progress_cb("analysis", 0.95, "Analyzing HDR output...")
            hdr_analysis = analyze_hdr(hdr_out)

            elapsed = time.time() - start_time

            # Store results in memory
            job.hdr_result = hdr_out
            job.hdr_analysis = hdr_analysis

            job.stage = "complete"
            job.progress = 1.0
            job.message = "Done"
            job.result_ready = True
            job.processing_time = elapsed

        except InterruptedError:
            logger.info("Job %s cancelled during inference.", job_id)
            job.stage = "cancelled"
            job.message = "Cancelled"
        except Exception as e:
            logger.exception("Inference failed for job %s", job_id)
            job.stage = "error"
            job.progress = 0.0
            job.message = str(e)
            job.error = str(e)
        finally:
            gc.collect()
            pipeline._clear_device_cache()
