"""FastAPI application for IntrinsicHDR web service."""

import asyncio
import json
import logging
import os
import struct
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from .analysis import analyze_sdr
from .models import (ErrorResponse, GenerateRequest,
                     ResultResponse, UploadResponse)
from .queue import JobQueue

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAX_MEGAPIXELS = int(os.environ.get("MAX_MEGAPIXELS", "50"))
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_HOURS", "24")) * 3600


@dataclass
class JobStatus:
    stage: str = "pending"
    progress: float = 0.0
    message: str = ""
    error: str = ""
    result_ready: bool = False
    processing_time: float = 0.0
    queue_position: int = 0
    enqueued_at: float = 0.0
    # In-memory data
    input_bytes: Optional[bytes] = None
    filename: str = ""
    input_analysis: Optional[dict] = None
    hdr_result: Optional[np.ndarray] = None
    hdr_analysis: Optional[dict] = None
    created_at: float = field(default_factory=time.time)


def _cleanup_old_jobs(app: FastAPI, ttl: int):
    """Background thread that removes expired jobs from memory."""
    stop_event = app.state._cleanup_stop
    while not stop_event.wait(timeout=600):  # check every 10 min
        now = time.time()
        to_remove = []
        for job_id, job in list(app.state.jobs.items()):
            if job.stage in ("complete", "error", "cancelled") and (now - job.created_at) > ttl:
                to_remove.append(job_id)
        for job_id in to_remove:
            app.state.jobs.pop(job_id, None)
        if to_remove:
            logger.info("Cleaned up %d expired job(s) from memory", len(to_remove))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.jobs: Dict[str, JobStatus] = {}

    # Load IntrinsicHDR pipeline
    from .inference import IntrinsicHDRPipeline
    logger.info("Loading IntrinsicHDR pipeline...")

    try:
        app.state.pipeline = IntrinsicHDRPipeline()
    except Exception as e:
        logger.warning("Could not load IntrinsicHDR pipeline: %s", e)
        app.state.pipeline = None

    # Start job queue worker
    job_queue = JobQueue(app)
    job_queue.start()
    app.state.job_queue = job_queue

    # Start memory cleanup thread
    app.state._cleanup_stop = threading.Event()
    cleanup_thread = threading.Thread(
        target=_cleanup_old_jobs, args=(app, JOB_TTL_SECONDS),
        daemon=True, name="job-cleanup"
    )
    cleanup_thread.start()

    logger.info("Pipeline loaded. Service ready.")
    yield

    # Shutdown
    job_queue.stop()
    app.state._cleanup_stop.set()
    if app.state.pipeline:
        app.state.pipeline.close()


app = FastAPI(title="IntrinsicHDR Service", lifespan=lifespan)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "pipeline": app.state.pipeline is not None,
        "queue_size": app.state.job_queue.size,
    }


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(400, "Empty file")

    filename = file.filename or "image.png"

    # Analyze SDR image
    try:
        info = analyze_sdr(img_bytes, len(img_bytes), filename)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Check resolution limit
    megapixels = (info["width"] * info["height"]) / 1_000_000
    if megapixels > MAX_MEGAPIXELS:
        raise HTTPException(
            413,
            f"Image too large: {megapixels:.1f} MP (max {MAX_MEGAPIXELS} MP). "
            f"Please resize to {MAX_MEGAPIXELS} megapixels or less.",
        )

    job_id = uuid.uuid4().hex[:12]
    app.state.jobs[job_id] = JobStatus(
        input_bytes=img_bytes,
        filename=filename,
        input_analysis=info,
    )

    return UploadResponse(job_id=job_id, filename=filename, **info)


@app.post("/api/generate/{job_id}", status_code=202)
async def generate(job_id: str, req: GenerateRequest):
    if job_id not in app.state.jobs:
        raise HTTPException(404, "Job not found")

    job = app.state.jobs[job_id]
    if job.input_bytes is None:
        raise HTTPException(404, "Input image not found")

    if job.stage not in ("pending", "complete", "error", "cancelled"):
        raise HTTPException(409, "Job is already queued or processing")

    # Check pipeline availability
    if app.state.pipeline is None:
        raise HTTPException(503, "IntrinsicHDR pipeline not available")

    # Reset job status
    job.progress = 0.0
    job.error = ""
    job.result_ready = False
    job.hdr_result = None
    job.hdr_analysis = None

    try:
        position = app.state.job_queue.enqueue(job_id, req)
    except ValueError:
        raise HTTPException(503, "Queue full. Please try again later.")

    return {"job_id": job_id, "status": "queued", "queue_position": position}


@app.post("/api/cancel/{job_id}")
async def cancel(job_id: str):
    if job_id not in app.state.jobs:
        raise HTTPException(404, "Job not found")

    if app.state.job_queue.cancel(job_id):
        return {"job_id": job_id, "status": "cancelled"}
    else:
        raise HTTPException(409, "Job is not queued or processing")


@app.get("/api/status/{job_id}")
async def status_sse(job_id: str):
    if job_id not in app.state.jobs:
        raise HTTPException(404, "Job not found")

    async def event_stream():
        job = app.state.jobs[job_id]
        last_sent = None
        while True:
            current = {
                "stage": job.stage,
                "progress": round(job.progress, 3),
                "message": job.message,
                "queue_position": job.queue_position,
            }

            current_key = (current["stage"], current["progress"], current["queue_position"])
            if current_key != last_sent:
                yield f"data: {json.dumps(current)}\n\n"
                last_sent = current_key

            if job.stage in ("complete", "error", "cancelled"):
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/status-poll/{job_id}")
async def status_poll(job_id: str):
    if job_id not in app.state.jobs:
        raise HTTPException(404, "Job not found")
    job = app.state.jobs[job_id]
    return {
        "stage": job.stage,
        "progress": round(job.progress, 3),
        "message": job.message,
        "queue_position": job.queue_position,
    }


@app.get("/api/result/{job_id}", response_model=ResultResponse)
async def result(job_id: str):
    if job_id not in app.state.jobs:
        raise HTTPException(404, "Job not found")

    job = app.state.jobs[job_id]
    if not job.result_ready:
        raise HTTPException(409, "Result not ready yet")

    return ResultResponse(
        job_id=job_id,
        download_url=f"/api/download/{job_id}",
        analysis=job.hdr_analysis or {},
        processing_time_seconds=round(job.processing_time, 2),
    )


@app.get("/api/hdr-raw/{job_id}")
async def hdr_raw(job_id: str, max_dim: int = Query(default=1024, ge=64, le=4096)):
    """Serve downscaled HDR data as raw float32 binary for client-side tone mapping.
    Format: uint32 width, uint32 height, then float32 RGB pixels (row-major)."""
    if job_id not in app.state.jobs:
        raise HTTPException(404, "Job not found")

    job = app.state.jobs[job_id]
    if job.hdr_result is None:
        raise HTTPException(404, "HDR result not available")

    hdr_img = job.hdr_result
    h, w = hdr_img.shape[:2]

    # Downscale if needed
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        hdr_img = cv2.resize(hdr_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    header = struct.pack('<II', w, h)
    pixel_data = hdr_img.astype(np.float32).tobytes()
    return Response(
        content=header + pixel_data,
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    if job_id not in app.state.jobs:
        raise HTTPException(404, "Job not found")

    job = app.state.jobs[job_id]
    if job.hdr_result is None:
        raise HTTPException(404, "HDR result not available")

    # Write EXR to a temp file, serve it, then delete
    tmp = tempfile.NamedTemporaryFile(suffix=".exr", delete=False)
    tmp.close()
    hdr_bgr = cv2.cvtColor(job.hdr_result.astype(np.float32), cv2.COLOR_RGB2BGR)
    cv2.imwrite(tmp.name, hdr_bgr)

    original_name = job.filename
    if '.' in original_name:
        original_name = original_name.rsplit('.', 1)[0]

    return FileResponse(
        tmp.name,
        media_type="application/octet-stream",
        filename=f"{original_name}_hdr.exr",
        background=BackgroundTask(lambda: os.unlink(tmp.name)),
    )


# Serve frontend static files
_static_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "static")
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
