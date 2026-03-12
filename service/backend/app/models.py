"""Pydantic models for API request/response schemas."""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    width: int
    height: int
    file_size_bytes: int
    format: str
    histogram: Dict[str, List[int]]
    dynamic_range_ev: float
    mean_brightness: float
    median_brightness: float
    clipping_percent: float
    min_luminance_linear: float = 0.0
    mean_luminance_linear: float = 0.0
    peak_luminance_linear: float = 0.0
    contrast_ratio: float = 0.0


class GenerateRequest(BaseModel):
    max_res: int = Field(default=4096, ge=256, le=8192, description="Max processing resolution")
    img_scale: float = Field(default=1.0, ge=0.1, le=5.0, description="Input brightness scale")
    proc_scale: float = Field(default=1.0, ge=0.25, le=2.0, description="Processing scale factor")


class ProgressEvent(BaseModel):
    stage: str
    progress: float
    message: str
    queue_position: int = 0


class HdrAnalysis(BaseModel):
    dynamic_range_ev: float
    contrast_ratio: float
    min_luminance: float = 0.0
    peak_luminance: float
    mean_luminance: float
    luminance_percentiles: Dict[str, float]
    hdr_histogram: dict


class ResultResponse(BaseModel):
    job_id: str
    download_url: str
    analysis: HdrAnalysis
    processing_time_seconds: float


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
