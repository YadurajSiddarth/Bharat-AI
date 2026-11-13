from typing import Literal, Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    max_new_tokens: int = Field(128, ge=1, le=1024)


class StageInfo(BaseModel):
    name: str
    status: Literal["waiting", "processing", "completed", "failed"]
    duration_ms: Optional[float] = None  # None = not measurable
    measurement: Literal["measured", "inferred", "not_separable"]
    note: Optional[str] = None


class GenerateResponse(BaseModel):
    response: str
    input_tokens: int
    generated_tokens: int
    inference_seconds: float
    total_seconds: float
    tokens_per_second: Optional[float]
    stages: list[StageInfo]
    model_name: str
    device: str
    settings: dict


class MemoryInfo(BaseModel):
    allocated_mb: Optional[float] = None
    total_mb: Optional[float] = None


class ModelStatus(BaseModel):
    state: Literal["unloaded", "loading", "loaded", "failed"]
    model_name: str
    device: str
    dtype: str
    load_seconds: Optional[float] = None
    error: Optional[str] = None
    memory: MemoryInfo
    gpu_utilization_percent: Optional[float] = None  # None = Unavailable