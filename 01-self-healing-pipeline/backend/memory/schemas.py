from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class EpisodicErrorRecord(BaseModel):
    id: int | None = None
    created_at: datetime | None = None
    document_type: str
    error_type: str
    principle: str
    context: dict[str, Any]
    resolution: str | None = None


class SimilarErrorMatch(BaseModel):
    record: EpisodicErrorRecord
    similarity: float = Field(..., ge=0.0, le=1.0)
