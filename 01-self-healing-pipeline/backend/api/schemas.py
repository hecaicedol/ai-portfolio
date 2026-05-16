from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


DocumentType = Literal["invoice", "contract", "purchase_order", "receipt", "generic"]


class ProcessRequest(BaseModel):
    document_type: DocumentType = "generic"
    content: str = Field(..., min_length=1, description="Raw document text")
    metadata: dict[str, Any] = Field(default_factory=dict)


class PrincipleScore(BaseModel):
    principle: Literal["completeness", "accuracy", "consistency", "format"]
    score: float = Field(..., ge=0.0, le=1.0)
    feedback: str


class CriticReport(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=1.0)
    principles: list[PrincipleScore]
    passes: bool
    similar_past_errors: list[str] = Field(default_factory=list)


class PipelineEvent(BaseModel):
    type: Literal[
        "run_started", "agent_started", "agent_finished",
        "critic_report", "reflection_triggered", "run_completed", "run_failed",
    ]
    agent: str | None = None
    iteration: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)
    at: datetime = Field(default_factory=datetime.utcnow)


class ProcessResponse(BaseModel):
    success: bool
    iterations: int
    final_score: float
    extracted_data: dict[str, Any]
    critic_report: CriticReport
    errors_history: list[dict[str, Any]] = Field(default_factory=list)
