"""Schemas for POST /api/snapshot/generate."""

from typing import Literal

from pydantic import BaseModel

from app.schemas.cv import CV


class Break(BaseModel):
    duration_years: float
    activities: list[str] = []


class SnapshotRequest(BaseModel):
    cv: CV
    break_: Break

    model_config = {"populate_by_name": True}

    # accept JSON key "break" (a Python keyword) via alias
    def __init__(self, **data):  # noqa: D401
        if "break" in data and "break_" not in data:
            data["break_"] = data.pop("break")
        super().__init__(**data)


class ProfessionalSkill(BaseModel):
    skill: str
    skill_id: str | None = None
    source: Literal["experience"] = "experience"
    evidence: str | None = None
    evidence_type: Literal["literal"] = "literal"


class ReframedSkill(BaseModel):
    skill: str
    skill_id: str
    reframed_label: str | None = None
    source: Literal["break"] = "break"
    from_activity: str | None = None


class PreviousOccupation(BaseModel):
    role: str
    confidence: float | None = None
    method: Literal["cv_title"] = "cv_title"
    esco_code: str | None = None
    esco_title: str | None = None
    comparison_score: float | None = None
    comparison_method: str | None = None


class RecommendedRole(BaseModel):
    role: str
    role_id: str
    masco_code: str
    esco_code: str | None = None
    similarity: float
    method: Literal["exact_title", "embedding"]


class SnapshotResponse(BaseModel):
    reference_version: str | None = None
    warnings: list[str] = []
    professional_skills: list[ProfessionalSkill] = []
    reframed_skills: list[ReframedSkill] = []
    previous_occupation: PreviousOccupation | None = None
    recommended_roles: list[RecommendedRole] = []
