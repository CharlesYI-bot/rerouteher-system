"""Schemas for POST /api/gap/compute."""

from typing import Literal

from pydantic import BaseModel, model_validator


class UserSkill(BaseModel):
    skill: str
    skill_id: str | None = None
    source: str | None = None
    evidence_type: str | None = None
    confirmed: bool = False


class GapRequest(BaseModel):
    skills: list[UserSkill] = []
    target_role: str | None = None
    target_role_id: str | None = None
    reference_version: str | None = None

    @model_validator(mode="after")
    def require_target(self):
        self.target_role_id = (self.target_role_id or "").strip() or None
        self.target_role = (self.target_role or "").strip() or None
        if not self.target_role_id and not self.target_role:
            raise ValueError("target_role_id or target_role is required")
        return self


class Gap(BaseModel):
    skill_id: str
    skill: str
    band: Literal["role", "ai_digital"]
    importance: float
    uplift: float


class GapResponse(BaseModel):
    readiness: float | None = None
    assessment_status: Literal["assessed", "not_assessed"] = "not_assessed"
    reason: str | None = None
    role_id: str | None = None
    masco_code: str | None = None
    esco_code: str | None = None
    reference_version: str | None = None
    required_skill_count: int = 0
    matched_skill_count: int = 0
    missing_skill_count: int = 0
    warnings: list[str] = []
    skills_have: list[str] = []
    gaps: list[Gap] = []
