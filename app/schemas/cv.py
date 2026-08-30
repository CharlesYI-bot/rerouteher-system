"""Schemas for POST /api/cv/parse."""

from pydantic import BaseModel


class Experience(BaseModel):
    title: str | None = None
    organisation: str | None = None
    start: str | None = None
    end: str | None = None
    description: str | None = None


class CV(BaseModel):
    raw_text: str
    experiences: list[Experience] = []
    skill_mentions: list[str] = []
    warnings: list[str] = []


class CVParseResponse(BaseModel):
    cv: CV


class ErrorResponse(BaseModel):
    error: str
