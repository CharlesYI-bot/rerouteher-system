"""Synthetic regression cases; no real resumes, database or models."""

from types import SimpleNamespace as Row
from unittest.mock import AsyncMock

import numpy as np
import pytest

from app.config import Settings
from app.repositories import caregiving, roles, skills
from app.repositories.caregiving import ReframedRow
from app.repositories.roles import NearestRole
from app.schemas.cv import CV, Experience
from app.schemas.snapshot import Break, SnapshotRequest
from app.services.occupation_matcher import OccupationMatch
from app.services.skill_catalog import SkillCatalog
from app.services.snapshot import SnapshotService

pytestmark = pytest.mark.asyncio


class Embedder:
    def encode_one(self, text):
        return np.ones(384)


class Matcher:
    def __init__(self, score=0.9, method="tfidf_logreg"):
        self.calls = []
        self.score = score
        self.method = method

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [OccupationMatch("1330.5", "ICT Manager", "1330", self.score, self.method)]


@pytest.fixture(autouse=True)
def repos(monkeypatch):
    catalog = [
        Row(skill_id="s1", canonical_name="Project Management", alias="project coordination"),
        Row(skill_id="s2", canonical_name="Budgeting", alias="budget management"),
        Row(skill_id="s3", canonical_name="User Research", alias=None),
        Row(skill_id="s4", canonical_name="Direct Inward Dialing", alias="did"),
    ]
    monkeypatch.setattr(skills, "load_catalog", AsyncMock(return_value=catalog))
    monkeypatch.setattr(roles, "exact_eligible_title", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        roles,
        "nearest_by_embedding",
        AsyncMock(
            return_value=[
                NearestRole("M151122", "IT Project Manager", 0.81, "151122", "1330.7"),
                NearestRole("M251116", "Technical Specialist (.Net)", 0.72, "251116", "2512.4"),
                NearestRole("M251201", "Software Developer", 0.62, "251201", "2512.4"),
            ]
        ),
    )
    monkeypatch.setattr(
        caregiving,
        "reframe",
        AsyncMock(
            return_value=[
                ReframedRow("A1", "Household Coordination", "s1", "Project Management"),
                ReframedRow("A2", "Daily Coordination", "s1", "Project Management"),
            ]
        ),
    )


def request(title="Software Engineer", text="Led project management and budgeting."):
    return SnapshotRequest(
        cv=CV(
            raw_text=text, experiences=[Experience(title=title)], skill_mentions=["User Research"]
        ),
        break_=Break(duration_years=2, activities=["A1", "A2"]),
    )


async def test_literal_evidence_only_no_fuzzy_or_semantic_invention():
    response = await SnapshotService(Settings(), Embedder(), None).generate(request(), object())
    assert {p.skill for p in response.professional_skills} == {"Project Management", "Budgeting"}
    assert all(
        p.evidence_type == "literal" and p.evidence in request().cv.raw_text
        for p in response.professional_skills
    )
    assert len(response.recommended_roles) == 2  # no weak third filler
    assert response.recommended_roles[0].similarity == 0.81  # not fabricated 1.0
    assert response.recommended_roles[0].role_id == "M151122"


async def test_esco_comparison_does_not_become_arbitrary_masco_history(monkeypatch):
    lookup = AsyncMock(side_effect=AssertionError("ESCO must not select a MASCO role"))
    monkeypatch.setattr(roles, "list_by_esco_code", lookup)
    matcher = Matcher()
    response = await SnapshotService(Settings(), Embedder(), matcher).generate(request(), object())
    assert response.previous_occupation.role == "Software Engineer"
    assert response.previous_occupation.method == "cv_title"
    assert response.previous_occupation.confidence is None
    assert response.previous_occupation.esco_code == "1330.5"
    assert response.previous_occupation.comparison_score == 0.9
    assert all(r.role != "Village Community Center Manager" for r in response.recommended_roles)
    assert matcher.calls[0]["job_title"] == "software developer"
    lookup.assert_not_awaited()


@pytest.mark.parametrize(
    "score,method",
    [(0.64, "tfidf_logreg"), (0.74, "tfidf_retrieval"), (float("nan"), "tfidf_logreg")],
)
async def test_low_confidence_comparison_rejected(score, method):
    response = await SnapshotService(Settings(), None, Matcher(score, method)).generate(
        request(), object()
    )
    assert response.previous_occupation.esco_code is None
    assert response.previous_occupation.role == "Software Engineer"


async def test_missing_or_placeholder_title_is_not_replaced_by_target_role():
    response = await SnapshotService(Settings(), Embedder(), Matcher()).generate(
        request(title="Company Name City, State"), object()
    )
    assert response.previous_occupation is None


async def test_older_job_cannot_override_recent_cv_title():
    req = request(title="English Teacher")
    req.cv.experiences.append(Experience(title="Software Engineer"))
    matcher = Matcher()
    response = await SnapshotService(Settings(), Embedder(), matcher).generate(req, object())
    assert response.previous_occupation.role == "English Teacher"
    assert len(matcher.calls) == 1


async def test_exact_eligible_match_without_model(monkeypatch):
    monkeypatch.setattr(
        roles,
        "exact_eligible_title",
        AsyncMock(
            return_value=[NearestRole("M251201", "Software Developer", 1, "251201", "2512.4")]
        ),
    )
    response = await SnapshotService(Settings(), None, None).generate(
        request("Software Developer"), object()
    )
    assert response.recommended_roles[0].method == "exact_title"
    assert response.recommended_roles[0].esco_code == "2512.4"


async def test_recommendation_dedupes_identity_not_shared_esco(monkeypatch):
    monkeypatch.setattr(
        roles,
        "nearest_by_embedding",
        AsyncMock(
            return_value=[
                NearestRole("M1", "Role One", 0.91, "251201", "2512.4"),
                NearestRole("M1", "Role One", 0.91, "251201", "2512.4"),
                NearestRole("M2", "Role Two", 0.89, "251116", "2512.4"),
                NearestRole("R1", "Legacy Role", 0.99, "2512", "2512.4"),
                NearestRole("M3", " role one ", 0.88, "251202", "2512.9"),
            ]
        ),
    )
    response = await SnapshotService(Settings(), Embedder(), None).generate(request(), object())
    assert [r.role_id for r in response.recommended_roles] == ["M1", "M2"]


async def test_reframe_has_canonical_id_and_deduplicates_shared_skills():
    response = await SnapshotService(Settings(), None, None).generate(request(), object())
    assert len(response.reframed_skills) == 1
    assert response.reframed_skills[0].skill_id == "s1"
    assert response.reframed_skills[0].skill == "Project Management"
    assert response.reframed_skills[0].reframed_label == "Household Coordination"


async def test_empty_profile_returns_no_recommendations():
    req = request(title="", text="")
    response = await SnapshotService(Settings(), Embedder(), Matcher()).generate(req, object())
    assert response.previous_occupation is None
    assert response.recommended_roles == []
    assert "no_literal_skill_evidence" in response.warnings
    roles.nearest_by_embedding.assert_not_awaited()


async def test_expired_cache_refreshes_names_and_removes_old_aliases(monkeypatch):
    loader = AsyncMock(
        side_effect=[
            [Row(skill_id="s1", canonical_name="Programming", alias="old alias")],
            [Row(skill_id="s1", canonical_name="Programming (DigComp Competence)", alias=None)],
        ]
    )
    monkeypatch.setattr(skills, "load_catalog", loader)
    cache = SkillCatalog(ttl_seconds=0)
    first = await cache.get(object())
    second = await cache.get(object())
    assert first.canonical["s1"] == "Programming"
    assert second.canonical["s1"] == "Programming (DigComp Competence)"
    assert "old alias" not in second.terms


async def test_unexpired_shared_cache_loads_once():
    cache = SkillCatalog(ttl_seconds=60)
    assert await cache.get(object()) is await cache.get(object())
    skills.load_catalog.assert_awaited_once()


async def test_ambiguous_alias_never_uses_first_row(monkeypatch):
    monkeypatch.setattr(
        skills,
        "load_catalog",
        AsyncMock(
            return_value=[
                Row(skill_id="s1", canonical_name="Programming (O*NET Skill)", alias="programming"),
                Row(
                    skill_id="s2",
                    canonical_name="Programming (DigComp Competence)",
                    alias="programming",
                ),
            ]
        ),
    )
    cache = await SkillCatalog().get(object())
    assert "programming" not in cache.terms
    assert len(cache.canonical) == 2


async def test_common_verb_did_is_not_telephony():
    response = await SnapshotService(Settings(), None, None).generate(
        request(text="I did project management."), object()
    )
    assert [p.skill_id for p in response.professional_skills] == ["s1"]


async def test_failed_refresh_does_not_serve_expired_claims(monkeypatch):
    cache = SkillCatalog(ttl_seconds=0)
    await cache.get(object())
    monkeypatch.setattr(
        skills, "load_catalog", AsyncMock(side_effect=RuntimeError("database offline"))
    )
    with pytest.raises(RuntimeError):
        await cache.get(object())
