"""Gap regression tests use synthetic canonical skill IDs only."""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.gap import router
from app.config import Settings
from app.db import get_session
from app.repositories import roles, skills
from app.repositories.roles import RoleSkillRow, RoleWithSkills
from app.schemas.gap import GapRequest, UserSkill
from app.services.gap import GapService


def role():
    return RoleWithSkills(
        "M251116",
        "Technical Specialist (.Net)",
        "medium",
        [
            RoleSkillRow(f"s{i}", f"Skill {i}", "digital" if i < 3 else "technical", 100)
            for i in range(10)
        ],
        "251116",
        "2512.4",
        True,
    )


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(roles, "get_role_with_skills", AsyncMock(return_value=role()))
    monkeypatch.setattr(
        skills, "valid_skill_ids", AsyncMock(side_effect=lambda _, ids: ids - {"unknown"})
    )
    return GapService(Settings())


def req(ids=("s1",)):
    return GapRequest(
        target_role_id="M251116", skills=[UserSkill(skill="label", skill_id=s) for s in ids]
    )


@pytest.mark.asyncio
async def test_real_counts_not_top_three_focus_count(service):
    result = await service.compute(req(), object())
    assert result.required_skill_count == 10
    assert result.matched_skill_count == 1
    assert result.missing_skill_count == 9
    assert len(result.gaps) == 3
    assert all(g.skill_id for g in result.gaps)
    assert result.readiness == 13.3


@pytest.mark.asyncio
async def test_all_required_ids_score_100(service):
    result = await service.compute(req(tuple(f"s{i}" for i in range(10))), object())
    assert result.readiness == 100
    assert result.missing_skill_count == 0
    assert result.gaps == []


@pytest.mark.asyncio
async def test_known_skills_without_overlap_are_genuine_zero(service):
    result = await service.compute(req(("another-canonical-skill",)), object())
    assert result.assessment_status == "assessed"
    assert result.readiness == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [(), ("unknown",)])
async def test_no_valid_evidence_is_not_zero_readiness(service, ids):
    result = await service.compute(req(ids), object())
    assert result.assessment_status == "not_assessed"
    assert result.reason == "insufficient_skill_evidence"
    assert result.readiness is None


@pytest.mark.asyncio
async def test_unknown_and_missing_ids_are_warned_not_counted(service):
    request = req(("s1", "unknown"))
    request.skills.append(UserSkill(skill="unmapped"))
    result = await service.compute(request, object())
    assert result.matched_skill_count == 1
    assert set(result.warnings) == {"unknown_skill_ids_ignored", "skills_without_ids_ignored"}


@pytest.mark.asyncio
async def test_duplicate_input_skill_ids_do_not_double_credit(service):
    result = await service.compute(req(("s1", "s1")), object())
    assert result.matched_skill_count == 1
    assert result.readiness == 13.3


@pytest.mark.asyncio
async def test_empty_requirements_never_score_100(service, monkeypatch):
    target = role()
    target.skills = []
    monkeypatch.setattr(roles, "get_role_with_skills", AsyncMock(return_value=target))
    result = await service.compute(req(), object())
    assert result.reason == "no_approved_requirements"
    assert result.readiness is None
    assert service._readiness([], {}, 0.6) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("band", ["technical", "digital"])
async def test_missing_band_gives_no_free_credit(service, monkeypatch, band):
    target = role()
    target.skills = [RoleSkillRow("s1", "One Skill", band, 100)]
    monkeypatch.setattr(roles, "get_role_with_skills", AsyncMock(return_value=target))
    result = await service.compute(req(("outside",)), object())
    assert result.readiness == 0
    assert len(result.warnings) == 1
    result = await service.compute(req(("s1",)), object())
    assert result.readiness == 100


@pytest.mark.asyncio
@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf")])
async def test_invalid_weights_are_not_scored(service, monkeypatch, weight):
    target = role()
    target.skills[0].importance = weight
    monkeypatch.setattr(roles, "get_role_with_skills", AsyncMock(return_value=target))
    result = await service.compute(req(), object())
    assert result.reason == "invalid_requirement_profile"


@pytest.mark.asyncio
async def test_duplicate_requirement_id_is_not_scored(service, monkeypatch):
    target = role()
    target.skills.append(target.skills[0])
    monkeypatch.setattr(roles, "get_role_with_skills", AsyncMock(return_value=target))
    assert (await service.compute(req(), object())).reason == "invalid_requirement_profile"


@pytest.mark.asyncio
async def test_pending_role_rejected_even_when_called_directly(service, monkeypatch):
    target = role()
    target.eligible = False
    monkeypatch.setattr(roles, "get_role_with_skills", AsyncMock(return_value=target))
    result = await service.compute(req(), object())
    assert result.reason == "role_not_eligible"
    assert result.masco_code == "251116"
    assert result.esco_code == "2512.4"


@pytest.mark.asyncio
async def test_unknown_or_ambiguous_title_is_not_assessed(service, monkeypatch):
    monkeypatch.setattr(roles, "get_role_with_skills", AsyncMock(return_value=None))
    assert (await service.compute(req(), object())).reason == "unknown_role"
    monkeypatch.setattr(
        roles, "get_role_with_skills", AsyncMock(side_effect=roles.AmbiguousRoleError())
    )
    assert (await service.compute(req(), object())).reason == "ambiguous_role_title"


@pytest.mark.asyncio
async def test_explicit_stale_reference_version_rejected(service):
    request = req()
    request.reference_version = "old.mapping."
    assert (await service.compute(request, object())).reason == "reference_version_changed"
    roles.get_role_with_skills.assert_not_awaited()


@pytest.mark.asyncio
async def test_unconfirmed_semantic_claim_not_credited(service):
    request = req()
    request.skills[0].evidence_type = "inferred"
    assert (await service.compute(request, object())).readiness is None
    request.skills[0].confirmed = True
    assert (await service.compute(request, object())).readiness == 13.3


@pytest.mark.parametrize("payload", [{}, {"target_role": " "}])
def test_target_is_required(payload):
    with pytest.raises(ValidationError):
        GapRequest(**payload)


def test_endpoint_returns_409_for_not_assessed_not_successful_null(service, monkeypatch):
    async def session():
        yield object()

    app = FastAPI()
    app.include_router(router)
    app.state.gap_service = service
    app.dependency_overrides[get_session] = session
    with TestClient(app) as client:
        response = client.post("/api/gap/compute", json={"target_role_id": "M251116", "skills": []})
        assert response.status_code == 409
        assert response.json()["readiness"] is None
        monkeypatch.setattr(roles, "get_role_with_skills", AsyncMock(return_value=None))
        response = client.post("/api/gap/compute", json={"target_role": "unknown"})
        assert response.status_code == 404
