"""Actual SQL runs in disposable in-memory PostgreSQL (PGlite + pgvector).

Opt in with PGLITE_DEPS_DIR after installing tests/sql/package.json dependencies.
This harness never uses DATABASE_URL and cannot connect to a deployed database.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace as Row

import numpy as np
import pytest
from sqlalchemy.dialects.postgresql import asyncpg

from app.config import Settings
from app.repositories import caregiving, roles, skills
from app.schemas.cv import CV, Experience
from app.schemas.gap import GapRequest
from app.schemas.snapshot import Break, SnapshotRequest
from app.services.gap import GapService
from app.services.snapshot import SnapshotService

PREFIX = Settings().mapping_metadata_prefix
ROOT = Path(__file__).parent


class Result:
    def __init__(self, rows):
        self.rows = [Row(**row) for row in rows]

    def all(self):
        return self.rows


class Database:
    def __init__(self, process):
        self.process = process

    def command(self, sql, params=None, exec=False):
        self.process.stdin.write(json.dumps({"sql": sql, "params": params, "exec": exec}) + "\n")
        self.process.stdin.flush()
        reply = json.loads(self.process.stdout.readline())
        if "error" in reply:
            raise AssertionError(reply["error"])
        return reply["result"]

    async def execute(self, statement, params=None):
        compiled = statement.compile(dialect=asyncpg.dialect())
        args = [(params or {}).get(key, compiled.params[key]) for key in compiled.positiontup]
        return Result(self.command(str(compiled), args)["rows"])


@pytest.fixture(scope="module")
def postgres():
    if not os.environ.get("PGLITE_DEPS_DIR"):
        pytest.skip("Optional SQL suite: set PGLITE_DEPS_DIR (see docs/BACKEND_TEST_BRANCH.md)")
    node = shutil.which("node")
    assert node, "Node is required for optional SQL tests"
    process = subprocess.Popen(
        [node, str(ROOT / "sql" / "pglite_runner.cjs")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert json.loads(process.stdout.readline())["ready"]
        db = Database(process)
        # Minimal test-only schema has existing production column names. No migrations.
        db.command(
            """
            CREATE TABLE roles (
                role_id text PRIMARY KEY, role_title text, masco_code text, esco_code text,
                flexible_role boolean, ai_exposure text, role_embedding vector(384)
            );
            CREATE TABLE dataset_metadata (metadata_key text PRIMARY KEY, metadata_value jsonb);
            CREATE TABLE skill_taxonomy (skill_id text PRIMARY KEY, canonical_name text, skill_type text);
            CREATE TABLE skill_aliases (skill_id text REFERENCES skill_taxonomy, alias text);
            CREATE TABLE role_skills (
                role_id text REFERENCES roles, skill_id text REFERENCES skill_taxonomy,
                skill_name text, skill_type text, importance numeric, PRIMARY KEY(role_id, skill_id)
            );
            CREATE TABLE caregiving_map (
                activity_id text, reframed_label text, onet_skill_id text REFERENCES skill_taxonomy
            );
        """,
            exec=True,
        )
        yield db
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)


@pytest.fixture
def db(postgres):
    postgres.command(
        "TRUNCATE caregiving_map, role_skills, skill_aliases, skill_taxonomy, dataset_metadata, roles",
        exec=True,
    )
    postgres.command(
        "INSERT INTO skill_taxonomy VALUES ('s1', 'Canonical Skill', 'technical')", exec=True
    )
    return postgres


def insert_role(
    db,
    code="251116",
    title="Technical Specialist (.Net)",
    confidence="medium",
    approved=True,
    flexible=True,
    core=True,
    metadata=True,
    esco="2512.4",
):
    rid = "M" + code
    vector = "[" + ",".join(["1"] + ["0"] * 383) + "]"
    db.command(
        "INSERT INTO roles VALUES ($1,$2,$3,$4,$5,'medium',$6::vector)",
        [rid, title, code, esco, flexible, vector],
    )
    if core:
        db.command(
            "INSERT INTO role_skills VALUES ($1,'s1','Old Display Name','technical',100)", [rid]
        )
    if metadata:
        meta = dict(
            role_id=rid,
            masco_code=code,
            esco_code=esco,
            mapping_confidence=confidence,
            review_status="auto_approved_medium_or_high_test_mapping"
            if approved
            else "pending_low_confidence_review",
            use_in_role_skills=approved,
        )
        db.command(
            "INSERT INTO dataset_metadata VALUES ($1,$2::jsonb)", [PREFIX + rid, json.dumps(meta)]
        )
    return rid


@pytest.mark.asyncio
async def test_automatic_title_variant_to_gap_with_real_queries(db):
    insert_role(db, code="251201", title="Software Developer")
    req = SnapshotRequest(
        cv=CV(
            raw_text="Used Canonical Skill.", experiences=[Experience(title="Software Engineer")]
        ),
        break_=Break(duration_years=2),
    )
    snapshot = await SnapshotService(Settings(), None, None).generate(req, db)
    assert snapshot.previous_occupation.role == "Software Engineer"
    assert len(snapshot.recommended_roles) == 1
    target = snapshot.recommended_roles[0]
    assert (target.role_id, target.masco_code, target.esco_code) == ("M251201", "251201", "2512.4")
    assert target.method == "title_variant"
    assert target.similarity is None
    gap = await GapService(Settings()).compute(
        GapRequest(
            target_role_id=target.role_id,
            reference_version=snapshot.reference_version,
            skills=[skill.model_dump() for skill in snapshot.professional_skills],
        ),
        db,
    )
    assert gap.assessment_status == "assessed"
    assert gap.readiness == 100
    assert gap.required_skill_count == gap.matched_skill_count == 1


@pytest.mark.asyncio
async def test_title_variant_never_bypasses_mapping_approval(db):
    insert_role(db, code="251201", title="Software Developer", confidence="low", approved=False)
    req = SnapshotRequest(
        cv=CV(
            raw_text="Used Canonical Skill.", experiences=[Experience(title="Software Engineer")]
        ),
        break_=Break(duration_years=2),
    )
    snapshot = await SnapshotService(Settings(), None, None).generate(req, db)
    assert snapshot.recommended_roles == []
    assert "no_eligible_role_above_threshold" in snapshot.warnings


@pytest.mark.asyncio
async def test_shared_esco_returns_all_candidates_not_first(db):
    insert_role(db, code="251116")
    insert_role(db, code="251201", title="Software Developer")
    result = await roles.list_by_esco_code(db, "2512.4")
    assert [r.masco_code for r in result] == ["251116", "251201"]


@pytest.mark.asyncio
async def test_four_digit_masco_is_not_a_role_or_prefix_lookup(db):
    insert_role(db)
    assert await roles.get_by_masco_code(db, "2511") is None
    assert (await roles.get_by_masco_code(db, "251116")).role_id == "M251116"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"confidence": "low"},
        {"approved": False},
        {"flexible": False},
        {"core": False},
        {"metadata": False},
        {"code": "2511"},
    ],
)
async def test_ineligible_roles_excluded_on_all_entry_paths(db, change):
    rid = insert_role(db, **change)
    assert await roles.exact_eligible_title(db, "Technical Specialist (.Net)", prefix=PREFIX) == []
    assert (
        await roles.nearest_by_embedding(
            db, np.array([1] + [0] * 383), 3, prefix=PREFIX, threshold=0.65
        )
        == []
    )
    target = await roles.get_role_with_skills(db, None, role_id=rid, prefix=PREFIX)
    assert target.eligible is False


@pytest.mark.asyncio
@pytest.mark.parametrize("confidence", ["medium", "high"])
async def test_medium_and_high_eligible_with_identity_codes(db, confidence):
    insert_role(db, confidence=confidence)
    result = await roles.nearest_by_embedding(
        db, np.array([1] + [0] * 383), 3, prefix=PREFIX, threshold=0.65
    )
    assert result[0].masco_code == "251116"
    assert result[0].esco_code == "2512.4"
    assert result[0].similarity == 1
    target = await roles.get_role_with_skills(db, None, role_id="M251116", prefix=PREFIX)
    assert target.eligible
    assert target.skills[0].skill_name == "Canonical Skill"  # not duplicated stale role label


@pytest.mark.asyncio
async def test_wrong_metadata_version_or_code_fails_closed(db):
    insert_role(db)
    assert (
        await roles.exact_eligible_title(db, "Technical Specialist (.Net)", prefix="old.mapping.")
        == []
    )
    db.command(
        "UPDATE dataset_metadata SET metadata_value = jsonb_set(metadata_value, '{esco_code}', '\"wrong\"')",
        exec=True,
    )
    assert await roles.exact_eligible_title(db, "Technical Specialist (.Net)", prefix=PREFIX) == []


@pytest.mark.asyncio
async def test_ambiguous_title_requires_stable_id(db):
    insert_role(db)
    insert_role(db, code="251201", title="technical specialist (.net)")
    with pytest.raises(roles.AmbiguousRoleError):
        await roles.get_role_with_skills(
            db, "Technical Specialist (.Net)", role_id=None, prefix=PREFIX
        )
    assert await roles.exact_eligible_title(db, "Technical Specialist (.Net)", prefix=PREFIX) == []
    assert (
        await roles.get_role_with_skills(db, None, role_id="M251201", prefix=PREFIX)
    ).role_id == "M251201"


@pytest.mark.asyncio
async def test_vector_threshold_excludes_weak_or_missing_profile(db):
    insert_role(db)
    assert (
        await roles.nearest_by_embedding(
            db, np.array([0, 1] + [0] * 382), 3, prefix=PREFIX, threshold=0.65
        )
        == []
    )
    assert (
        await roles.nearest_by_embedding(db, np.zeros(384), 3, prefix=PREFIX, threshold=0.65) == []
    )


@pytest.mark.asyncio
async def test_gap_end_to_end_with_real_queries_and_no_digital_bonus(db):
    insert_role(db)
    db.command("INSERT INTO skill_taxonomy VALUES ('other','Other Skill','technical')", exec=True)
    request = GapRequest(
        target_role_id="M251116", skills=[{"skill": "Other Skill", "skill_id": "other"}]
    )
    result = await GapService(Settings()).compute(request, db)
    assert result.assessment_status == "assessed"
    assert result.readiness == 0
    assert result.required_skill_count == 1
    assert result.warnings == ["digital_band_missing_score_uses_role_band_only"]


@pytest.mark.asyncio
async def test_existing_caregiving_id_and_catalog_queries(db):
    db.command("INSERT INTO caregiving_map VALUES ('A1','Reframed display','s1')", exec=True)
    db.command("INSERT INTO skill_aliases VALUES ('s1','alias')", exec=True)
    assert (await caregiving.reframe(db, ["A1"]))[0].skill_id == "s1"
    assert (await skills.load_catalog(db))[0].canonical_name == "Canonical Skill"
    assert await skills.valid_skill_ids(db, {"s1", "missing"}) == {"s1"}


@pytest.mark.asyncio
@pytest.mark.parametrize("weight", [None, "NaN", "Infinity", "0", "-1"])
async def test_malformed_requirement_weight_fails_closed_without_crash(db, weight):
    insert_role(db)
    db.command("UPDATE role_skills SET importance = $1::numeric", [weight])
    assert await roles.exact_eligible_title(db, "Technical Specialist (.Net)", prefix=PREFIX) == []
    target = await roles.get_role_with_skills(db, None, role_id="M251116", prefix=PREFIX)
    assert target.eligible is False
    result = await GapService(Settings()).compute(GapRequest(target_role_id="M251116"), db)
    assert result.assessment_status == "not_assessed"


@pytest.mark.asyncio
async def test_snapshot_to_gap_api_journey_uses_real_sql(db):
    from httpx import ASGITransport, AsyncClient

    from app.db import get_session
    from app.main import create_app
    from app.services.snapshot import SnapshotService

    insert_role(db, title="Software Engineer")
    app = create_app()
    app.state.snapshot_service = SnapshotService(Settings(), None, None)
    app.state.gap_service = GapService(Settings())

    async def session():
        yield db

    app.dependency_overrides[get_session] = session
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
        response = await client.post(
            "/api/snapshot/generate",
            json={
                "cv": {
                    "raw_text": "Used Canonical Skill.",
                    "experiences": [{"title": "Software Engineer"}],
                },
                "break": {"duration_years": 2, "activities": []},
            },
        )
        assert response.status_code == 200
        snapshot = response.json()
        assert snapshot["previous_occupation"]["role"] == "Software Engineer"
        assert snapshot["professional_skills"][0]["skill_id"] == "s1"
        assert snapshot["recommended_roles"][0]["masco_code"] == "251116"
        response = await client.post(
            "/api/gap/compute",
            json={
                "target_role_id": snapshot["recommended_roles"][0]["role_id"],
                "reference_version": snapshot["reference_version"],
                "skills": snapshot["professional_skills"],
            },
        )
        assert response.status_code == 200
        assert response.json()["readiness"] == 100
        assert response.json()["required_skill_count"] == 1
