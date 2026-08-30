"""Read-only role resolution. ESCO comparison codes are not unique MASCO identities."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class AmbiguousRoleError(ValueError):
    pass


@dataclass
class RoleSkillRow:
    skill_id: str
    skill_name: str
    skill_type: str
    importance: float


@dataclass
class RoleWithSkills:
    role_id: str
    role_title: str
    ai_exposure: str
    skills: list[RoleSkillRow]
    masco_code: str | None = None
    esco_code: str | None = None
    eligible: bool = False


@dataclass
class NearestRole:
    role_id: str
    role_title: str
    similarity: float
    masco_code: str
    esco_code: str | None


@dataclass
class Role:
    role_id: str
    role_title: str
    masco_code: str
    esco_code: str | None = None


# Same policy for exact-title/vector recommendations and direct gap targets.
# Missing/stale metadata fails closed. Text comparisons avoid unsafe JSON casts.
_JOIN = "LEFT JOIN dataset_metadata dm ON dm.metadata_key = :prefix || r.role_id"
_ELIGIBLE = """
    r.masco_code ~ '^[0-9]{6}$'
    AND r.flexible_role IS TRUE
    AND r.ai_exposure IN ('low', 'medium', 'high')
    AND dm.metadata_value->>'role_id' = r.role_id
    AND dm.metadata_value->>'masco_code' = r.masco_code
    AND dm.metadata_value->>'esco_code' = r.esco_code
    AND dm.metadata_value->>'mapping_confidence' IN ('medium', 'high')
    AND dm.metadata_value->>'use_in_role_skills' = 'true'
    AND dm.metadata_value->>'review_status' IN
        ('auto_approved_medium_or_high_test_mapping', 'approved', 'manually_approved')
    AND EXISTS (SELECT 1 FROM role_skills rs WHERE rs.role_id = r.role_id
        AND rs.importance > 0 AND rs.skill_type IN ('technical', 'soft', 'digital'))
    AND NOT EXISTS (SELECT 1 FROM role_skills rs WHERE rs.role_id = r.role_id
        AND (rs.importance IS NULL OR rs.importance <= 0
             OR rs.importance::text IN ('NaN', 'Infinity', '-Infinity')
             OR rs.skill_type IS NULL OR rs.skill_type NOT IN ('technical', 'soft', 'digital')))
"""


async def list_by_esco_code(session: AsyncSession, esco_code: str) -> list[Role]:
    """All comparison candidates, never an arbitrary first MASCO role."""
    rows = (
        await session.execute(
            text(
                "SELECT role_id, role_title, masco_code, esco_code FROM roles "
                "WHERE esco_code = :c ORDER BY role_id"
            ),
            {"c": esco_code},
        )
    ).all()
    return [Role(r.role_id, r.role_title, r.masco_code, r.esco_code) for r in rows]


async def get_by_masco_code(session: AsyncSession, masco_code: str) -> Role | None:
    # Four-digit groups cannot identify a six-digit occupation.
    if not re.fullmatch(r"[0-9]{6}", masco_code or ""):
        return None
    rows = (
        await session.execute(
            text(
                "SELECT role_id, role_title, masco_code, esco_code FROM roles WHERE masco_code = :c"
            ),
            {"c": masco_code},
        )
    ).all()
    if len(rows) > 1:
        raise AmbiguousRoleError("Duplicate MASCO identity")
    return (
        Role(rows[0].role_id, rows[0].role_title, rows[0].masco_code, rows[0].esco_code)
        if rows
        else None
    )


async def get_role_with_skills(
    session: AsyncSession,
    role_title: str | None,
    *,
    role_id: str | None,
    prefix: str,
) -> RoleWithSkills | None:
    selector = (
        "r.role_id = :target" if role_id else "lower(btrim(r.role_title)) = lower(btrim(:target))"
    )
    rows = (
        await session.execute(
            text(
                f"SELECT r.role_id, r.role_title, r.ai_exposure, r.masco_code, r.esco_code, "
                f"COALESCE(({_ELIGIBLE}), false) AS eligible FROM roles r {_JOIN} WHERE {selector}"
            ),
            {"target": role_id or role_title, "prefix": prefix},
        )
    ).all()
    if len(rows) > 1:
        raise AmbiguousRoleError("Ambiguous title; use target_role_id")
    if not rows:
        return None
    role = rows[0]
    if not role.eligible:
        # Do not parse malformed/NULL requirement weights for a rejected profile.
        return RoleWithSkills(
            role.role_id,
            role.role_title,
            role.ai_exposure,
            [],
            role.masco_code,
            role.esco_code,
            False,
        )
    rows = (
        await session.execute(
            text(
                "SELECT rs.skill_id, st.canonical_name AS skill_name, rs.skill_type, rs.importance "
                "FROM role_skills rs JOIN skill_taxonomy st ON st.skill_id = rs.skill_id "
                "WHERE rs.role_id = :rid ORDER BY rs.skill_id"
            ),
            {"rid": role.role_id},
        )
    ).all()
    skills = [
        RoleSkillRow(str(r.skill_id), r.skill_name, r.skill_type, float(r.importance)) for r in rows
    ]
    return RoleWithSkills(
        role.role_id,
        role.role_title,
        role.ai_exposure,
        skills,
        role.masco_code,
        role.esco_code,
        role.eligible,
    )


async def exact_eligible_title(
    session: AsyncSession, title: str, *, prefix: str
) -> list[NearestRole]:
    rows = (
        await session.execute(
            text(
                f"SELECT r.role_id, r.role_title, r.masco_code, r.esco_code FROM roles r {_JOIN} "
                f"WHERE {_ELIGIBLE} AND lower(btrim(r.role_title)) = lower(btrim(:title)) ORDER BY r.role_id"
            ),
            {"title": title, "prefix": prefix},
        )
    ).all()
    if len(rows) != 1:  # ambiguous labels are not exact identity matches
        return []
    r = rows[0]
    return [NearestRole(r.role_id, r.role_title, 1.0, r.masco_code, r.esco_code)]


async def nearest_by_embedding(
    session: AsyncSession,
    query_vec: np.ndarray,
    k: int,
    *,
    prefix: str,
    threshold: float,
) -> list[NearestRole]:
    vec = np.asarray(query_vec)
    if vec.shape != (384,) or not np.isfinite(vec).all() or not np.any(vec):
        return []
    literal = "[" + ",".join(f"{x:.6f}" for x in vec.tolist()) + "]"
    rows = (
        await session.execute(
            text(
                f"SELECT r.role_id, r.role_title, r.masco_code, r.esco_code, "
                f"1 - (r.role_embedding <=> CAST(:v AS vector)) AS similarity FROM roles r {_JOIN} "
                f"WHERE {_ELIGIBLE} AND r.role_embedding IS NOT NULL "
                "AND 1 - (r.role_embedding <=> CAST(:v AS vector)) >= :threshold "
                "ORDER BY r.role_embedding <=> CAST(:v AS vector), r.role_id LIMIT :k"
            ),
            {"v": literal, "k": k, "prefix": prefix, "threshold": threshold},
        )
    ).all()
    return [
        NearestRole(r.role_id, r.role_title, float(r.similarity), r.masco_code, r.esco_code)
        for r in rows
    ]
