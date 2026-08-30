"""Skill taxonomy queries: alias lookup and pgvector semantic match."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class SkillMatch:
    skill_id: str
    canonical_name: str
    similarity: float


@dataclass
class SkillRow:
    skill_id: str
    canonical_name: str
    skill_type: str


async def load_catalog(session: AsyncSession):
    return (
        await session.execute(
            text(
                "SELECT st.skill_id, st.canonical_name, sa.alias FROM skill_taxonomy st "
                "LEFT JOIN skill_aliases sa ON sa.skill_id = st.skill_id "
                "ORDER BY st.skill_id, sa.alias"
            )
        )
    ).all()


async def list_skills(session: AsyncSession) -> list[SkillRow]:
    """Canonical skill rows for building a skill_id -> canonical lookup."""
    rows = (
        await session.execute(
            text("SELECT skill_id, canonical_name, skill_type FROM skill_taxonomy")
        )
    ).all()
    return [SkillRow(str(r.skill_id), r.canonical_name, r.skill_type) for r in rows]


async def load_alias_dictionary(session: AsyncSession) -> list[tuple[str, str]]:
    """(skill_id, term) pairs from canonical names and the skill_aliases table."""
    rows = (
        await session.execute(
            text(
                "SELECT skill_id, canonical_name AS term FROM skill_taxonomy "
                "UNION ALL "
                "SELECT skill_id, alias AS term FROM skill_aliases"
            )
        )
    ).all()
    return [(str(r.skill_id), r.term) for r in rows if r.term]


async def valid_skill_ids(session: AsyncSession, skill_ids: set[str]) -> set[str]:
    if not skill_ids:
        return set()
    rows = (
        await session.execute(
            text("SELECT skill_id FROM skill_taxonomy WHERE skill_id = ANY(:ids)"),
            {"ids": sorted(skill_ids)},
        )
    ).all()
    return {str(r.skill_id) for r in rows}


async def match_by_embedding(
    session: AsyncSession, query_vec: np.ndarray, k: int, threshold: float
) -> list[SkillMatch]:
    """Cosine kNN over skill_taxonomy.embedding, keeping matches above `threshold`."""
    vec_literal = "[" + ",".join(f"{x:.6f}" for x in query_vec.tolist()) + "]"
    rows = (
        await session.execute(
            text(
                "SELECT skill_id, canonical_name, "
                "1 - (embedding <=> CAST(:v AS vector)) AS similarity "
                "FROM skill_taxonomy "
                "ORDER BY embedding <=> CAST(:v AS vector) LIMIT :k"
            ),
            {"v": vec_literal, "k": k},
        )
    ).all()
    return [
        SkillMatch(r.skill_id, r.canonical_name, float(r.similarity))
        for r in rows
        if float(r.similarity) >= threshold
    ]
