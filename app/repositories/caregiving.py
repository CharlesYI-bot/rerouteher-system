"""Caregiving map queries: break activity -> reframed professional label."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class ReframedRow:
    activity_id: str
    reframed_label: str
    skill_id: str
    canonical_name: str


async def reframe(session: AsyncSession, activities: list[str]) -> list[ReframedRow]:
    """Reframe break activities. `activities` are the activity ids the UI sends."""
    if not activities:
        return []
    rows = (
        await session.execute(
            text(
                "SELECT DISTINCT cm.activity_id, cm.reframed_label, st.skill_id, st.canonical_name "
                "FROM caregiving_map cm JOIN skill_taxonomy st ON st.skill_id = cm.onet_skill_id "
                "WHERE cm.activity_id = ANY(:acts) ORDER BY st.skill_id, cm.activity_id"
            ),
            {"acts": activities},
        )
    ).all()
    return [
        ReframedRow(r.activity_id, r.reframed_label, str(r.skill_id), r.canonical_name)
        for r in rows
    ]
