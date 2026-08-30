"""Evidence-grounded skills, CV-stated occupation, and eligible MASCO recommendations.

ESCO classification is retained as a comparison only. It never chooses a MASCO role
by shared ESCO code or a four-digit prefix. No writes, migrations or model training.
"""

from __future__ import annotations

import logging
import math
import re

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.repositories import caregiving as caregiving_repo
from app.repositories import roles as roles_repo
from app.schemas.snapshot import (
    PreviousOccupation,
    ProfessionalSkill,
    RecommendedRole,
    ReframedSkill,
    SnapshotRequest,
    SnapshotResponse,
)
from app.services.cv_extractor import occupation_title
from app.services.occupation_matcher import normalize_text
from app.services.role_titles import candidate_titles
from app.services.skill_catalog import SkillCatalog, find_evidence

logger = logging.getLogger("rerouteher")


class SnapshotService:
    def __init__(self, settings: Settings, embedder, tfidf_matcher, skill_catalog=None) -> None:
        self._settings = settings
        self._embedder = embedder
        self._tfidf = tfidf_matcher
        self.skill_catalog = skill_catalog or SkillCatalog(settings.skill_cache_ttl_seconds)

    async def generate(self, req: SnapshotRequest, session: AsyncSession) -> SnapshotResponse:
        catalog = await self.skill_catalog.get(session)
        found: dict[str, ProfessionalSkill] = {}
        for term, skill_id, pattern in catalog.patterns:
            evidence = find_evidence(pattern, term, req.cv.raw_text)
            if evidence:
                found.setdefault(
                    skill_id,
                    ProfessionalSkill(
                        skill=catalog.canonical[skill_id],
                        skill_id=skill_id,
                        evidence=evidence,
                        evidence_type="literal",
                    ),
                )
        # skill_mentions from older parsers are not independent evidence. Semantic
        # similarities are not verified claims and are no longer auto-credited.
        professional = list(found.values())
        reframed = await self._reframe_break(req, session)
        previous, recommended = await self._match_occupation(req, professional, session)
        warnings = list(req.cv.warnings)
        if not professional:
            warnings.append("no_literal_skill_evidence")
        if not recommended:
            warnings.append("no_eligible_role_above_threshold")
        logger.info(
            "snapshot: professional=%d reframed=%d recommended=%d",
            len(professional),
            len(reframed),
            len(recommended),
        )
        return SnapshotResponse(
            professional_skills=professional,
            reframed_skills=reframed,
            previous_occupation=previous,
            recommended_roles=recommended,
            warnings=warnings,
            reference_version=self._settings.mapping_metadata_prefix,
        )

    async def _reframe_break(self, req, session) -> list[ReframedSkill]:
        rows = await caregiving_repo.reframe(session, req.break_.activities)
        found: dict[str, ReframedSkill] = {}
        for row in rows:
            # Repository joins the explicit caregiving_map.onet_skill_id. Never
            # guess an ESCO identity from a similarly worded reframed display label.
            if row.skill_id:
                found.setdefault(
                    row.skill_id,
                    ReframedSkill(
                        skill=row.canonical_name,
                        skill_id=row.skill_id,
                        reframed_label=row.reframed_label,
                        from_activity=row.activity_id,
                    ),
                )
        return list(found.values())

    async def _match_occupation(self, req, professional, session):
        # Preserve first valid CV-stated title (CV order). An older high-scoring
        # classifier result cannot overwrite it; no recommendation becomes history.
        title = next(
            (clean for exp in req.cv.experiences if (clean := occupation_title(exp.title or ""))),
            None,
        )
        previous = PreviousOccupation(role=title, method="cv_title") if title else None
        skill_names = [p.skill for p in professional]
        if previous is not None and self._tfidf is not None:
            try:
                matches = self._tfidf.predict(
                    job_title=normalize_text(title), skills=skill_names, top_k=1
                )
            except Exception:  # optional model must not leak CV/title/exception payloads
                logger.warning("ESCO comparison unavailable")
                matches = []
            if matches:
                match = matches[0]
                threshold = (
                    self._settings.occupation_confidence_threshold
                    if match.method == "tfidf_logreg"
                    else self._settings.occupation_retrieval_threshold
                )
                if math.isfinite(match.score) and threshold <= match.score <= 1:
                    previous.esco_code = match.esco_code
                    previous.esco_title = match.esco_title
                    previous.comparison_score = match.score
                    previous.comparison_method = match.method

        prefix = self._settings.mapping_metadata_prefix
        exact = (
            await roles_repo.exact_eligible_title(session, title, prefix=prefix) if title else []
        )
        ranked = [(role, "exact_title") for role in exact]
        if title and not exact:
            for variant in candidate_titles(title):
                ranked.extend(
                    (role, "title_variant")
                    for role in await roles_repo.exact_eligible_title(
                        session, variant, prefix=prefix
                    )
                )
        # A long skills list can dilute the occupational title in the profile
        # embedding. Retrieve by title as well, without lowering the quality gate.
        profile = " ".join([title or "", *skill_names[:12]]).strip()
        queries = list(dict.fromkeys(text for text in (title, profile) if text))
        vector_candidates = {}
        if self._embedder is not None:
            for query_text in queries:
                query = self._embedder.encode_one(query_text)
                for role in await roles_repo.nearest_by_embedding(
                    session,
                    query,
                    k=12,
                    prefix=prefix,
                    threshold=self._settings.role_cosine_threshold,
                ):
                    previous_match = vector_candidates.get(role.role_id)
                    if math.isfinite(role.similarity) and (
                        previous_match is None or role.similarity > previous_match.similarity
                    ):
                        vector_candidates[role.role_id] = role
        ranked.extend(
            (role, "embedding")
            for role in sorted(
                vector_candidates.values(), key=lambda role: (-role.similarity, role.role_id)
            )
        )
        recommended = []
        ids, titles, codes = set(), set(), set()
        for role, method in ranked:
            title_key = " ".join(role.role_title.casefold().split())
            if not re.fullmatch(r"[0-9]{6}", role.masco_code or ""):
                continue
            if not math.isfinite(role.similarity) or not 0 <= role.similarity <= 1:
                continue
            if method == "embedding" and role.similarity < self._settings.role_cosine_threshold:
                continue
            if role.role_id in ids or title_key in titles or role.masco_code in codes:
                continue
            ids.add(role.role_id)
            titles.add(title_key)
            codes.add(role.masco_code)
            recommended.append(
                RecommendedRole(
                    role=role.role_title,
                    role_id=role.role_id,
                    masco_code=role.masco_code,
                    esco_code=role.esco_code,
                    similarity=None if method == "title_variant" else round(role.similarity, 3),
                    method=method,
                )
            )
            if len(recommended) == 3:
                break
        return previous, recommended
