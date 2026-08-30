"""Two-band skill-gap engine (exact skill_id match).

A role skill is covered only when the user has the exact canonical skill_id.
Readiness blends the two bands (role skills, AI/digital) by the role's AI-exposure, and
the focus list is the top-3 uncovered skills by readiness uplift. Nothing is stored.
"""

from __future__ import annotations

import math

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.repositories import roles as roles_repo
from app.repositories import skills as skills_repo
from app.schemas.gap import Gap, GapRequest, GapResponse


class GapService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def compute(self, req: GapRequest, session: AsyncSession) -> GapResponse:
        version = self._settings.mapping_metadata_prefix
        if req.reference_version is not None and req.reference_version != version:
            return GapResponse(reason="reference_version_changed", reference_version=version)
        try:
            role = await roles_repo.get_role_with_skills(
                session, req.target_role, role_id=req.target_role_id, prefix=version
            )
        except roles_repo.AmbiguousRoleError:
            return GapResponse(reason="ambiguous_role_title", reference_version=version)
        if role is None:
            return GapResponse(reason="unknown_role", reference_version=version)
        identity = dict(
            role_id=role.role_id,
            masco_code=role.masco_code,
            esco_code=role.esco_code,
            reference_version=version,
        )
        if not role.eligible:
            return GapResponse(reason="role_not_eligible", **identity)
        if not role.skills:
            return GapResponse(reason="no_approved_requirements", **identity)
        if len({s.skill_id for s in role.skills}) != len(role.skills) or any(
            s.skill_type not in {"technical", "soft", "digital"}
            or not math.isfinite(s.importance)
            or s.importance <= 0
            for s in role.skills
        ):
            return GapResponse(reason="invalid_requirement_profile", **identity)
        if role.ai_exposure not in {"low", "medium", "high"}:
            return GapResponse(reason="invalid_ai_exposure", **identity)

        supplied_ids = {s.skill_id for s in req.skills if s.skill_id}
        known_ids = await skills_repo.valid_skill_ids(session, supplied_ids)
        have_ids = {
            s.skill_id
            for s in req.skills
            if s.skill_id in known_ids
            and (s.evidence_type not in {"inferred", "semantic"} or s.confirmed)
        }
        if not have_ids:
            return GapResponse(
                reason="insufficient_skill_evidence",
                **identity,
                required_skill_count=len(role.skills),
            )
        warnings = []
        if supplied_ids - known_ids:
            warnings.append("unknown_skill_ids_ignored")
        if any(not s.skill_id for s in req.skills):
            warnings.append("skills_without_ids_ignored")
        if any(s.evidence_type in {"inferred", "semantic"} and not s.confirmed for s in req.skills):
            warnings.append("unconfirmed_inferences_ignored")
        if not any(s.skill_type == "digital" for s in role.skills):
            warnings.append("digital_band_missing_score_uses_role_band_only")
        if not any(s.skill_type in {"technical", "soft"} for s in role.skills):
            warnings.append("role_band_missing_score_uses_digital_band_only")
        cov = {rs.skill_id: (1.0 if rs.skill_id in have_ids else 0.0) for rs in role.skills}

        exposure_w = self._settings.ai_exposure_weight(role.ai_exposure)
        readiness = self._readiness(role.skills, cov, exposure_w)
        skills_have = sorted({rs.skill_name for rs in role.skills if cov[rs.skill_id] >= 1.0})
        gaps = self._rank_gaps(role.skills, cov, exposure_w, readiness)
        matched = sum(s.skill_id in have_ids for s in role.skills)
        return GapResponse(
            readiness=round(readiness, 1),
            assessment_status="assessed",
            skills_have=skills_have,
            gaps=gaps,
            **identity,
            warnings=warnings,
            required_skill_count=len(role.skills),
            matched_skill_count=matched,
            missing_skill_count=len(role.skills) - matched,
        )

    # --------------------------------------------------------------- readiness
    def _readiness(self, role_skills, cov: dict[str, float], exposure_w: float) -> float | None:
        role_band = [rs for rs in role_skills if rs.skill_type in ("technical", "soft")]
        ai_band = [rs for rs in role_skills if rs.skill_type == "digital"]
        role_cov = self._band_coverage(role_band, cov)
        ai_cov = self._band_coverage(ai_band, cov)
        # Empty bands are not competence. Score only populated bands and warn the
        # caller about incomplete references in compute; this is not a job-readiness probability.
        populated = [
            (w, coverage)
            for w, coverage in [(1 - exposure_w, role_cov), (exposure_w, ai_cov)]
            if coverage is not None
        ]
        if not populated:
            return None
        return sum(w * coverage for w, coverage in populated) / sum(w for w, _ in populated) * 100

    @staticmethod
    def _band_coverage(band, cov: dict[str, float]) -> float | None:
        total = sum(float(rs.importance) for rs in band)
        if total == 0:
            return None
        covered = sum(float(rs.importance) * cov.get(rs.skill_id, 0.0) for rs in band)
        return covered / total

    # -------------------------------------------------------------------- gaps
    def _rank_gaps(
        self, role_skills, cov: dict[str, float], exposure_w: float, base: float
    ) -> list[Gap]:
        gaps: list[Gap] = []
        for rs in role_skills:
            if cov.get(rs.skill_id, 0.0) >= 1.0:
                continue
            uplift = self._uplift(role_skills, cov, exposure_w, base, rs.skill_id)
            band = "ai_digital" if rs.skill_type == "digital" else "role"
            gaps.append(
                Gap(
                    skill_id=rs.skill_id,
                    skill=rs.skill_name,
                    band=band,
                    importance=float(rs.importance),
                    uplift=uplift,
                )
            )
        gaps.sort(key=lambda g: (g.uplift, g.importance), reverse=True)
        return gaps[:3]

    def _uplift(
        self, role_skills, cov: dict[str, float], exposure_w, base: float, skill_id: str
    ) -> float:
        # marginal readiness gain if this skill were covered
        boosted = dict(cov)
        boosted[skill_id] = 1.0
        return round(self._readiness(role_skills, boosted, exposure_w) - base, 1)
