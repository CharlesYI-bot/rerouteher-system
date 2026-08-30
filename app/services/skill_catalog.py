"""Shared, bounded-lifetime canonical skill lookup for parser and snapshot."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass

from app.repositories import skills as skills_repo


def normalize_term(term: str) -> str:
    return " ".join(term.casefold().split())


@dataclass(frozen=True)
class SkillCatalogData:
    canonical: dict[str, str]
    terms: dict[str, str]
    patterns: tuple[tuple[str, str, re.Pattern], ...]


def find_evidence(pattern: re.Pattern, term: str, text: str) -> str | None:
    """Literal surface evidence only; ambiguous short labels need local context."""
    for match in pattern.finditer(text):
        surface = match.group()
        context = text[max(0, match.start() - 80) : match.end() + 80].casefold()
        if term == "did" and not re.search(r"\b(telephon\w*|dialing|dialling|pbx|voip)\b", context):
            continue
        if term == "cta" and not re.search(
            r"\b(marketing|conversion|campaign|call.to.action)\b", context
        ):
            continue
        if term in {"c", "r", "go", "swift"} and not re.search(
            r"\b(programming|language|developer|software|ios|python|statistics|statistical)\b",
            context,
        ):
            continue
        if (
            len(term) <= 3
            and term.isalpha()
            and term not in {"git", "sql", "css", "php", "aws", "sap"}
        ):
            if surface != surface.upper():
                continue
        return surface
    return None


def term_pattern(term: str) -> re.Pattern:
    # Includes punctuation-bearing and >4-word skills; never fuzzy-expands a claim.
    escaped = r"\s+".join(re.escape(word) for word in term.split())
    return re.compile(r"(?<![\w+#.])" + escaped + r"(?![\w+#])", re.IGNORECASE)


class SkillCatalog:
    def __init__(self, ttl_seconds: float = 60) -> None:
        self._ttl = ttl_seconds
        self._loaded_at = float("-inf")
        self._data: SkillCatalogData | None = None
        self._lock = asyncio.Lock()

    async def get(self, session) -> SkillCatalogData:
        if self._data is not None and time.monotonic() - self._loaded_at < self._ttl:
            return self._data
        async with self._lock:
            if self._data is not None and time.monotonic() - self._loaded_at < self._ttl:
                return self._data
            # One SQL snapshot: canonical names and aliases cannot come from different commits.
            rows = await skills_repo.load_catalog(session)
            canonical = {str(row.skill_id): row.canonical_name for row in rows}
            owners: dict[str, set[str]] = {}
            for row in rows:
                for label in (row.canonical_name, row.alias):
                    key = normalize_term(label or "")
                    if key:
                        owners.setdefault(key, set()).add(str(row.skill_id))
            # Never resolve ambiguous aliases by database row order.
            terms = {term: next(iter(ids)) for term, ids in owners.items() if len(ids) == 1}
            data = SkillCatalogData(
                canonical,
                terms,
                tuple(
                    (term, skill_id, term_pattern(term)) for term, skill_id in sorted(terms.items())
                ),
            )
            self._data = data  # publish complete replacement, not a partially mutated cache
            self._loaded_at = time.monotonic()
            return data
