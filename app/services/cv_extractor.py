"""Conservative text-PDF parser: literal skills and dated employment only.

Email/phone redaction is best-effort, not a guarantee that a CV contains no PII.
Documents remain in memory; request/response bodies must never be logged.
"""

from __future__ import annotations

import re

import pymupdf

from app.schemas.cv import CV, Experience
from app.services.skill_catalog import find_evidence, normalize_term, term_pattern


class UnreadableCVError(Exception):
    """No extractable text layer; OCR is not performed."""


_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Do not consume newlines (which can join phone numbers to employment dates).
_PHONE_RE = re.compile(r"(?<!\w)(\+?\d[\d \t().\-]{7,}\d)(?!\w)")
_EXPERIENCE_HEADER_RE = re.compile(
    r"^\s*(work\s+experience|professional\s+experience|employment(?:\s+history)?"
    r"|work\s+history|career\s+history|experience|professional\s+background"
    r"|relevant\s+experience|career\s+experience)\s*:?\s*$",
    re.IGNORECASE,
)
_OTHER_SECTION_RE = re.compile(
    r"^\s*(education(?:\s+and\s+training)?|skills?|technical\s+skills?|certifications?"
    r"|projects?|awards?|references?|summary|profile|objective|interests?|languages?"
    r"|volunteer|publications?|achievements?|contact|core\s+(skills|competencies)"
    r"|software\s+knowledge|personal\s+information|hobbies)\s*:?\s*$",
    re.IGNORECASE,
)
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DATE = rf"(?:{_MONTH}\s*)?(?:\d{{1,2}}[/-])?\d{{4}}"
_DATE_RANGE_RE = re.compile(
    rf"({_DATE})\s*(?:-|–|—|to|until|\bthrough\b)\s*({_DATE}|present|current|now|ongoing)",
    re.IGNORECASE,
)
_JOB_WORD_RE = re.compile(
    r"\b(manager|engineer|developer|analyst|scientist|technician|teacher|instructor"
    r"|lecturer|tutor|designer|artist|programmer|administrator|coordinator|officer"
    r"|director|supervisor|consultant|specialist|executive|assistant|secretary|clerk"
    r"|accountant|bookkeeper|nurse|pharmacist|dietitian|dietician|mechanic|chef|cook"
    r"|intern|associate|representative|president|ceo|cto|cfo|lead|head|operator|researcher)\b",
    re.IGNORECASE,
)
_DUTY_RE = re.compile(
    r"^(?:[-•*]|\d+[.)]|led\b|managed\b|developed\b|designed\b|conducted\b|responsible\b"
    r"|worked\b|performed\b|supported\b|assisted\b|provided\b|created\b|implemented\b)",
    re.IGNORECASE,
)
_EDUCATION_RE = re.compile(
    r"\b(bachelor|master of|bsc|msc|phd|diploma|degree|gpa)\b", re.IGNORECASE
)


def occupation_title(value: str) -> str | None:
    """Reject obvious employer/location/education/duty lines rather than invent a role."""
    value = _DATE_RANGE_RE.sub("", value).strip(" -–—|,:()[]{}\t")
    value = re.sub(r"\bcompany name\b.*$", "", value, flags=re.IGNORECASE).strip(" -–—|,")
    for part in re.split(r"\s*\|\s*|\s{2,}|\s+[-–—]\s+", value):
        if (
            part
            and len(part.split()) <= 12
            and not _DUTY_RE.search(part)
            and not _EDUCATION_RE.search(part)
            and _JOB_WORD_RE.search(part)
            and not re.search(r"\b(inc|ltd|llc|sdn bhd)\.?$", part, re.IGNORECASE)
        ):
            return part.strip()
    return None


class CVExtractor:
    def __init__(self, nlp, skill_dictionary: list[str], embedder=None) -> None:
        self._nlp = nlp
        self._skill_dictionary = skill_dictionary
        # Semantic header matching intentionally removed: job titles are not headers.

    def parse(self, pdf_bytes: bytes, skill_dictionary: list[str] | None = None) -> CV:
        try:
            raw_text = self._extract_text(pdf_bytes)
        except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
            raise UnreadableCVError("unreadable, please upload a text-based PDF") from exc
        if not raw_text.strip():
            raise UnreadableCVError("unreadable, please upload a text-based PDF")
        raw_text = self._redact_pii(raw_text)
        experiences = self._segment_experiences(raw_text)
        warnings = (
            [] if any(e.title for e in experiences) else ["employment_title_not_reliably_extracted"]
        )
        return CV(
            raw_text=raw_text,
            experiences=experiences,
            skill_mentions=self._match_skills(raw_text, skill_dictionary),
            warnings=warnings,
        )

    # ------------------------------------------------------------------ text
    def _extract_text(self, pdf_bytes: bytes) -> str:
        parts: list[str] = []
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                parts.append(self._page_text(page))
        return "\n".join(parts).strip()

    def _page_text(self, page) -> str:
        # blocks: (x0, y0, x1, y1, text, block_no, block_type); type 0 = text
        blocks = [b for b in page.get_text("blocks") if len(b) >= 7 and b[6] == 0 and b[4].strip()]
        if not blocks:
            return ""
        ordered = self._reading_order(blocks, page.rect.width)
        return "\n".join(b[4].strip() for b in ordered)

    @staticmethod
    def _reading_order(blocks: list, page_width: float) -> list:
        """Column-aware sort so two-column CVs keep reading order.

        Heuristic: if content splits cleanly into a left and a right column with a
        near-empty gutter down the middle, read the whole left column then the whole
        right column. Otherwise fall back to a plain top-to-bottom, left-to-right sort.
        """
        mid = page_width / 2
        left = [b for b in blocks if (b[0] + b[2]) / 2 < mid]
        right = [b for b in blocks if (b[0] + b[2]) / 2 >= mid]
        crossing = [b for b in blocks if b[0] < mid < b[2]]  # blocks spanning the gutter

        two_column = (
            len(left) >= 2
            and len(right) >= max(2, 0.2 * len(blocks))
            and len(crossing) <= max(1, 0.1 * len(blocks))
        )
        if two_column:
            key = lambda b: (round(b[1], 1), b[0])  # noqa: E731
            return sorted(left, key=key) + sorted(right, key=key)
        return sorted(blocks, key=lambda b: (round(b[1], 1), b[0]))

    # ------------------------------------------------------------- experience
    def _segment_experiences(self, raw_text: str) -> list[Experience]:
        lines = [line.rstrip() for line in raw_text.splitlines()]
        section = self._experience_section(lines)
        # No whole-document retry after an explicit empty experience section:
        # education dates must never become employment.
        return [
            exp
            for entry in self._split_entries(section)
            if (exp := self._build_experience(entry)) is not None
        ]

    @staticmethod
    def _experience_section(lines: list[str]) -> list[str]:
        explicit = any(_EXPERIENCE_HEADER_RE.match(line.strip()) for line in lines)
        current = not explicit
        exp = []
        for line in lines:
            if _EXPERIENCE_HEADER_RE.match(line.strip()):
                current = True
            elif _OTHER_SECTION_RE.match(line.strip()):
                current = False
            elif current:
                exp.append(line)
        return exp

    @staticmethod
    def _split_entries(section: list[str]) -> list[list[str]]:
        # Join date-only fragments split by PDF line wrapping, without joining titles.
        lines = []
        index = 0
        while index < len(section):
            line = section[index].strip()
            if re.match(rf"^{_DATE}\s*(?:-|–|—|to)?$", line, re.IGNORECASE):
                for count in (3, 2):
                    candidate = " ".join(s.strip() for s in section[index : index + count])
                    if _DATE_RANGE_RE.fullmatch(candidate):
                        line = candidate
                        index += count - 1
                        break
            lines.append(line)
            index += 1

        entries, current = [], []
        for line in lines:
            if not line:
                continue  # blank PDF blocks do not end a dated job prematurely
            dated = any(_DATE_RANGE_RE.search(item) for item in current)
            titled = any(occupation_title(item) for item in current)
            if dated and (_DATE_RANGE_RE.search(line) or (titled and occupation_title(line))):
                entries.append(current)
                current = []
            current.append(line)
        if current:
            entries.append(current)
        return entries

    def _build_experience(self, entry: list[str]) -> Experience | None:
        text = "\n".join(entry)
        date_match = _DATE_RANGE_RE.search(text)
        if date_match is None:
            return None
        title = next((title for line in entry if (title := occupation_title(line))), None)
        # Unknown dated entries are not reliable work history (e.g. headerless degrees).
        if title is None:
            return None
        clean = [_DATE_RANGE_RE.sub("", line).strip(" -–—|,:\t") for line in entry]
        organisation = self._first_org(text)
        description = " ".join(
            line
            for line in clean
            if line and occupation_title(line) != title and line != organisation
        )
        return Experience(
            title=title,
            organisation=organisation,
            start=date_match.group(1).strip(),
            end=date_match.group(2).strip(),
            description=description or None,
        )

    def _first_org(self, text: str) -> str | None:
        if self._nlp is None:
            return None
        for ent in self._nlp(text).ents:
            if ent.label_ == "ORG" and occupation_title(ent.text) is None:
                return ent.text.strip()
        return None

    def _match_skills(self, raw_text: str, skill_dictionary: list[str] | None = None) -> list[str]:
        dictionary = self._skill_dictionary if skill_dictionary is None else skill_dictionary
        matched = {}
        for label in dictionary:
            term = normalize_term(label)
            if term and term not in matched:
                evidence = find_evidence(term_pattern(term), term, raw_text)
                if evidence:
                    matched[term] = evidence
        return sorted(matched.values(), key=str.casefold)

    @staticmethod
    def _redact_pii(text: str) -> str:
        text = _EMAIL_RE.sub("[email]", text)
        dates = [match.span() for match in _DATE_RANGE_RE.finditer(text)]

        def redact_phone(match):
            if any(start < match.end() and end > match.start() for start, end in dates):
                return match.group()  # preserve 2019 - 2024 and similar ranges
            return "[phone]"

        return _PHONE_RE.sub(redact_phone, text)
