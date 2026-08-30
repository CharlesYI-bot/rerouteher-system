"""Tests for the CV extractor. Builds real PDFs in-memory with PyMuPDF.

spaCy is optional here (nlp=None): text extraction, regex segmentation, skill matching,
and PII redaction all work without it. ORG detection is covered separately when spaCy is present.
"""

import pymupdf
import pytest

from app.services.cv_extractor import CVExtractor, UnreadableCVError

SKILLS = ["project management", "stakeholder management", "budgeting", "python", "javascript"]


def _pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    return doc.tobytes()


def _extractor() -> CVExtractor:
    return CVExtractor(nlp=None, skill_dictionary=SKILLS)


def test_extracts_text_and_experiences():
    cv_text = (
        "Jane Doe\n"
        "jane.doe@example.com | +60 12-345 6789\n"
        "\n"
        "Work Experience\n"
        "Project Coordinator\n"
        "Acme Sdn Bhd\n"
        "Jan 2018 - Mar 2020\n"
        "Led project management and budgeting for cross-functional teams.\n"
        "\n"
        "Education\n"
        "BSc Computer Science, 2014\n"
    )
    cv = _extractor().parse(_pdf(cv_text))

    assert "Work Experience" in cv.raw_text
    assert len(cv.experiences) == 1
    exp = cv.experiences[0]
    assert exp.start == "Jan 2018"
    assert exp.end.lower() == "mar 2020"
    assert exp.title == "Project Coordinator"


def test_typo_is_not_promoted_to_verified_skill():
    cv_text = (
        "Work Experience\n"
        "Manager\n"
        "2019 - 2021\n"
        "Responsible for projct management and budgeting.\n"  # typo: projct
    )
    cv = _extractor().parse(_pdf(cv_text))
    assert "budgeting" in cv.skill_mentions  # exact
    assert "project management" not in cv.skill_mentions


def test_pii_is_redacted_from_raw_text():
    cv = _extractor().parse(
        _pdf(
            "Contact: jane.doe@example.com +60 12-345 6789\nWork Experience\nRole\n2020 - 2021\nDid things.\n"
        )
    )
    assert "jane.doe@example.com" not in cv.raw_text
    assert "[email]" in cv.raw_text
    assert "[phone]" in cv.raw_text


def test_scanned_pdf_is_unreadable():
    # a PDF page with no text layer (image-only) -> no extractable text
    doc = pymupdf.open()
    doc.new_page()  # blank, no text
    with pytest.raises(UnreadableCVError):
        _extractor().parse(doc.tobytes())
