"""Synthetic text-layout and literal-evidence regressions."""

import pytest

from app.services.cv_extractor import CVExtractor, occupation_title


@pytest.fixture
def parser():
    return CVExtractor(nlp=None, skill_dictionary=[])


@pytest.mark.parametrize(
    "dates", ["2019 - 2024", "2019–2024", "Jan 2019 — Jun 2024", "01/2019 to 06/2024"]
)
def test_phone_redaction_preserves_employment_dates(parser, dates):
    text = parser._redact_pii(f"test@example.com +60 12-345 6789\nSoftware Engineer\n{dates}")
    assert dates in text
    assert "[phone]" in text
    assert "[email]" in text
    exps = parser._segment_experiences("Work Experience\n" + text)
    assert exps[0].title == "Software Engineer"


@pytest.mark.parametrize(
    "layout",
    [
        "Software Engineer\nCompany Name City, State\n2019 - 2024\nDeveloped services.",
        "Software Engineer\n\nCompany Name City, State 2019 - 2024\n\nDeveloped services.",
        "2019 - 2024\nSoftware Engineer\nCompany Name City, State\nDeveloped services.",
        "Software Engineer\nJan 2019 -\nJun 2024\nDeveloped services.",
        "Software Engineer\nJan 2019\n-\nJun 2024\nDeveloped services.",
        "Software Engineer Company Name City, State 2019 - 2024\nDeveloped services.",
    ],
)
def test_title_layouts_recover_role_not_company(parser, layout):
    exps = parser._segment_experiences(
        "Work Experience\n" + layout + "\nEducation\nBSc 2014 - 2018"
    )
    assert len(exps) == 1
    assert exps[0].title == "Software Engineer"
    assert exps[0].description and "Developed services" in exps[0].description


def test_blank_blocks_do_not_drop_job_description(parser):
    exps = parser._segment_experiences(
        "Experience\nSubstitute Teacher\n2019 - 2024\n\nProvided classroom instruction.\n\n"
        "English Teacher\n2015 - 2019\nManaged teaching resources.\n"
    )
    assert [e.title for e in exps] == ["Substitute Teacher", "English Teacher"]
    assert exps[0].description == "Provided classroom instruction."


@pytest.mark.parametrize(
    "text",
    [
        "Education\nBachelor of Science\nUniversity Example\n2015 - 2019",
        "Experience\nEducation\nMaster of Science\n2019 - 2021",
        "Bachelor of Science\n2015 - 2019",
    ],
)
def test_education_never_fallback_employment(parser, text):
    assert parser._segment_experiences(text) == []


@pytest.mark.parametrize(
    "title",
    ["Company Name City, State", "Acme Ltd", "Bachelor of Science", "Managed a team of engineers."],
)
def test_non_titles_rejected(title):
    assert occupation_title(title) is None


def test_long_and_punctuation_skills_keep_literal_evidence(parser):
    text = "Programming languages: C++, C#, .NET. Used Tools for Software Configuration Management."
    dictionary = [
        "C++",
        "C#",
        ".NET",
        "Tools for Software Configuration Management",
        "Lisp",
        "Swift",
    ]
    assert set(parser._match_skills(text, dictionary)) == {
        "C++",
        "C#",
        ".NET",
        "Tools for Software Configuration Management",
    }


@pytest.mark.parametrize(
    "text", ["I did things.", "DID many things.", "A swift response.", "Go to work."]
)
def test_ambiguous_language_and_acronym_aliases_need_context(parser, text):
    assert parser._match_skills(text, ["did", "swift", "go"]) == []


def test_contextual_did_and_swift_are_available(parser):
    assert parser._match_skills(
        "Configured PBX DID telephony. Swift programming.", ["did", "swift"]
    ) == ["DID", "Swift"]
