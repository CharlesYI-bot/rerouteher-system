"""Narrow candidate-retrieval variants, not crosswalks or identity equivalences.

Do not reuse the classifier's broader normalization (e.g. data scientist -> analyst).
Variants still resolve through the same approved six-digit MASCO eligibility gate.
"""

_VARIANTS = {
    "software engineer": ("Software Developer",),
    "senior software engineer": ("Software Developer",),
    "junior software engineer": ("Software Developer",),
    "software development engineer": ("Software Developer",),
}


def candidate_titles(title: str) -> tuple[str, ...]:
    return _VARIANTS.get(" ".join(title.casefold().split()), ())
