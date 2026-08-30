# Backend safeguards proposal — test branch

Branch: **fix/backend-readiness-safeguards**

Baseline: 23ed6b88a54be269449715656cc51d21c8e8af3b (main, 30 August 2026)

This is a review/test proposal, not a production deployment. It changes backend code
and API behavior only. No database migration, table/column changes, data updates,
frontend changes, model retraining, or real résumé uploads are included.

## Changes

| Failure | New behavior |
|---|---|
| No requirements scored 100% | No score; explicit not_assessed response |
| Missing digital band gave free 20–60% credit | Renormalize across populated bands and warn about incomplete references |
| Shared ESCO code selected first MASCO row | Recommend eligible six-digit MASCO roles directly; keep ESCO comparison |
| Four-digit group resolved to first six-digit role | Prefix fallback removed |
| Pending/Low roles appeared as targets | Same eligibility predicate for exact-title, vector and gap paths |
| Weak matches filled all three slots | Return 0–3 roles, with actual similarity and method |
| Recommendation became previous occupation | Preserve résumé-stated title; separate optional ESCO comparison |
| Fuzzy/semantic guesses became verified skills | Only literal canonical/alias evidence is auto-credited |
| Short aliases matched ordinary words | Context/case guards for DID, CTA, C, R, Go, Swift and short terms |
| Labels stayed stale after an import | Shared TTL-refreshed catalog; ambiguous aliases resolve to neither ID |
| Break skills had no canonical IDs | Reuse existing caregiving_map.onet_skill_id |
| Company placeholders/dates/degrees became jobs | Conservative dated employment extraction; no education fallback |
| Chunked JSON requests were truncated | Forward the original ASGI receive/send streams |
| Logs contained résumé bodies | Log method, route template, status and duration only |

Coverage and similarity are **not calibrated employment-readiness probabilities**.
Full coverage means all requirements in this particular reference profile matched,
not that every aspect of the job has been assessed.

## Existing database contract

The app still reads the existing rerouteher schema and pgvector columns. It performs
no database writes. Default configuration:

    MAPPING_METADATA_PREFIX=d13_quality_fix_20260830.mapping.

The existing metadata key must be exactly this prefix plus role ID. The metadata
must agree with the role's ID, six-digit MASCO code and ESCO code, with:

- mapping_confidence: medium or high;
- use_in_role_skills: true;
- review_status: auto_approved_medium_or_high_test_mapping, approved, or manually_approved.

Targets must also be flexible, have a valid AI-exposure band, and have nonempty,
valid core requirements. Pending, Low, unversioned and legacy four-digit roles stay
in the database but are excluded from this pool. Missing/stale metadata fails closed.
This gate is not independent validation of the dataset's remote-work rating.

Do not fall back to an older approved metadata row. When the data team releases a
new namespace, change the setting explicitly and rerun tests.

**The old 10-role db/00_import.sql bootstrap is not the approved D13 release.**
It will not satisfy this gate. Do not import it over your current test database.

Different MASCO roles may legitimately share an ESCO code. The branch does not merge
those roles, union multiple ESCO candidates' skill sets, or merge distinct skill
concepts solely because their names are similar.

## API changes requiring frontend review

### Snapshot

POST /api/snapshot/generate accepts the existing request shape.

- Recommendations add role_id, six-digit masco_code, esco_code and method
  (exact_title or embedding). They can contain fewer than three roles, including none.
- Similarity is the real cosine score. Exact-title agreement uses 1.0, which is
  label agreement, not a suitability confidence.
- Previous occupation uses method "cv_title" and confidence null. It is the first
  reliable résumé-stated title in CV order, not independently verified history.
  An out-of-catalog job is not replaced with a STEM recommendation.
- Optional esco_code, esco_title, comparison_score and comparison_method describe
  a separate ESCO comparison. Classifier probability and retrieval similarity have
  separate thresholds.
- Professional skills include literal surface evidence and evidence_type "literal".
  Old ungrounded skill_mentions and semantic/fuzzy suggestions are not auto-credited.
  A literal mention is still not proof of proficiency.
- Break skills add canonical skill_id. The skill field now holds the canonical
  name; the friendly wording remains in reframed_label.
- reference_version is the configured mapping namespace; warnings report limitations.

### Gap

Prefer the stable role ID returned by snapshot:

    {
      "target_role_id": "M251116",
      "reference_version": "d13_quality_fix_20260830.mapping.",
      "skills": [{
        "skill": "<canonical name from snapshot>",
        "skill_id": "<canonical ID from snapshot>",
        "source": "experience",
        "evidence_type": "literal"
      }]
    }

Legacy target_role titles are supported when unique. An explicit target_role_id takes
precedence. There is no fuzzy title resolution.

Successful assessments retain numeric readiness, skills_have and top-three gaps.
They add assessment_status "assessed", required_skill_count, matched_skill_count,
missing_skill_count, identity fields, reference_version, gap skill_id and warnings.
**Use the count fields, not matched-list length plus three focus gaps.**

Unknown roles return HTTP 404. Non-assessable profiles/evidence return HTTP 409 with
assessment_status "not_assessed", readiness null, reason and error. Show a
not-assessed state, not 0%, 100%, “ready today,” or a skill-deficit conclusion.

An empty/unrecognized skill list is insufficient evidence. Recognized skills with
zero requirement overlap are genuinely 0% coverage. Duplicate IDs count once.
Unknown IDs and name-only claims are ignored with warnings. Explicit
semantic/inferred claims need confirmed true to count.

Legacy payloads without evidence metadata remain caller-supplied skill claims;
this stateless API cannot authenticate résumé evidence. Clear old browser diagnostic
state before testing. The optional namespace check is not a hash of all reference
tables and cannot detect every in-place edit or stale legacy inference.

### Missing-band proposal

Both populated bands retain the original exposure-weighted blend. With one band,
that band gets 100% of the available reference weight, with a missing-band warning.
With neither, no score is produced. Missing requirements are not assumed irrelevant
or complete. The team should approve this product decision before merging.

## Run safely

Use a separate checkout and API process, not the deployed app:

    git fetch origin
    git switch fix/backend-readiness-safeguards
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-test.txt
    python -m pytest -q
    python -m ruff check app tests

The fast suite mocks models/repositories. Optional actual SQL/vector regressions use
an isolated, in-memory [PGlite engine with pgvector](https://pglite.dev/extensions/):

    npm install --prefix tests/sql --ignore-scripts --no-audit --no-fund
    PGLITE_DEPS_DIR="$PWD/tests/sql" python -m pytest -q

These tests never use DATABASE_URL. They create synthetic tables in memory only.
Node packages are test-only dependencies, not part of the backend runtime.

For interactive evaluation:

1. Install requirements.txt and the spaCy model.
2. Retrieve repository model assets with Git LFS if your checkout has pointer files.
   Confirm the trusted joblib artifact and vendored embedding model are available.
   Missing models degrade to exact-title-only recommendations.
3. Copy .env.example to an untracked .env. Point DATABASE_URL at a separate test
   database already containing the approved MASCO 2020 D13 release. Prefer a
   SELECT-only account. Do not run import/reset scripts.
4. Set SKILL_CACHE_TTL_SECONDS=0 while checking an import; use 60 normally.
5. Run: uvicorn app.main:app --host 127.0.0.1 --port 8081
6. Open http://127.0.0.1:8081/docs. For UI evaluation, point a separate frontend
   build at this API and adapt the status/count/identity behavior above.

Check: approved software role, pending Village Community Center Manager target
(no score), out-of-catalog teacher, no employment, no skills, valid unrelated skills,
all required IDs and a one-band profile.

## Validation and remaining work

Tests cover synthetic unit/API cases and actual PostgreSQL/pgvector query execution.
No personal résumé content or private credentials are committed.

Local validation on 30 August 2026: **95 passed** (74 fast tests plus 21 optional
PostgreSQL/pgvector tests), Ruff passed, Python compilation passed, and git whitespace
checks passed. Python 3.12 was used. One non-failing Starlette/httpx deprecation
warning remains. No database SQL, model assets or trained model files were modified.

Before production the team must still:

- replay the 25-résumé evaluation on staging with the actual full models and dataset,
  and manually assess relevance;
- calibrate retrieval thresholds; defaults are conservative test settings, not
  validated accuracy guarantees;
- review Medium/broader-proxy requirements, missing bands, flat importance weights
  and inherited remote/AI ratings;
- review extraction recall, negated/ambiguous mentions, unusual titles, chronology
  (currently CV order), and complex PDF layouts;
- integrate frontend contract changes and stale-state handling;
- verify native PostgreSQL/asyncpg connectivity and Docker/model startup: the
  in-memory harness is not a full deployment test.

Low scores can remain when canonical requirements do not match résumé evidence.
This branch does not inflate scores or approve missing dataset mappings.
