# ReRouteHer System (Backend)

FastAPI backend for the ReRouteHer Iteration 1 guest journey: CV parse, skill snapshot, and readiness/gap.

## Backend safeguard test branch

This branch proposes six-digit MASCO eligibility, honest missing-data scoring,
literal skill evidence, stable IDs and privacy-safe request logging. **No database
schema/data changes or deployment are included.** It has API behavior changes.
Read [the review and testing guide](docs/BACKEND_TEST_BRANCH.md) before trying it;
the older bootstrap database below lacks the required approved D13 metadata.

This is a Monash FIT5120 academic project. See `plan.md` for the full design derived from the user stories and data-governance deliverables.

## Endpoints (It1)

| Endpoint | Story | Job |
|---|---|---|
| `POST /api/cv/parse` | E2 / US2.1 | PDF to structured CV (NLP extractor) |
| `POST /api/snapshot/generate` | E3 / US3.1-3.4 | skills + previous occupation + recommended roles |
| `POST /api/gap/compute` | E4 / US4.1-4.3 | readiness % + merged top-3 gap |

All three are stateless and computed at request time. Guests persist nothing.

## Run with Docker (primary path)

The whole stack is containerized: a pgvector Postgres and the FastAPI API.

```bash
./scripts/local-start.sh     # build + start, waits for DB and API health, prints a banner
./scripts/local-stop.sh      # stop and reset the DB (add --keep-data to preserve it)
```

If port 5432 is taken on your host: `DB_PORT=5433 ./scripts/local-start.sh`.

Under the hood this is just:

```bash
docker compose up --build
```

On first boot the `db` service auto-loads `db/00_import.sql` (reference data) then `db/01_pgvector_migrate.sql` (converts the embedding columns to pgvector `vector(384)` and adds cosine indexes). The API waits for the DB healthcheck before starting.

- API: http://localhost:8080/docs (or `/api/health`)
- Postgres: localhost:5432 (db `rerouteher`, user/pass `postgres`)

The TF-IDF occupation model lives at `ml/tfidf_logreg.joblib` (produced by the data team) and is mounted read-only into the API container. The app boots without it (occupation falls back to embedding).

To re-run DB init from scratch: `docker compose down -v` (wipes the `pgdata` volume) then `up` again.

## Run locally without Docker (optional)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip3 install -r requirements.txt
python3 -m spacy download en_core_web_sm
cp .env.example .env   # edit DATABASE_URL to your local Postgres

psql -v ON_ERROR_STOP=1 -d rerouteher -f db/00_import.sql
psql -v ON_ERROR_STOP=1 -d rerouteher -f db/01_pgvector_migrate.sql

uvicorn app.main:app --reload --port 8080
```

## Test

```bash
pytest
```

## Status

All three It1 endpoints are implemented and unit-tested:

- **`/api/cv/parse`** - column-aware text, conservative dated employment extraction, literal skill mentions, best-effort email/phone redaction, no OCR.
- **`/api/snapshot/generate`** - literal canonical/alias evidence, CV-stated occupation with separate ESCO comparison, 0–3 eligible six-digit MASCO recommendations, canonical break skill IDs.
- **`/api/gap/compute`** - exact-ID requirement coverage, explicit not-assessed responses, true requirement counts and top-three focus gaps.

The app boots even when models are absent (embedder/classifier optional) so the frontend can integrate against the schemas. `pytest` runs the fast suite without a DB or torch (repos/models faked for the snapshot); the real DB + model path is exercised by running the container.

## Not in It1

Employer Fit (E4 / US4.4-4.5) is deferred to Iteration 2. The `employers` and `role_sector_map` tables import but are not queried yet.
