"""ReRouteHer backend. Guest journey: CV parse -> snapshot -> gap.

Models and reference-derived assets load once at startup (lifespan) and live on
app.state so requests have no cold start. Model loading is resilient: if an ML asset
is missing, the app still boots and the affected endpoint degrades at call time.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import cv, gap, snapshot
from app.config import get_settings
from app.core.logging import RequestLoggingMiddleware, configure_logging
from app.services.cv_extractor import CVExtractor
from app.services.embedder import Embedder
from app.services.gap import GapService
from app.services.occupation_matcher import EscoTfidfMatcher
from app.services.skill_catalog import SkillCatalog
from app.services.snapshot import SnapshotService

configure_logging()
logger = logging.getLogger("rerouteher")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # 1. Embedder (all-MiniLM-L6-v2, CPU). Optional so the app boots without the model cache.
    embedder = None
    try:
        embedder = Embedder(settings.embedding_model)
    except Exception:  # noqa: BLE001
        logger.warning("Embedder not loaded; vector role recommendations disabled")

    # 2. Optional ESCO comparison model; never a six-digit MASCO identity lookup.
    try:
        tfidf_matcher = EscoTfidfMatcher.load(settings.tfidf_model_path)
    except Exception:
        logger.warning("TF-IDF model unavailable; ESCO comparison disabled")
        tfidf_matcher = None
    if tfidf_matcher is None:
        logger.warning("ESCO comparison model unavailable")

    # 3. spaCy is optional and independent of the database skill dictionary.
    nlp = None
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
    except Exception:  # noqa: BLE001
        logger.warning("spaCy not loaded; organisation extraction degraded")

    skill_catalog = SkillCatalog(settings.skill_cache_ttl_seconds)

    # Parser and snapshot share one TTL-refreshed canonical skill catalog.
    app.state.cv_extractor = CVExtractor(nlp=nlp, skill_dictionary=[], embedder=embedder)
    app.state.snapshot_service = SnapshotService(
        settings=settings,
        embedder=embedder,
        tfidf_matcher=tfidf_matcher,
        skill_catalog=skill_catalog,
    )
    app.state.gap_service = GapService(settings=settings)

    logger.info(
        "startup: embedder=%s tfidf=%s spacy=%s",
        embedder is not None,
        tfidf_matcher is not None,
        nlp is not None,
    )

    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="ReRouteHer API", version="0.2.1", lifespan=lifespan)

    # Allow the browser client to call the API directly.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(RequestLoggingMiddleware)

    app.include_router(cv.router)
    app.include_router(snapshot.router)
    app.include_router(gap.router)

    @app.get("/api/health", tags=["meta"])
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
