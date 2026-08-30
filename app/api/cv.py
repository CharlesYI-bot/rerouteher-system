"""POST /api/cv/parse."""

import logging

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db import get_session
from app.schemas.cv import CVParseResponse
from app.services.cv_extractor import UnreadableCVError

router = APIRouter(prefix="/api/cv", tags=["cv"])


@router.post("/parse", response_model=CVParseResponse, responses={400: {"model": dict}})
async def parse_cv(request: Request, file: UploadFile = File(...), session=Depends(get_session)):
    settings = get_settings()

    if file.content_type != "application/pdf":
        return JSONResponse(status_code=400, content={"error": "PDF only"})

    data = await file.read(settings.max_cv_bytes + 1)
    if len(data) > settings.max_cv_bytes:
        return JSONResponse(status_code=400, content={"error": "file too large (max 10MB)"})

    extractor = request.app.state.cv_extractor
    catalog_unavailable = False
    try:
        catalog = await request.app.state.snapshot_service.skill_catalog.get(session)
        terms = list(catalog.terms)
    except Exception:
        logging.getLogger("rerouteher").warning("Skill catalog unavailable during CV parsing")
        terms = []
        catalog_unavailable = True
    try:
        cv = extractor.parse(data, skill_dictionary=terms)
    except UnreadableCVError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    if catalog_unavailable:
        cv.warnings.append("skill_catalog_unavailable")
    return CVParseResponse(cv=cv)
