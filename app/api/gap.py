"""POST /api/gap/compute."""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.schemas.gap import GapRequest, GapResponse

router = APIRouter(prefix="/api/gap", tags=["gap"])


@router.post(
    "/compute",
    response_model=GapResponse,
    responses={404: {"model": GapResponse}, 409: {"model": GapResponse}},
)
async def compute_gap(
    req: GapRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    service = request.app.state.gap_service
    result = await service.compute(req, session)
    if result.assessment_status == "not_assessed":
        # Existing clients must see an error, not coerce null into 0% or show 100%.
        return JSONResponse(
            status_code=404 if result.reason == "unknown_role" else 409,
            content={**result.model_dump(), "error": result.reason},
        )
    return result
