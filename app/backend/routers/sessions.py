from fastapi import APIRouter, File, UploadFile, HTTPException
from app.backend.services.session_service import analyze_session
from app.backend.schemas.session import SessionResult

router = APIRouter()

@router.post(
    "/analyze_session",
    response_model=SessionResult,
    summary="Analyze Exercise Session Video",
    description="Upload a video to get exercise type and repetition counts."
)
async def analyze_session_endpoint(file: UploadFile = File(...)):
    try:
        video_bytes = await file.read()
        result = analyze_session(video_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))