from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from app.backend.services.exercises_service import detect_exercise_from_youtube
from app.backend.schemas.exercises import ExerciseDetectionResult, ExerciseDB
from app.database import SessionLocal
from app.backend.models.exercise import Exercise

router = APIRouter()

# DB dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post(
    "/detect_exercise",
    response_model=ExerciseDetectionResult,
    summary="Detect Exercise from YouTube URL",
)
async def detect_exercise_endpoint(
    youtube_url: str = Query(..., description="YouTube video URL")
):
    try:
        result = detect_exercise_from_youtube(youtube_url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# New Endpoint → List all exercises
@router.get(
    "/list",
    response_model=list[ExerciseDB],
    summary="Get all exercises (predefined + new)",
)
async def list_exercises(db: Session = Depends(get_db)):
    try:
        exercises = db.query(Exercise).all()
        return exercises
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
