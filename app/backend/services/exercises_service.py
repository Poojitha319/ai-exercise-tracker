from app.backend.models.gemini_model import GeminiModel
from app.backend.models.exercise import Exercise
from app.database import SessionLocal
from fastapi import HTTPException
import logging
import json

def detect_exercise_from_youtube(youtube_url: str) -> dict:
    logging.info("Starting exercise detection from YouTube URL")
    try:
        result_text = GeminiModel.detect_exercise_from_youtube(youtube_url)
        if not result_text:
            raise HTTPException(status_code=500, detail="No response from Gemini API")

        logging.info(f"Gemini API Response Text (Exercise Detection): {result_text}")

        clean_result_text = result_text.strip("```json\n").strip("```")
        result = json.loads(clean_result_text)

        exercise_type = result.get("exercise_type")
        repetitions = result.get("repetitions")
        form_feedback = result.get("form_feedback")
        improvement_tip = result.get("improvement_tip")

        if not all([exercise_type, repetitions, form_feedback, improvement_tip]):
            raise HTTPException(
                status_code=500,
                detail="Missing fields in Gemini response"
            )

        # save to DB if not already there
        db = SessionLocal()
        existing = db.query(Exercise).filter(Exercise.name.ilike(exercise_type)).first()
        if not existing:
            new_exercise = Exercise(
                name=exercise_type,
                source_url=youtube_url,
                is_predefined=False
            )
            db.add(new_exercise)
            db.commit()
            db.refresh(new_exercise)
            logging.info(f"Added new exercise: {exercise_type}")
        else:
            logging.info(f"Exercise already exists: {exercise_type}")

        db.close()

    except Exception as e:
        logging.error(f"Exception (Exercise Detection): {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "exercise_type": exercise_type,
        "repetitions": repetitions,
        "form_feedback": form_feedback,
        "improvement_tip": improvement_tip,
    }
