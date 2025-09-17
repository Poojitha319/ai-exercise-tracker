from fastapi import APIRouter, HTTPException
from app.backend.schemas.nutrition import ProfileData, NutritionPlan
from app.backend.services.nutrition_service import generate_nutrition_plan

router = APIRouter()


@router.post("/generate", response_model=NutritionPlan)
def get_nutrition_plan(profile_data: ProfileData):
    try:
        return generate_nutrition_plan(profile_data)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))