from pydantic import BaseModel
from typing import Optional

class ExerciseDetectionResult(BaseModel):
    exercise_type: str
    repetitions: int
    form_feedback: str
    improvement_tip: str

class ExerciseDB(BaseModel):
    id: int
    name: str
    source_url: Optional[str] = None
    is_predefined: bool

    class Config:
        from_attributes = True
        orm_mode = True