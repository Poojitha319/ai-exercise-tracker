from pydantic import BaseModel

class SessionResult(BaseModel):
    curl_reps: int
    press_reps: int
    squat_reps: int
    detected_action: str