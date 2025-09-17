from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_status():
    return {"status": "ok", "message": "Backend is running ✅"}
