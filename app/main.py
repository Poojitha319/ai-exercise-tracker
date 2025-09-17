import os
import logging
from fastapi import FastAPI
from dotenv import load_dotenv
from app.backend.routers import nutrition,meals,workouts,exercises,health


# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the FastAPI app
app = FastAPI(
    title="Exercise Tracker API",
    description="An AI-powered fitness application for coaches and athletes.",
    version="1.0.0",
)

# Include routers for different endpoints
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(nutrition.router, prefix="/nutrition-plans", tags=["nutrition"])
app.include_router(meals.router, prefix="/meals", tags=["meals"])
app.include_router(workouts.router, prefix="/workouts", tags=["workouts"])
app.include_router(exercises.router, prefix="/detect_exercise", tags=["exercises"])



# Define a root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to Exercise Tracker API"}

