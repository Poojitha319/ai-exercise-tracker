# app/models/gemini_models.py

import os
import logging
from PIL import Image
from io import BytesIO
from pytube import YouTube
import google.generativeai as gemini

# Import prompts
from app.backend.models.prompts import (
    ANALYZE_MEAL_PROMPT,
    workout_plan_prompt,
    nutrition_plan_prompt,
    EXERCISE_DETECTION_PROMPT,
    SUGGESTION_PROMPT,
)

# Setup Gemini API key
api_key = os.getenv("GEMINI_API_KEY")
gemini.configure(api_key=api_key)

# Initialize Gemini model
MODEL_NAME = "gemini-1.5-flash"
model = gemini.GenerativeModel(MODEL_NAME)


class GeminiModel:
    @staticmethod
    def analyze_meal(image_data):
        """Analyze meal image and return JSON with calories."""
        try:
            image = Image.open(BytesIO(image_data))
            response = model.generate_content([ANALYZE_MEAL_PROMPT, image])
            logging.info(f"Analyze Meal Response: {response}")
            return response.text
        except Exception as e:
            logging.error(f"Analyze Meal Error: {str(e)}")
            return None

    @staticmethod
    def generate_workout_plan(profile_data):
        """Generate workout plan based on user profile."""
        try:
            prompt = workout_plan_prompt(profile_data)
            response = model.generate_content(prompt)
            logging.info(f"Workout Plan Response: {response}")
            return response.text
        except Exception as e:
            logging.error(f"Workout Plan Error: {str(e)}")
            return None

    @staticmethod
    def generate_nutrition_plan(profile_data):
        """Generate nutrition plan based on user profile."""
        try:
            prompt = nutrition_plan_prompt(profile_data)
            response = model.generate_content(prompt)
            logging.info(f"Nutrition Plan Response: {response}")
            return response.text
        except Exception as e:
            logging.error(f"Nutrition Plan Error: {str(e)}")
            return None

    @staticmethod
    def detect_exercise(video_data):
        """Analyze uploaded workout video for reps, posture, and tips."""
        try:
            response = model.generate_content([EXERCISE_DETECTION_PROMPT, video_data])
            logging.info(f"Exercise Detection Response: {response}")
            return response.text
        except Exception as e:
            logging.error(f"Exercise Detection Error: {str(e)}")
            return None

    @staticmethod
    def generate_workout_feedback(action, reps, time_per_rep):
        """Generate short motivational feedback during exercise."""
        try:
            prompt = SUGGESTION_PROMPT.format(action=action, reps=reps, time_per_rep=time_per_rep)
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logging.error(f"Workout Feedback Error: {str(e)}")
            return "Keep going strong!"

    @staticmethod
    def detect_exercise_from_youtube(youtube_url, save_path):
        """Download YouTube workout video, analyze with Gemini."""
        try:
            # Step 1: Download YouTube video
            yt = YouTube(youtube_url)
            stream = yt.streams.filter(file_extension="mp4", progressive=True).first()
            stream.download(filename=save_path)
            logging.info(f"Downloaded YouTube video to {save_path}")

            # Step 2: Upload video to Gemini
            uploaded_file = gemini.upload_file(path=save_path)

            # Step 3: Ask Gemini to analyze
            response = model.generate_content([EXERCISE_DETECTION_PROMPT, uploaded_file])
            logging.info(f"YouTube Exercise Detection Response: {response}")
            return response.text
        except Exception as e:
            logging.error(f"YouTube Exercise Detection Error: {str(e)}")
            return None
