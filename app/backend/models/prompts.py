# app/models/prompts.py




# -----------------------------Meal Analysis Prompt-----------------------------
ANALYZE_MEAL_PROMPT = """
Analyze the following meal image and provide the name of the food, 
total calorie count, and calories per ingredient. 
Respond in the following JSON format:
{
  "food_name": "<food name>",
  "total_calories": <total calorie count>,
  "calories_per_ingredient": {
    "<ingredient1>": <calories>,
    "<ingredient2>": <calories>
  }
}
"""



# -----------------------------Workout Plan Prompt-----------------------------

def workout_plan_prompt(profile_data):
    return f"""
Create a workout plan for a {profile_data['age']} year old {profile_data['sex']}, 
weighing {profile_data['weight']}kg and {profile_data['height']}cm tall, 
with the goal of {profile_data['goal']}. 
The workout plan should include {profile_data['workouts_per_week']} sessions per week. 

Focus exclusively on safe, appropriate, and positive exercise recommendations. 
Avoid any mention of sensitive or controversial topics. 

Respond strictly in valid JSON format:
{{
  "warmup": {{"description": "<description>", "duration": <minutes>}},
  "cardio": {{"description": "<description>", "duration": <minutes>}},
  "sessions_per_week": <sessions>,
  "workout_sessions": [
    {{
      "exercises": [
        {{"name": "<exercise name>", "sets": <sets>, "reps": "<reps>", "rest": <rest seconds>}}
      ]
    }}
  ],
  "cooldown": {{"description": "<description>", "duration": <minutes>}}
}}
"""




# -----------------------------Nutrition Plan Prompt-----------------------------
def nutrition_plan_prompt(profile_data):
    return f"""
      Provide a personalized nutrition plan for a {profile_data['age']} year old {profile_data['sex']}, 
      weighing {profile_data['weight']}kg, height {profile_data['height']}cm, with the goal of {profile_data['goal']}.

      The plan must include:
      - Daily calorie intake range.
      - Macronutrient distribution in grams (protein, carbs, fat).
      - A meal plan (breakfast, lunch, dinner, snacks) with 3 options each.
      - Each meal must have description, ingredients with quantities, calories per ingredient, total calories, and recipe.

      Respond strictly in valid JSON format:
      {{
        "daily_calories_range": {{"min": <min>, "max": <max>}},
        "macronutrients_range": {{
          "protein": {{"min": <min>, "max": <max>}},
          "carbohydrates": {{"min": <min>, "max": <max>}},
          "fat": {{"min": <min>, "max": <max>}}
        }},
        "meal_plan": {{
          "breakfast": [...],
          "lunch": [...],
          "dinner": [...],
          "snacks": [...]
        }}
      }}
      """




# -----------------------------Exercise Detection Prompt-----------------------------
EXERCISE_DETECTION_PROMPT = """
        You are a fitness coach AI. I will provide you with a youtube url link of a workout video.
        Your task is to analyze the video and return the results in structured JSON.

        Instructions:
        1. Detect the type of exercise being performed (e.g., push-up, squat, sit-up, bicep curl).
        2. Count the total number of repetitions completed.
        3. Analyze the user’s posture and form. Provide constructive feedback:
        - If correct, say “Good form”.
        - If incorrect, specify what needs improvement.
        4. Suggest 1 short improvement tip for better performance.

        Output format (JSON):
        {
            "exercise_type": "<name>",
            "repetitions": <number>,
            "form_feedback": "<feedback on posture>",
            "improvement_tip": "<suggestion>"
        }
        """
# -----------------------------Real-time Suggestion Prompt-----------------------------
SUGGESTION_PROMPT = """
        You are a fitness trainer AI. A user is doing {action}.
        - Total reps completed so far: {reps}
        - Last rep duration: {time_per_rep:.2f} seconds

        Based on this info:
        1. Give 1 short motivational feedback (e.g., "Great job on form!", "Push a little slower").
        2. Keep it under 15 words.
        3. Do NOT include JSON, just plain text feedback.
        """