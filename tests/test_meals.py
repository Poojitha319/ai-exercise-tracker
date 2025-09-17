#sample meal analysis test with mocked Gemini API response
import json
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
import unittest

client = TestClient(app)

class TestMeals(unittest.TestCase):
    @patch("app.models.gemini_model.GeminiModel.analyze_meal")
    def test_analyze_meal(self, mock_analyze_meal):
        # Mock response from the Gemini API
        mock_response = {
            "food_name": "Breakfast Burrito",
            "total_calories": 540,
            "calories_per_ingredient": {
                "eggs": 140,
                "tortilla": 100,
                "cheese": 100,
                "sausage": 100,
            },
        }
        mock_analyze_meal.return_value = json.dumps(mock_response)

        # Test file
        with open(r"D:\new\new\exercise_tracker\tests\test_image.jpg", "rb") as image_file:
            response = client.post(
                "/meals/analyze",
                files={"file": ("test_image.jpg", image_file, "image/jpeg")},
            )

        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["food_name"], "Breakfast Burrito")
        self.assertEqual(data["total_calories"], 540)
        self.assertIn("eggs", data["calories_per_ingredient"])
        self.assertEqual(data["calories_per_ingredient"]["eggs"], 140)
        self.assertEqual(data["calories_per_ingredient"]["tortilla"], 100)
        self.assertEqual(data["calories_per_ingredient"]["cheese"], 100)
        self.assertEqual(data["calories_per_ingredient"]["sausage"], 100)

if __name__ == "__main__":
    unittest.main()