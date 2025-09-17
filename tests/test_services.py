# still need to test some of the gemini model functions
import sys
import os
import logging
from app.backend.models.gemini_model import GeminiModel

# Make sure Python can find your app folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example workout video
    youtube_url = "https://www.youtube.com/watch?v=BlCH0o8szoA"

    print("Downloading and analyzing workout video...")
    result = GeminiModel.detect_exercise_from_youtube(
        youtube_url,
        save_path="test_video.mp4"   # will download into your project folder
    )

    print("\n=== Gemini Exercise Detection Result ===")
    if result:
        print(result)
    else:
        print("❌ No result returned. Check logs above for errors.")
