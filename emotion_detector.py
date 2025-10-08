import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from deepface import DeepFace

# ✅ Preload the emotion model once at startup
emotion_model = DeepFace.build_model("Emotion")

def detect_mood(image_path: str):
    try:
        # Use OpenCV backend and enforce_detection=False
        # Pass the preloaded model directly in analyze function using model_name param
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )

        if isinstance(result, list):
            result = result[0]

        dominant_emotion = result['dominant_emotion']
        return {"emotion": dominant_emotion}

    except Exception as e:
        return {"error": str(e)}
