import warnings

warnings.filterwarnings("ignore")


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from deepface import DeepFace

def detect_mood(image_path: str):
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'])


        if isinstance(result, list):
            result = result[0]

        dominant_emotion = result['dominant_emotion']
        print(f"Mood Detected: {dominant_emotion}")
        return dominant_emotion

    except Exception as e:
        print(f"Error: {e}")
        return None







