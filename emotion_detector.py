import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from deepface import DeepFace

def detect_mood(image_path: str):
    """
    Detects the dominant emotion in the image at `image_path`.
    Uses DeepFace analyze without preloading models (slower, compatible with older versions).
    """
    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )

        # DeepFace can return a list in some versions
        if isinstance(result, list):
            result = result[0]

        dominant_emotion = result['dominant_emotion']
        return {"emotion": dominant_emotion}

    except Exception as e:
        return {"error": str(e)}
