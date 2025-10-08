from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import os
from deepface import DeepFace
import warnings
from concurrent.futures import ThreadPoolExecutor

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = FastAPI(title="Mood Detection API")

executor = ThreadPoolExecutor(max_workers=2)  # Background threads

def analyze_image(path: str):
    """
    Safely analyze image and return dominant emotion.
    Runs in a background thread to prevent blocking.
    """
    try:
        result = DeepFace.analyze(
            img_path=path,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'  # lighter and safer for Render
        )

        if isinstance(result, list):
            result = result[0]

        return {"emotion": result.get('dominant_emotion', 'unknown')}

    except Exception as e:
        # Return error without crashing the server
        return {"error": str(e)}

@app.get("/")
def home():
    return {"message": "Upload an image at /predict/ to detect mood"}

@app.post("/predict/")
async def predict_emotion(file: UploadFile = File(...)):
    if not file:
        return JSONResponse(content={"error": "No file uploaded"}, status_code=400)

    # Check file type
    filename_lower = file.filename.lower()
    if filename_lower.endswith((".jpg", ".jpeg")):
        suffix = ".jpg"
    elif filename_lower.endswith(".png"):
        suffix = ".png"
    else:
        return JSONResponse(
            content={"error": "Unsupported file type. Use jpg, jpeg, or png."},
            status_code=400
        )

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    # Run analysis in background thread
    future = executor.submit(analyze_image, temp_path)
    result = future.result()  # Wait for result safely

    # Delete temp file
    os.remove(temp_path)

    return JSONResponse(content=result)
