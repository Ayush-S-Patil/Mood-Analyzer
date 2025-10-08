from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import os
from emotion_detector import detect_mood

app = FastAPI(title="Mood Detection API")

@app.get("/")
def home():
    return {"message": "Upload an image at /predict to detect mood"}

@app.post("/predict/")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        # Determine file suffix based on original filename
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

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(await file.read())
            temp_path = temp.name

        # Call the emotion detector
        result = detect_mood(temp_path)

        # Delete temp file
        os.remove(temp_path)

        return JSONResponse(content=result, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
