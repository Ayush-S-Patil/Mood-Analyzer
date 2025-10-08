import os
from fastapi import FastAPI, UploadFile, File
from emotion_detector import detect_mood
from fastapi.responses import  JSONResponse
import tempfile

app = FastAPI(title="Mood Detection API")

@app.get("/")
def root():
    return "API is running"

@app.post("/predict/")
async def predict_emotion(file: UploadFile = File()):
    try:
        filename_lower = file.filename.lower()
        if filename_lower.endswith(".jpg", ".jpeg"):
            suffix = ".jpg"
        elif filename_lower.endswith(".png"):
            suffix = ".png"
        else:

            return JSONResponse(
            content={"error": "Unsupported file type. Use jpg, jpeg, or png."},
         status_code=400)


        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(await file.read())
            temp_path = temp.name


        result = detect_mood(temp_path)

        os.remove(temp_path)
        return JSONResponse(content={"emotion": result}, status_code=200)


    except Exception as e:
        return JSONResponse(content= {"error": str(e)}, status_code = 500)


