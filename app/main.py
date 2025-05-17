from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import joblib

# Load your trained model
model = joblib.load('digit_classifier.pkl')

app = FastAPI()

# CORS configuration (important for frontend-backend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Verify the file is an image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('L').resize((28, 28))
        img_array = np.array(image).reshape(1, -1) / 255.0

        # Make prediction
        prediction = int(model.predict(img_array)[0])
        confidence = float(np.max(model.predict_proba(img_array)))

        return {
            "success": True,
            "prediction": prediction,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "OK", "message": "Digit Classifier API is running"}
