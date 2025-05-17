from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pickle
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
with open("app/digit_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
        img = img.resize((8, 8))  # Resize to match training data
        img_array = np.array(img).reshape(1, -1)  # Flatten image

        # Normalize if needed
        if np.max(img_array) > 1:
            img_array = img_array / 255.0

        # Make prediction
        prediction = model.predict(img_array)[0]
        
        # Get confidence if supported
        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(img_array)))
        else:
            confidence = None  # fallback if model doesn't support probabilities

        return {
            "success": True,
            "prediction": int(prediction),
            "confidence": confidence
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
