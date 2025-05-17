from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
from PIL import Image
import io

app = FastAPI()

# CORS for all origins (not recommended for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained model once at startup
with open("app/digit_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = img.resize((8, 8))
        img_array = np.array(img)

        # Optional: scale to 0-16 (as in sklearn digits dataset)
        if img_array.max() > 16:
            img_array = (img_array / 255.0) * 16

        img_array = img_array.reshape(1, -1)
        prediction = model.predict(img_array)

        return {
            "success": True,
            "prediction": int(prediction[0]),
            "confidence": 1.0  # placeholder confidence
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
