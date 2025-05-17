from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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

# Load your pre-trained model
with open("app/digit_classifier_model.pkl", "rb") as f: 
    model = pickle.load(f)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
    img = img.resize((8, 8))  # Resize to match training data
    img_array = np.array(img).reshape(1, -1)  # Flatten the image
    prediction = model.predict(img_array)
    return {"prediction": int(prediction[0])} main.py

    
