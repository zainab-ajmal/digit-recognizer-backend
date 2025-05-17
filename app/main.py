import joblib
import numpy as np
from PIL import Image

# Load model
model = joblib.load('digit_classifier.pkl')

# Example: predict from a 28x28 grayscale image
def predict_image(img_path):
    img = Image.open(img_path).convert('L').resize((28, 28))
    img_array = np.array(img).reshape(1, -1) / 255.0
    pred = model.predict(img_array)[0]
    prob = model.predict_proba(img_array)[0][pred]
    return pred, prob

digit, confidence = predict_image("path_to_image.png")
print(f"Predicted: {digit}, Confidence: {confidence:.4f}")
