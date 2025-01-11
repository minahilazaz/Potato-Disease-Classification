from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

# Initialize the FastAPI app
app = FastAPI()

# Load the TensorFlow model
MODEL_PATH = "/Users/minahil/Desktop/PotatoDisease/saved_models/13"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

MODEL = tf.keras.models.load_model(MODEL_PATH)

# Define class names
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Enable CORS if required
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins if needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    """
    Health check endpoint.
    """
    return {"message": "Server is running"}

def read_file_as_image(data: bytes) -> np.ndarray:
    """
    Converts the uploaded file's bytes to a NumPy array.
    """
    image = np.array(Image.open(BytesIO(data)).convert("RGB"))  # Ensure 3 channels (RGB)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict endpoint: Accepts an uploaded image file and returns the class and confidence.
    """
    try:
        # Read the uploaded file and convert it to an image array
        image = read_file_as_image(await file.read())
        
        # Preprocess the image (ensure size compatibility with the model)
        image = tf.image.resize(image, (224, 224))  # Adjust size as per your model's input
        image = image / 255.0  # Normalize to range [0, 1]
        
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)
        
        # Make predictions
        predictions = MODEL.predict(image_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        
        # Return response
        return {
            "class": predicted_class,
            "confidence": float(confidence),
        }
    except Exception as e:
        return {
            "error": str(e)
        }




if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
