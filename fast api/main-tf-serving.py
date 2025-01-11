from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests

app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust based on your environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TensorFlow Serving endpoint
ENDPOINT = "http://localhost:8501/v1/models/potatoes_model:predict"

# Class names for predictions
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    """
    Health check endpoint to verify the server is running.
    """
    return {"message": "Hello, the server is running!"}

def read_file_as_image(data: bytes) -> np.ndarray:
    """
    Convert the uploaded file into a preprocessed NumPy array.
    """
    try:
        image = Image.open(BytesIO(data)).convert("RGB")  # Ensure RGB format
        image = image.resize((256, 256))  # Resize to model's expected input size
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of the uploaded image using TensorFlow Serving.
    """
    try:
        # Step 1: Preprocess the uploaded file
        image_data = await file.read()
        image = read_file_as_image(image_data)
        image_batch = np.expand_dims(image, axis=0)  # Add batch dimension

        # Step 2: Prepare the payload
        json_data = {"instances": image_batch.tolist()}

        # Step 3: Send request to TensorFlow Serving
        response = requests.post(ENDPOINT, json=json_data)
        response.raise_for_status()  # Raise error for HTTP issues

        # Step 4: Parse the response
        predictions = response.json().get("predictions", [])
        if not predictions:
            raise HTTPException(status_code=500, detail="No predictions returned by the model.")

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = predictions[0][np.argmax(predictions[0])]

        # Step 5: Return the result
        return {"class": predicted_class, "confidence": confidence}

    except HTTPException as http_ex:
        # Reraise known HTTP exceptions
        raise http_ex
    except Exception as e:
        # Catch-all for unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
