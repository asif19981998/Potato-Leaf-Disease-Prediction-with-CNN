from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
app = FastAPI()

MODEL = tf.keras.models.load_model("../models/1.keras")
Class_Names = ["Early Bright", "Late Bright", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:

    try:
        image = Image.open(BytesIO(data))
        resized_image = image.resize((256, 256))
        resized_image_array = np.array(resized_image)
        return resized_image_array
    except Exception as e:
        return f"An error occurred: {e}"

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    images = read_file_as_image(await file.read())
    try:

        image_batch = np.expand_dims(images, 0)
        prediction = MODEL.predict(image_batch)
        predicted_class = Class_Names[np.argmax(prediction)]
        confidence = np.max(prediction[0])
    except Exception as e:
        return f"An error occurred: {e}"

    return {
        'prediction': predicted_class,
        'confidence': float(confidence)
    }




if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
