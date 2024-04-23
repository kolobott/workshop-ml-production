import io
import os

from PIL import Image
from fastapi import FastAPI, UploadFile, File
from keras.src.saving.saving_api import load_model
from starlette.responses import JSONResponse

from app.utils.utils import preprocess_image

model_path = os.path.join(os.getenv('MODEL_DIR'), "model_best.h5")
app = FastAPI()
model = load_model(model_path)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Open the image, convert it to RGB and resize to expected input shape
        image = Image.open(io.BytesIO(await file.read()))
        image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(image)
        predicted_class = 'Dog' if prediction[0] > 0.5 else 'Cat'

        return JSONResponse(content={"predicted_class": predicted_class}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
