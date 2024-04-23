import os

import PIL.Image
import gradio as gr
from keras.src.saving.saving_api import load_model
from utils.utils import preprocess_image


model_path = os.path.join(os.getenv('MODEL_DIR'), "model_best.h5")
model = load_model(model_path)


def classify_image(image: PIL.Image, threshold=0.5):
    image = preprocess_image(image)
    # Make prediction
    prediction = model.predict(image)
    pred_class = 'Cat' if prediction[0] < threshold else 'Dog'
    return pred_class


# Set up Gradio interface
interface = gr.Interface(
    fn=classify_image,
    inputs=[
        gr.Image(type='pil'),  # Adjust the shape as per your model's requirement
        gr.Slider(minimum=0, maximum=1, value=0.5, step=0.05, label="Threshold")
    ],
    outputs="text",
    title="Cat vs Dog Classifier",
    description="Upload an image to classify it as a cat or a dog, with a customizable threshold."
)

# Define credentials (Username and Password)
auth = ('username', 'password')  # Replace with your actual username and password

# Launch the app with authentication
interface.launch(auth=auth, server_name="0.0.0.0", server_port=8000, share=False)
