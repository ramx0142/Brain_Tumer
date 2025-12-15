
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
@st.cache_resource
def load_model():
    with open("brain_tumor_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()



IMG_SIZE = 224

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_brain_tumor(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]

    if prediction >= 0.5:
        return "🧠 Brain Tumor Detected"
    else:
        return "✅ No Brain Tumor Detected"

# Gradio Interface
interface = gr.Interface(
    fn=predict_brain_tumor,
    inputs=gr.Image(type="pil", label="Upload Brain MRI Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Brain Tumor Detection using CNN",
    description="Upload a brain MRI image to detect the presence of a tumor."
)

interface.launch()
