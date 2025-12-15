import streamlit as st
import pickle
import numpy as np
import cv2
from PIL import Image

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection using CNN")
st.write("Upload an MRI image to predict whether a brain tumor is present.")

# ------------------------------
# Load the trained model
# ------------------------------
@st.cache_resource
def load_model():
    with open("brain_tumor_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ------------------------------
# Image preprocessing function
# ------------------------------
def preprocess_image(image):
    """
    Preprocess the uploaded MRI image
    """
    img = np.array(image)

    # Convert to grayscale if RGB
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize (change to your model input size)
    img = cv2.resize(img, (128, 128))

    # Normalize
    img = img / 255.0

    # Reshape for CNN: (1, height, width, channels)
    img = img.reshape(1, 128, 128, 1)

    return img

# ------------------------------
# File uploader
# ------------------------------
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)

            # Binary classification threshold
            result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"

        st.subheader("Prediction Result")
        if result == "Tumor Detected":
            st.error(result)
        else:
            st.success(result)
