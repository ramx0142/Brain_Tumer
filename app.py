import streamlit as st
import pickle
import numpy as np
from PIL import Image

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered"
)

st.title("Brain Tumor Detection using CNN")
st.write("Upload a brain MRI image to detect the presence of a tumor.")

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    with open("brain_tumor_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# -------------------------------
# Image preprocessing
# -------------------------------
IMG_SIZE = 224

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # (1, 224, 224, 3)
    return image

# -------------------------------
# Prediction function
# -------------------------------
def predict_brain_tumor(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]

    if prediction >= 0.5:
        return "🧠 Brain Tumor Detected"
    else:
        return "✅ No Brain Tumor Detected"

# -------------------------------
# Streamlit UI
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            result = predict_brain_tumor(image)

        st.subheader("Prediction Result")
        if "Tumor" in result:
            st.error(result)
        else:
            st.success(result)
