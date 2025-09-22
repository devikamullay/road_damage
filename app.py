import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import requests

# ----------------------------
# Model setup
# ----------------------------

# Change this to your hosted model URL if file >1 GB
MODEL_URL = "https://huggingface.co/devikaaaa/yolo11s/resolve/main/yolo11s_trained.pt"
MODEL_PATH = "yolo11s_trained.pt"

# Download once if not available locally
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

# Load YOLO model
model = YOLO(MODEL_PATH)

# ----------------------------
# Streamlit UI
# ----------------------------

st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload a single road image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

st.title("Road Damage Detection")
st.write("Upload an image of a road, to detect damage.")

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        # Resize large images
        max_dim = 640
        if max(image.size) > max_dim:
            image.thumbnail((max_dim, max_dim))

        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        with st.spinner("Detecting road damage..."):
            results = model(image, conf=conf_threshold)

            # Plot detections
            detected_img_array = results[0].plot()
            detected_image = Image.fromarray(detected_img_array)

        st.image(detected_image, caption="Detected Damage")

    except Exception as e:
        st.error(f"Model failed: {e}")
