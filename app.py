import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import requests

# ----------------------------
# Model setup
# ----------------------------

# Load YOLO model
model = YOLO("yolo11s_trained.pt")

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
