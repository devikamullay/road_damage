import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ExifTags
import folium
from streamlit_folium import st_folium
import networkx as nx
from datetime import datetime

# MongoDB functions
from db_models import init_db, save_detection

# Test MongoDB connectivity
init_db()

st.set_page_config(page_title="Road Damage Detection", layout="wide")


# ----------------------------
# Helper functions for EXIF / GPS
# ----------------------------


def get_exif_data(img: Image.Image):
    """
    Extract EXIF data from a PIL image and return a dict
    containing at least the GPSInfo block (if present).
    If the image has no EXIF or _getexif, return None.
    """
    # Some formats (e.g. PNG) don't have _getexif at all
    if not hasattr(img, "_getexif"):
        return None

    try:
        info = img._getexif()
    except Exception:
        return None

    if not info:
        return None

    exif_data = {}
    for tag, value in info.items():
        decoded = ExifTags.TAGS.get(tag, tag)
        if decoded == "GPSInfo":
            gps_data = {}
            for t in value:
                sub_decoded = ExifTags.GPSTAGS.get(t, t)
                gps_data[sub_decoded] = value[t]
            exif_data["GPSInfo"] = gps_data

    return exif_data if exif_data else None


def _convert_to_degrees(value):
    def to_float(x):
        try:
            return float(x)
        except Exception:
            num, den = x
            return num / den
    seq = list(value)
    d, m, s = seq
    return to_float(d) + (to_float(m) / 60.0) + (to_float(s) / 3600.0)


def get_lat_lon_from_exif(exif_data):
    if not exif_data or "GPSInfo" not in exif_data:
        return None, None
    gps_info = exif_data["GPSInfo"]
    gps_lat = gps_info.get("GPSLatitude")
    gps_lat_ref = gps_info.get("GPSLatitudeRef")
    gps_lon = gps_info.get("GPSLongitude")
    gps_lon_ref = gps_info.get("GPSLongitudeRef")

    if not gps_lat or not gps_lat_ref or not gps_lon or not gps_lon_ref:
        return None, None

    lat = _convert_to_degrees(gps_lat)
    if gps_lat_ref != "N":
        lat = -lat

    lon = _convert_to_degrees(gps_lon)
    if gps_lon_ref != "E":
        lon = -lon
    return lat, lon


# ----------------------------
# Load YOLO model
# ----------------------------
model = YOLO("yolo11s_trained.pt")


# ----------------------------
# Streamlit UI
# ----------------------------
st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload a single geotagged road image (JPEG with GPS EXIF)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

st.title("Road Damage Detection & Visualisation")

if uploaded_file:
    try:
        # Always open + convert strictly to RGB
        pil_img = Image.open(uploaded_file).convert("RGB")

        # Extract EXIF BEFORE thumbnail
        exif_data = get_exif_data(pil_img)
        lat, lon = get_lat_lon_from_exif(exif_data)

        # Work on a copy for display / YOLO
        image = pil_img.copy()

        # Resize large images
        max_dim = 640
        if max(image.size) > max_dim:
            image.thumbnail((max_dim, max_dim))

        # Coordinates info
        if lat is None or lon is None:
            st.error("No GPS coordinates found. Use a geotagged image.")
        else:
            st.success(f"Image coordinates: {lat:.6f}, {lon:.6f}")

        # Convert to NumPy array for YOLO (avoids mode issues)
        img_array = np.array(image)

        with st.spinner("Detecting road damage..."):
            results = model(img_array, conf=conf_threshold)
            detected_img_array = results[0].plot()
            detected_image = Image.fromarray(detected_img_array)

        # --- SIDE BY SIDE VIEW ---
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(detected_image, caption="Detected Road Damage", use_container_width=True)

        # Save to MongoDB
        boxes = results[0].boxes
        damage_count = len(boxes)
        if damage_count > 0:
            mean_conf = boxes.conf.mean()
            avg_conf = float(mean_conf.item()) if hasattr(mean_conf, "item") else float(mean_conf)
        else:
            avg_conf = 0.0

        record = {
            "filename": uploaded_file.name,
            "lat": lat,
            "lon": lon,
            "damage_count": damage_count,
            "avg_conf": avg_conf,
            "timestamp": datetime.utcnow(),
        }

        save_detection(record)
        st.success(f"Saved to MongoDB (Damage: {damage_count}, Avg Conf: {avg_conf:.3f})")

        # Map
        if lat is not None and lon is not None:
            st.subheader("Map View of Detected Damage")
            G = nx.Graph()
            G.add_node("damage_location", pos=(lon, lat))
            m = folium.Map(location=[lat, lon], zoom_start=18, tiles="OpenStreetMap")
            folium.Marker(
                location=[lat, lon],
                popup="Detected Road Damage",
                tooltip="Damage",
                icon=folium.Icon(icon="warning")
            ).add_to(m)
            st_folium(m, width=700, height=500)

    except Exception as e:
        st.error(f"Processing failed: {e}")
