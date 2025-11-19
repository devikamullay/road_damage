import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ExifTags
import folium
from streamlit_folium import st_folium
from datetime import datetime
import pandas as pd

# MongoDB functions
from db_models import init_db, save_detection, get_all_detections

# Test MongoDB connectivity
init_db()

st.set_page_config(page_title="Road Damage Detection", layout="wide")

# ----------------------------
# Session state init
# ----------------------------
if "current_detection" not in st.session_state:
    # will hold last detection results so we don't recompute on every rerun
    st.session_state["current_detection"] = None

if "saved_keys" not in st.session_state:
    # track which files (with given threshold) we've already saved to DB this session
    st.session_state["saved_keys"] = set()


# ----------------------------
# Helper functions for EXIF / GPS
# ----------------------------

def get_exif_data(img: Image.Image):
    """
    Extract EXIF data from a PIL image and return a dict
    containing at least the GPSInfo block (if present).
    If the image has no EXIF or _getexif, return None.
    """
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
    """
    Convert EXIF GPS coordinates to decimal degrees.
    Handles:
    - (num, den) tuples
    - IFDRational objects
    - plain numbers
    - and avoids division-by-zero.
    """

    def to_float(x):
        # Already a simple number
        if isinstance(x, (int, float)):
            return float(x)

        # IFDRational-like object
        if hasattr(x, "numerator") and hasattr(x, "denominator"):
            num = float(x.numerator)
            den = float(x.denominator) if x.denominator not in (0, None) else 1.0
            return num / den

        # (num, den) tuple/list
        if isinstance(x, (tuple, list)) and len(x) == 2:
            num, den = x
            num = float(num)
            den = float(den) if den not in (0, None) else 1.0
            return num / den

        # Fallback
        try:
            return float(x)
        except Exception:
            return 0.0  # last-resort fallback

    # Some formats give a 3-element sequence (d, m, s), some are weird
    try:
        seq = list(value)
    except TypeError:
        # Not iterable -> treat as a single degree value
        return to_float(value)

    if len(seq) == 3:
        d, m, s = seq
        return to_float(d)

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

# ----------------------------
# Per-image detection + single-point map
# ----------------------------
if uploaded_file:
    try:
        # Unique key for this file + threshold in this session
        file_key = f"{uploaded_file.name}_{uploaded_file.size}_{conf_threshold}"

        # Decide whether we need to recompute detection
        need_recompute = (
            st.session_state["current_detection"] is None
            or st.session_state["current_detection"]["file_key"] != file_key
        )

        if need_recompute:
            # 1) Open original image (do NOT convert yet)
            original_img = Image.open(uploaded_file)

            # 2) Extract EXIF / GPS from the *original* image
            exif_data = get_exif_data(original_img)
            lat, lon = get_lat_lon_from_exif(exif_data)

            # 3) Now make an RGB copy for YOLO + display
            pil_img = original_img.convert("RGB")
            image = pil_img.copy()

            # 4) Resize large images for display
            max_dim = 640
            if max(image.size) > max_dim:
                image.thumbnail((max_dim, max_dim))

            # 5) Detection using YOLO (only once per file+threshold)
            img_array = np.array(image)

            with st.spinner("Detecting road damage..."):
                results = model(img_array, conf=conf_threshold)
                detected_img_array = results[0].plot()  # numpy array

            boxes = results[0].boxes
            damage_count = len(boxes)
            if damage_count > 0:
                mean_conf = boxes.conf.mean()
                avg_conf = float(mean_conf.item()) if hasattr(mean_conf, "item") else float(mean_conf)
            else:
                avg_conf = 0.0

            # Store everything in session_state so reruns reuse it
            st.session_state["current_detection"] = {
                "file_key": file_key,
                "image": image,
                "detected_image": detected_img_array,
                "lat": lat,
                "lon": lon,
                "damage_count": damage_count,
                "avg_conf": avg_conf,
                "filename": uploaded_file.name,
            }

            # Save to MongoDB only once per file_key in this session
            if file_key not in st.session_state["saved_keys"]:
                record = {
                    "filename": uploaded_file.name,
                    "lat": lat,
                    "lon": lon,
                    "damage_count": damage_count,
                    "avg_conf": avg_conf,
                    "timestamp": datetime.utcnow(),
                }
                save_detection(record)
                st.session_state["saved_keys"].add(file_key)
                st.success(f"Saved to MongoDB (Damage: {damage_count}, Avg Conf: {avg_conf:.3f})")
            else:
                st.info("This detection was already saved in this session.")
        # ---- Use cached detection on rerun ----
        det = st.session_state["current_detection"]
        image = det["image"]
        detected_image = det["detected_image"]
        lat = det["lat"]
        lon = det["lon"]
        damage_count = det["damage_count"]
        avg_conf = det["avg_conf"]

        # Show coord info
        if lat is None or lon is None:
            st.error("No GPS coordinates found. Use a geotagged image.")
        else:
            st.success(f"Image coordinates: {lat:.6f}, {lon:.6f}")

        # --- SIDE BY SIDE VIEW ---
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(detected_image, caption="Detected Road Damage", use_container_width=True)

        # Map for THIS detection
        if lat is not None and lon is not None:
            st.subheader("Map View of This Detection")
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

# ----------------------------
# Accumulated map of ALL detections (outside if uploaded_file)
# ----------------------------
st.subheader("Accumulated Map of All Detected Damage")

try:
    all_records = get_all_detections()
except Exception as e:
    st.error(f"Failed to load past detections: {e}")
    all_records = []

if all_records:
    df = pd.DataFrame(all_records)

    # keep only rows with valid coordinates
    if {"lat", "lon"}.issubset(df.columns):
        df = df.dropna(subset=["lat", "lon"])

        if not df.empty:
            center_lat = df["lat"].mean()
            center_lon = df["lon"].mean()

            m_all = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles="OpenStreetMap"
            )

            for _, row in df.iterrows():
                popup_text = (
                    f"File: {row.get('filename', 'N/A')}<br>"
                    f"Damage count: {row.get('damage_count', 'N/A')}<br>"
                    f"Avg conf: {row.get('avg_conf', 'N/A')}"
                )

                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=8,
                    popup=popup_text,
                    tooltip="Damage",
                    color="red",            # outline
                    fill=True,
                    fill_color="red",       # fill
                    fill_opacity=0.8
                ).add_to(m_all)

            st_folium(m_all, width=700, height=500)
        else:
            st.info("No detections with valid coordinates yet.")
    else:
        st.info("No latitude/longitude fields found in database.")
else:
    st.info("No past detections in database yet.")
