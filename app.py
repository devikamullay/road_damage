import streamlit as st
from ultralytics import YOLO
from PIL import Image, ExifTags
import folium
from streamlit_folium import st_folium
import networkx as nx

# NEW: database imports
from db_models import Detection, SessionLocal, init_db

# NEW: init database + session
init_db()
db = SessionLocal()

# ----------------------------
# Helper functions for EXIF / GPS
# ----------------------------

def get_exif_data(img: Image.Image):
    """
    Extract EXIF data from a PIL image and return a dict
    containing at least the GPSInfo block (if present).
    """
    exif_data = {}
    info = img._getexif()

    if not info:
        return None

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
    Safely convert GPS (deg, min, sec) to decimal degrees.
    Handles IFDRational objects, (num, den) tuples, and weird formats.
    """
    def to_float(x):
        try:
            return float(x)
        except Exception:
            try:
                num, den = x
            except Exception:
                raise ValueError("Bad EXIF rational")
            if den == 0:
                raise ValueError("Zero denominator in EXIF")
            return num / den

    try:
        seq = list(value)
    except TypeError:
        return to_float(value)

    if len(seq) != 3:
        return to_float(value)

    d, m, s = seq
    d = to_float(d)
    m = to_float(m)
    s = to_float(s)

    return d + (m / 60.0) + (s / 3600.0)


def get_lat_lon_from_exif(exif_data):
    """
    Returns (lat, lon) in decimal degrees if available,
    otherwise (None, None). Any EXIF errors → (None, None).
    """
    if not exif_data or "GPSInfo" not in exif_data:
        return None, None

    gps_info = exif_data["GPSInfo"]

    gps_lat = gps_info.get("GPSLatitude")
    gps_lat_ref = gps_info.get("GPSLatitudeRef")
    gps_lon = gps_info.get("GPSLongitude")
    gps_lon_ref = gps_info.get("GPSLongitudeRef")

    if not gps_lat or not gps_lat_ref or not gps_lon or not gps_lon_ref:
        return None, None

    try:
        lat = _convert_to_degrees(gps_lat)
        if gps_lat_ref != "N":
            lat = -lat

        lon = _convert_to_degrees(gps_lon)
        if gps_lon_ref != "E":
            lon = -lon
    except Exception:
        return None, None

    return lat, lon


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
    "Upload a single geotagged road image (JPEG with GPS EXIF)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

st.title("Road Damage Detection and Visualisation")
st.write(
    "Upload a **geotagged** image of a road. "
    "The app will detect damage using YOLO and visualise the location on a map."
)

if uploaded_file:
    try:
        # Open image once; extract EXIF before convert/resize
        pil_img = Image.open(uploaded_file)

        # Extract GPS / EXIF
        exif_data = get_exif_data(pil_img)
        lat, lon = get_lat_lon_from_exif(exif_data)

        # Convert to RGB for YOLO and display
        image = pil_img.convert("RGB")

        # Resize large images (for display & inference)
        max_dim = 640
        if max(image.size) > max_dim:
            image.thumbnail((max_dim, max_dim))

        # st.image(
        #     image,
        #     caption=f"Uploaded: {uploaded_file.name}",
        #     use_container_width=True
        # )

        # Display coordinates or error
        if lat is None or lon is None:
            st.error("No GPS coordinates found in image metadata (EXIF). "
                     "Make sure the image is geotagged.")
        else:
            st.success(f"Image coordinates (from EXIF): {lat:.6f}, {lon:.6f}")

        # Run YOLO detection
        with st.spinner("Detecting road damage..."):
            results = model(image, conf=conf_threshold)
            detected_img_array = results[0].plot()
            detected_image = Image.fromarray(detected_img_array)


        # --- SIDE BY SIDE VIEW ---
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption=f"Original: {uploaded_file.name}", use_container_width=True)

        with col2:
            st.image(detected_image, caption="Detected Road Damage", use_container_width=True)


        

        
         # NEW: save detection info to database
        boxes = results[0].boxes
        damage_count = len(boxes)

        if damage_count > 0:
            # handle tensor/array -> float
            avg_conf = float(boxes.conf.mean().item()) if hasattr(boxes.conf.mean(), "item") else float(boxes.conf.mean())
        else:
            avg_conf = 0.0

        record = Detection(
            filename=uploaded_file.name,
            lat=lat,
            lon=lon,
            damage_count=damage_count,
            avg_conf=avg_conf,
        )

        db.add(record)
        db.commit()

        st.success(f"Saved to database: {damage_count} detections (avg conf {avg_conf:.3f})")


        #MAP BLOCK -----
          # Only proceed with map if coordinates exist
        if lat is not None and lon is not None:
            st.subheader("Map View of Detected Damage")

            # ----------------------------
            # NetworkX: build a simple road graph
            # ----------------------------
            # Here we're modelling the damaged road segment as a node in a graph.
            # You can extend this by adding more nodes/edges for a full road network.
            G = nx.Graph()
            G.add_node("damage_location", pos=(lon, lat))  # (lon, lat) for geometry use

            # You could add more nodes/edges here
            # e.g., G.add_edge("damage_location", "nearby_intersection")

            # Extract node positions for plotting on the map
            pos = nx.get_node_attributes(G, "pos")

            # ----------------------------
            # Folium map centered on damage location
            # ----------------------------
            m = folium.Map(location=[lat, lon], zoom_start=18, tiles="OpenStreetMap")

            # Add a marker for the damage location
            folium.Marker(
                location=[lat, lon],
                popup="Detected damage location",
                tooltip="Damage",
                icon=folium.Icon(icon="warning")
            ).add_to(m)

            # Draw NetworkX nodes as circle markers on map
            for node_name, (x_lon, y_lat) in pos.items():
                folium.CircleMarker(
                    location=[y_lat, x_lon],
                    radius=8,
                    popup=f"Graph node: {node_name}",
                    tooltip=node_name
                ).add_to(m)

            # Render in Streamlit
            st_folium(m, width=700, height=500)

    except Exception as e:
        st.error(f"Processing failed: {e}")


# --------------------------------
# VIEW DATABASE RECORDS (TEST)
# --------------------------------
# st.subheader("Database Records")

# try:
#     records = db.query(Detection).all()
#     if records:
#         st.write([
#             {
#                 "ID": r.id,
#                 "File": r.filename,
#                 "Lat": r.lat,
#                 "Lon": r.lon,
#                 "Damage Count": r.damage_count,
#                 "Avg Conf": r.avg_conf,
#                 "Time": r.timestamp
#             } 
#             for r in records
#         ])
#     else:
#         st.info("No records in database yet.")
# except Exception as e:
#     st.error(f"Database read error: {e}")
