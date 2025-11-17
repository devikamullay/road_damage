# db_models.py — MongoDB Atlas backend

from datetime import datetime
from pymongo import MongoClient
import os

# -------------------------------------------------------
# Use Streamlit Secrets (MONGODB_URI) instead of hardcode
# -------------------------------------------------------
MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise RuntimeError("❌ MONGODB_URI not found. Add it in Streamlit Secrets.")

# Connect to MongoDB Atlas
client = MongoClient(MONGODB_URI)

# Database + Collection
db = client["road_damage"]              # database name
detections_collection = db["detections"]  # collection name


def init_db():
    """Ping database to ensure connection works."""
    client.admin.command("ping")


def save_detection(record: dict):
    """Insert record into MongoDB."""
    detections_collection.insert_one(record)


def get_all_detections():
    """
    Return all detection records from MongoDB as a list of dicts,
    with Mongo's internal _id field removed.
    """
    return list(detections_collection.find({}, {"_id": 0}))

