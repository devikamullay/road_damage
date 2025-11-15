from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# ----------------------------
# MySQL connection via XAMPP
# ----------------------------
# Adjust user/password/host/dbname to match your setup.
# Default XAMPP setup (user: root, no password, db: road_damage):
DATABASE_URL = "mysql+pymysql://root:@localhost:3306/road_damage"
# If you have a password, e.g. mypass:
# DATABASE_URL = "mysql+pymysql://root:mypass@localhost:3306/road_damage"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # helps avoid stale connections
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    damage_count = Column(Integer, nullable=False)
    avg_conf = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)

