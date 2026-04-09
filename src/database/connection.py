from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.config import DB_PATH
from models.models import Base

engine = create_engine(DB_PATH)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db_connection():
    return SessionLocal()

def init_db():
    Base.metadata.create_all(bind=engine)
