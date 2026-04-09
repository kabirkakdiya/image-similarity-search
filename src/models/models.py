from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Integer
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String, nullable=False)
    sha256 = Column(String, unique=True, nullable=False)
    embedding = Column(Vector(1024))

class SimilarityResponse(BaseModel):
    similar_image_url: Optional[str] = None
    similarity_percent: float
    stored: bool = False

# class ImageEntry(BaseModel):
#     path: str
#     sha256: str
#     embedding: List[float] 