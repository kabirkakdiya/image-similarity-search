from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Integer, Index
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Image(Base):    # Image model with HNSW index
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String, nullable=False)
    sha256 = Column(String, unique=True, nullable=False)
    embedding = Column(Vector(1024))

    __table_args__ = (
        Index(
            'hnsw_embedding_idx',
            'embedding',
            postgresql_using='hnsw',
            postgresql_with={
                'm': 16,           
                'ef_construction': 64 
            },
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
    )

class SimilarityResponse(BaseModel):
    duplicate: bool = False,
    matched_image: Optional[str] = None
    similarity: float
    stored: bool = False