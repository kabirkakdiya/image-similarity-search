from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc, text
from sqlalchemy.exc import IntegrityError
from models.models import Image
import os

def get_image_by_sha256(session: Session, sha256: str) -> Optional[str]:
    image = session.query(Image).filter(Image.sha256 == sha256).first()
    return image.image_path if image else None

def insert_image_metadata_and_vector(
    session: Session,
    image_path: str,
    sha256: str,
    embedding: List[float]
) -> Optional[int]:
    new_image = Image(
        image_path=image_path,
        sha256=sha256,
        embedding=embedding
    )
    session.add(new_image)
    try:
        session.commit()
        session.refresh(new_image)
        return new_image.id
    except IntegrityError:
        session.rollback()
        return None

def find_most_similar(
    session: Session, 
    query_embedding: List[float], 
    threshold: float = float(os.getenv("THRESHOLD", 0.80))
) -> Optional[Tuple[int, str, float]]:
    
    sql = text("""
        SELECT 
            id, 
            image_path, 
            ((1 + (1 - (embedding <=> :embedding))) / 2) AS similarity_percentage
        FROM images
        WHERE (1 - (embedding <=> :embedding)) > :threshold
        ORDER BY similarity_percentage DESC
        LIMIT 1
    """)
    result = session.execute(
        sql, 
        {"embedding": str(query_embedding), "threshold": threshold}
    ).fetchone()
    
    if not result:
        return None

    return result[0], result[1], round(float(result[2]), 2)