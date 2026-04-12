from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc
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
    similarity_expr = 1 - Image.embedding.cosine_distance(query_embedding)
    
    similarity_percentage_expr = (1 + similarity_expr)/2 * 100 # to inflate the resulting percentage, 0.0 -> 50% and 1.0 -> 100%


# SELECT image_id, (1 - (embedding1 <=> embedding2)) * 100 AS similarity_percentage FROM image_tbl WHERE (1- (embedding1 <=> embedding2)) > 0.85 ORDER BY similarity_percentage DESC LIMIT 1;
    result = session.query(
        Image.id,
        Image.image_path,
        similarity_percentage_expr.label('similarity_percentage')
    ).filter(
        similarity_expr > threshold  # WHERE (1 - distance) > 0.85
    ).order_by(
        desc('similarity_percentage')
    ).first()

    if not result:
        return None

    return result.id, result.image_path, round(float(result.similarity_percentage), 2)