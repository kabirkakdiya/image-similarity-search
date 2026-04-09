from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from models.models import Image

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
    session: Session, query_embedding: List[float]
) -> Optional[Tuple[int, str, float]]:
    result = session.query(
        Image.id,
        Image.image_path,
        Image.embedding.cosine_distance(query_embedding).label('distance')
    ).order_by('distance').first()

    if not result:
        return None

    distance = float(result.distance) if result.distance is not None else 1.0
    cosine_similarity = 1.0 - distance

    # Always check that cosine_similarity is between -1 and 1
    if cosine_similarity < -1.0:
        cosine_similarity = -1.0
    elif cosine_similarity > 1.0:
        cosine_similarity = 1.0
        
    similarity_percent = round(((cosine_similarity + 1) / 2) * 100, 2)

    return result.id, result.image_path, similarity_percent
