from pydantic import BaseModel
from typing import Optional, List

class SimilarityResponse(BaseModel):
    similar_image_url: Optional[str] = None
    similarity_percent: float
    stored: bool = False

class ImageEntry(BaseModel):
    path: str
    sha256: str
    embedding: List[float]
