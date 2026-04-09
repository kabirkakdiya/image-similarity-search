import hashlib
import io
import os
import struct
from PIL import Image

def compute_sha256(image_bytes: bytes) -> str:
    """Compute SHA-256 hash of raw image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()

def save_image_to_disk(image_bytes: bytes, sha256: str, storage_dir: str = os.getenv("IMAGE_DIR", "image_storage")) -> str:
    """Save image as JPEG using SHA-256 as filename. Returns full path."""
    os.makedirs(storage_dir, exist_ok=True)
    filepath = os.path.join(storage_dir, f"{sha256}.jpg")
    
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.save(filepath, format="JPEG", quality=95)
    return filepath

# Converts Python list to binary blob
def serialize_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)