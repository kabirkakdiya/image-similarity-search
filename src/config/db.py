import sqlite3
import sqlite_vec
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple
from utils.image_utils import serialize_embedding

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="DINOv2 Image Search API", lifespan=lifespan)

# --- Database Configuration ---
DB_PATH = os.getenv("DATABASE_URI")

def get_db_connection():
    """Connects to SQLite and loads the sqlite-vec extension."""
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes tables for metadata and vector storage."""
    conn = get_db_connection()
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            sha256 TEXT UNIQUE NOT NULL, -- For exact match/deduplication
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # For vectors(embeddings)
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_images USING vec0(
            image_id INTEGER PRIMARY KEY,
            embedding float[1024]   
        )
    """)
    
    conn.commit()
    conn.close()

# --- Schemas ---
class ImageEntry(BaseModel) :
    path: str
    sha256: str
    embedding: List[float] # DINOv2 output (1024 dims)

def get_image_by_sha256(conn, sha256: str) -> Optional[str]:
    """Return image_path if SHA-256 already exists."""
    cur = conn.cursor()
    cur.execute("SELECT image_path FROM images WHERE sha256 = ?", (sha256,))
    row = cur.fetchone()
    return row["image_path"] if row else None

def insert_image_metadata_and_vector(
    conn,
    image_path: str,
    sha256: str,
    embedding: List[float]
) -> Optional[int]:
    """Insert metadata + vector. Returns image_id or None if duplicate."""
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO images (image_path, sha256)
        VALUES (?, ?)
        ON CONFLICT(sha256) DO NOTHING
        RETURNING id
    """, (image_path, sha256))

    row = cur.fetchone()
    if not row:
        return None

    image_id = row["id"]

    cur.execute("""
        INSERT INTO vec_images (image_id, embedding)
        VALUES (?, ?)
    """, (image_id, serialize_embedding(embedding)))

    conn.commit()
    return image_id

def find_most_similar(
    conn, query_embedding: List[float]
) -> Optional[Tuple[int, str, float]]:
    """Return (id, image_path, cosine_distance) of the most similar image."""
    cur = conn.cursor()
    cur.execute("""
        SELECT
            i.id,
            i.image_path,
            v.distance
        FROM vec_images v
        JOIN images i ON v.image_id = i.id
        WHERE v.embedding MATCH ?
        AND k = 1
    """, (serialize_embedding(query_embedding), ))  # used trailing comma to convert it to tuple

    row = cur.fetchone()
    if not row:
        return None
    return row["id"], row["image_path"], row["distance"]