import sqlite3
import sqlite_vec
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI(title="DINOv2 Image Search API")

# --- Database Configuration ---
DB_PATH = os.getenv("DATABASE_URI")

def get_db_connection():
    """Connects to SQLite and loads the sqlite-vec extension."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes tables for metadata and vector storage."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Standard table for metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            sha256 TEXT UNIQUE NOT NULL, -- For exact match/deduplication
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        )
    """)
    
    # 2. Virtual table for vector search (sqlite-vec)
    # We use float[1024] for DINOv2-Large embeddings
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_images USING vec0(
            image_id INTEGER PRIMARY KEY,
            embedding float[1024]
        )
    """)
    
    conn.commit()
    conn.close()

# Initialize DB on startup
@app.on_event("startup")
async def startup_event():
    init_db()

# --- Schemas ---
class ImageEntry(BaseModel):
    path: str
    sha256: str
    embedding: List[float] # DINOv2 output (1024 dims)
    category: Optional[str] = "general"
    extra_metadata: Optional[str] = "{}"