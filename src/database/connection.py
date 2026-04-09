import sqlite3, sqlite_vec
from core.config import DB_PATH

def get_db_connection():
    conn = sqlite3.connect(
        DB_PATH, 
        check_same_thread=False,
        timeout=30.0
    )
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.row_factory = sqlite3.Row
    
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA mmap_size=30000000000;")
    conn.execute("PRAGMA cache_size=-2000;")
    
    return conn

def init_db():
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
    
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_images USING vec0(
            image_id INTEGER PRIMARY KEY,
            embedding float[1024]   
        )
    """)
    
    conn.commit()
    conn.close()
