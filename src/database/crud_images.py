from typing import List, Optional, Tuple
from sqlite_vec import serialize_float32

def get_image_by_sha256(conn, sha256: str) -> Optional[str]:
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
    """, (image_id, serialize_float32(embedding)))

    conn.commit()
    return image_id

def find_most_similar(
    conn, query_embedding: List[float]
) -> Optional[Tuple[int, str, float]]:
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
    """, (serialize_float32(query_embedding), ))

    row = cur.fetchone()
    if not row:
        return None
    return row["id"], row["image_path"], row["distance"]
