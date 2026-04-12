import httpx, os
from fastapi import HTTPException
from core.config import IMAGE_STORAGE_PATH, BASE_URL
from utils.image_utils import compute_sha256, save_image_to_disk
from database.connection import get_db_connection
from database.crud_images import get_image_by_sha256, insert_image_metadata_and_vector

async def download_image(url: str) -> bytes:
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()  # catches 4xx/5xx HTTP errors
            content_type = resp.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                raise ValueError(f"URL did not return an image (got content-type: {content_type})")
            return resp.content

    except httpx.TimeoutException:
        raise ValueError(f"Request timed out while fetching: {url}")
    except httpx.ConnectError:
        raise ValueError(f"Could not connect to URL: {url}")
    except httpx.HTTPStatusError as e:
        raise ValueError(f"URL returned HTTP {e.response.status_code}: {url}")
    except ValueError:
        raise  # re-raise content-type error from above
    except Exception as e:
        raise ValueError(f"Failed to fetch image: {str(e)}")

def path_to_url(file_path: str | None) -> str | None:
    """Convert an absolute/relative disk path to a publicly accessible URL."""
    if file_path is None:
        return None
    rel = os.path.relpath(file_path, IMAGE_STORAGE_PATH)
    return f"{BASE_URL}/images/{rel}"

async def process_urls(urls: list[str], embedder):
    results = {"stored": 0, "skipped": 0, "failed": []}
    conn = get_db_connection()
    try:
        for url in urls:
            try:
                image_bytes = await download_image(url)
                sha256 = compute_sha256(image_bytes)
                if get_image_by_sha256(conn, sha256):
                    results["skipped"] += 1
                    continue
                embedding = embedder.embed(image_bytes)
                saved_path = save_image_to_disk(image_bytes, sha256)
                insert_image_metadata_and_vector(conn, saved_path, sha256, embedding)
                results["stored"] += 1
            except Exception as e:
                results["failed"].append({"url": url, "error": str(e)})
    finally:
        conn.close()
    print(f"Ingestion complete: {results}")
