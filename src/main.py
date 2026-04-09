import httpx, os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles 
from fastapi import BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pipeline import DinoEmbedder
from config.db import init_db, get_db_connection, get_image_by_sha256, insert_image_metadata_and_vector, find_most_similar
from utils.image_utils import compute_sha256, save_image_to_disk

IMAGE_STORAGE_PATH = os.environ.get("IMAGE_DIR", "./images")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder
    init_db()
    embedder = DinoEmbedder()
    yield

app = FastAPI(title="DINOv2 Image Similarity Search API", lifespan=lifespan)
app.mount("/images", StaticFiles(directory=IMAGE_STORAGE_PATH), name="images")

@app.get("/")
def health_check():
    return {"Hello": "from FastAPI with uv"}


class SimilarityResponse(BaseModel):
    similar_image_url: Optional[str] = None
    similarity_percent: float
    stored: bool = False

async def download_image(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"URL did not return an image (got content-type: {content_type})"
            )
        return resp.content

def path_to_url(file_path: Optional[str]) -> Optional[str]:
    """Convert an absolute/relative disk path to a publicly accessible URL."""
    if file_path is None:
        return None
    rel = os.path.relpath(file_path, IMAGE_STORAGE_PATH)
    return f"{BASE_URL}/images/{rel}"

@app.post("/search", response_model=SimilarityResponse)
async def search_image(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    future_use: bool = Form(False),
):
    if not file and not image_url:
        raise HTTPException(status_code=400, detail="Provide either 'file' or 'image_url'")
    if file and image_url:
        raise HTTPException(status_code=400, detail="Provide only one of 'file' or 'image_url'")

    image_bytes = await file.read() if file else await download_image(image_url)

    sha256 = compute_sha256(image_bytes)
    conn = get_db_connection()

    try:
        existing_path = get_image_by_sha256(conn, sha256)
        if existing_path:
            return SimilarityResponse(
                similar_image_url=path_to_url(existing_path),
                similarity_percent=100.0
            )

        embedding = embedder.embed(image_bytes)
        result = find_most_similar(conn, embedding)

        if result is None:
            similar_path = None
            similarity_percent = 0.0
        else:
            _, similar_path, distance = result
            similarity_percent = (1 - distance) * 100

        stored = False
        if future_use:
            saved_path = save_image_to_disk(image_bytes, sha256)
            insert_image_metadata_and_vector(conn, saved_path, sha256, embedding)
            stored = True

        return SimilarityResponse(
            similar_image_url=path_to_url(similar_path),
            similarity_percent=round(similarity_percent, 2),
            stored=stored
        )

    finally:
        conn.close()


# API to fill the db
@app.post("/ingest")
async def ingest_images(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    content = await file.read()
    urls = [line.strip() for line in content.decode().splitlines() if line.strip()]

    if not urls:
        raise HTTPException(status_code=400, detail="No URLs found in file")

    background_tasks.add_task(process_urls, urls)
    return {"message": f"Ingestion started for {len(urls)} URLs"}

async def process_urls(urls: list[str]):
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
    print(f"Ingestion complete: {results}")  # visible in server logs