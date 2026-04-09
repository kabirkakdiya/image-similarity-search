from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from typing import Optional
from models.models import SimilarityResponse
from utils.image_utils import compute_sha256, save_image_to_disk
from database.connection import get_db_connection
from database.crud_images import get_image_by_sha256, insert_image_metadata_and_vector, find_most_similar
from services.image_service import download_image, path_to_url, process_urls

router = APIRouter()

@router.get("/")
def health_check():
    return {"Hello": "from FastAPI with uv"}

@router.post("/search", response_model=SimilarityResponse)
async def search_image(
    request: Request,
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

    embedder = request.app.state.embedder

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
            similarity_percent = (1 - (distance / 2)) * 100  # distance is 0-2

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

@router.post("/ingest")
async def ingest_images(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    content = await file.read()
    urls = [line.strip() for line in content.decode().splitlines() if line.strip()]

    if not urls:
        raise HTTPException(status_code=400, detail="No URLs found in file")

    embedder = request.app.state.embedder
    background_tasks.add_task(process_urls, urls, embedder)
    return {"message": f"Ingestion started for {len(urls)} URLs"}
