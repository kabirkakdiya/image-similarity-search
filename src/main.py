from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles 
from contextlib import asynccontextmanager
from ml.embedders.dino import DinoEmbedder
from database.connection import init_db
from api.routes import router
from core.config import IMAGE_STORAGE_PATH
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.embedder = DinoEmbedder()
    yield

app = FastAPI(title="Image Similarity Search", lifespan=lifespan)

os.makedirs(IMAGE_STORAGE_PATH, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGE_STORAGE_PATH), name="images")

app.include_router(router, prefix="/api/v1", tags=["v1"])