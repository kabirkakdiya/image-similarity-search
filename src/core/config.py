import os

IMAGE_STORAGE_PATH = os.environ.get("IMAGE_DIR", "./images")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
DB_PATH = os.getenv("DATABASE_URI")
