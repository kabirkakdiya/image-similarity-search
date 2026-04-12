# Image Similarity Search

A FastAPI service that detects duplicate images using [DINOv2](https://github.com/facebookresearch/dinov2) (large) embeddings and cosine similarity, backed by PostgreSQL + pgvector.

## Stack

- **FastAPI** — API framework
- **PostgreSQL + pgvector** — vector storage and similarity search
- **DINOv2 (large)** — image embeddings
- **uv** — package manager

## Setup

```bash
# 1. Clone and install dependencies
git clone <repo-url>
cd <repo>
uv sync

# 2. Configure environment
cp .env.example .env

# 3. Create the database
psql -c "CREATE DATABASE image_duplication_detection_db;"

# 4. Run the dev server
uv run --env-file=./.env fastapi dev src/main.py
```

## API

All endpoints are under `/api/v1`.

### `GET /api/v1/`
Health check.

---

### `POST /api/v1/ingest`
Ingest a batch of images into the vector store.

Accepts a **plain text file** with one image URL per line. Images are downloaded and embedded in the background.

```
https://example.com/image1.jpg
https://example.com/image2.jpg
```

---

### `POST /api/v1/search`
Search for a duplicate of a given image.

| Field | Type | Description |
|---|---|---|
| `file` | file (optional) | Image to search |
| `image_url` | string (optional) | URL of image to search |
| `include` | boolean | If `true`, stores the image if no duplicate is found |

**Response (`SimilarityResponse`):**

```json
{
  "duplicate": false,
  "matched_image": "path/or/url",
  "similarity": 0.97,
  "stored": true
}
```