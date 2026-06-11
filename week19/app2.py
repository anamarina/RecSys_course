from __future__ import annotations

import sys
from pathlib import Path
from typing import List

try:
    from fastapi import FastAPI, Query
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FastAPI app requires 'fastapi' and 'uvicorn'. "
        "Install them with: python3 -m pip install fastapi uvicorn"
    ) from exc

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from week19.app import IMAGE_DIR, RECOMMENDER, STATIC_DIR
except ModuleNotFoundError:
    from app import IMAGE_DIR, RECOMMENDER, STATIC_DIR


class RecommendationRequest(BaseModel):
    cart_item_ids: List[str] = Field(default_factory=list)
    exclude_item_ids: List[str] = Field(default_factory=list)
    limit: int = 5


app = FastAPI(
    title="Week 19 Grocery Recommendation Demo",
    description="FastAPI version of the grocery recommendation demo with EASE.",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/bootstrap")
async def bootstrap() -> dict[str, int]:
    return {
        "page_size": 5,
        "catalog_size": len(RECOMMENDER.item_meta),
        "trained_items": len(RECOMMENDER.item_index),
    }


@app.get("/api/feed")
async def feed(
    exclude: List[str] = Query(default_factory=list),
    limit: int = 5,
) -> dict[str, list[dict]]:
    exclude_ids = {item_id for item_id in exclude if item_id in RECOMMENDER.item_meta}
    safe_limit = max(1, int(limit))
    return {"items": RECOMMENDER.feed(exclude_ids=exclude_ids, limit=safe_limit)}


@app.post("/api/recommendations")
async def recommendations(payload: RecommendationRequest) -> dict[str, list[dict]]:
    cart_ids = [item_id for item_id in payload.cart_item_ids if item_id in RECOMMENDER.item_meta]
    exclude_ids = {item_id for item_id in payload.exclude_item_ids if item_id in RECOMMENDER.item_meta}
    safe_limit = max(1, int(payload.limit))
    items = RECOMMENDER.recommend(cart_item_ids=cart_ids, exclude_ids=exclude_ids, limit=safe_limit)
    return {"items": items}


if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Running app2.py directly requires 'uvicorn'. "
            "Install it with: python3 -m pip install fastapi uvicorn"
        ) from exc

    uvicorn.run(app, host="127.0.0.1", port=8020, reload=False)
