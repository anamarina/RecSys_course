from __future__ import annotations

import argparse
import json
import mimetypes
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "grocery_synthetic_dataset"
STATIC_DIR = BASE_DIR / "static"
CATALOG_PATH = DATA_DIR / "item_catalog.csv"
LINE_ITEMS_PATH = DATA_DIR / "grocery_baskets_line_items.csv"
IMAGE_DIR = DATA_DIR / "item_images"


@dataclass(frozen=True)
class CatalogItem:
    item_id: str
    item_name: str
    category: str
    image_url: str
    popularity: int


class GroceryRecommender:
    def __init__(self, catalog_path: Path, line_items_path: Path, l2: float = 80.0):
        self.catalog_path = catalog_path
        self.line_items_path = line_items_path
        self.l2 = l2

        self.catalog_df: Optional[pd.DataFrame] = None
        self.item_meta: dict[str, CatalogItem] = {}
        self.item_index: dict[str, int] = {}
        self.index_item: list[str] = []
        self.B: Optional[np.ndarray] = None
        self.popularity_order: list[str] = []
        self.feed_order: list[str] = []

        self._fit()

    def _fit(self) -> None:
        catalog = pd.read_csv(self.catalog_path, encoding="utf-8-sig")
        baskets = pd.read_csv(self.line_items_path, encoding="utf-8-sig")

        basket_items = baskets[["basket_id", "item_id"]].drop_duplicates()
        popularity = basket_items["item_id"].value_counts()

        catalog = catalog.copy()
        catalog["popularity"] = catalog["item_id"].map(popularity).fillna(0).astype(int)
        catalog["image_url"] = catalog["item_id"].map(lambda item_id: f"/images/{item_id}.png")
        catalog = catalog.sort_values(["popularity", "item_name"], ascending=[False, True]).reset_index(drop=True)

        trained_items = sorted(basket_items["item_id"].unique().tolist())
        self.item_index = {item_id: idx for idx, item_id in enumerate(trained_items)}
        self.index_item = trained_items

        matrix = np.zeros((basket_items["basket_id"].nunique(), len(trained_items)), dtype=np.float64)
        basket_index = {basket_id: idx for idx, basket_id in enumerate(sorted(basket_items["basket_id"].unique()))}
        for row in basket_items.itertuples(index=False):
            matrix[basket_index[row.basket_id], self.item_index[row.item_id]] = 1.0

        gram = matrix.T @ matrix
        diag = np.arange(gram.shape[0])
        gram[diag, diag] += self.l2
        precision = np.linalg.inv(gram)
        B = precision / (-np.diag(precision))
        B[diag, diag] = 0.0

        self.catalog_df = catalog
        self.B = B
        self.popularity_order = catalog.sort_values(["popularity", "item_name"], ascending=[False, True])["item_id"].tolist()
        self.feed_order = self._build_diverse_feed_order(catalog)
        self.item_meta = {
            row.item_id: CatalogItem(
                item_id=row.item_id,
                item_name=row.item_name,
                category=row.category,
                image_url=row.image_url,
                popularity=int(row.popularity),
            )
            for row in catalog.itertuples(index=False)
        }

    def _build_diverse_feed_order(self, catalog: pd.DataFrame) -> list[str]:
        per_category: dict[str, list[str]] = {}
        for category, group in catalog.groupby("category", sort=False):
            ordered = (
                group.sort_values(["popularity", "item_name"], ascending=[False, True])["item_id"].tolist()
            )
            per_category[category] = ordered

        category_order = (
            catalog.groupby("category")["popularity"]
            .sum()
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        feed_order: list[str] = []
        category_offsets = {category: 0 for category in per_category}
        while True:
            appended = False
            for category in category_order:
                offset = category_offsets[category]
                items = per_category[category]
                if offset < len(items):
                    feed_order.append(items[offset])
                    category_offsets[category] += 1
                    appended = True
            if not appended:
                break
        return feed_order

    def serialize_item(self, item_id: str) -> dict[str, Any]:
        item = self.item_meta[item_id]
        return {
            "item_id": item.item_id,
            "item_name": item.item_name,
            "category": item.category,
            "image_url": item.image_url,
            "popularity": item.popularity,
        }

    def feed(self, exclude_ids: set[str], limit: int = 5) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for item_id in self.feed_order:
            if item_id in exclude_ids:
                continue
            items.append(self.serialize_item(item_id))
            if len(items) >= limit:
                break
        return items

    def recommend(self, cart_item_ids: list[str], exclude_ids: set[str], limit: int = 5) -> list[dict[str, Any]]:
        cart_unique = [item_id for item_id in dict.fromkeys(cart_item_ids) if item_id in self.item_index]
        if not cart_unique:
            return self.feed(exclude_ids=exclude_ids, limit=limit)

        assert self.B is not None
        cart_idx = [self.item_index[item_id] for item_id in cart_unique]
        user_vector = np.zeros(len(self.index_item), dtype=np.float64)
        user_vector[cart_idx] = 1.0
        scores = user_vector @ self.B
        scores[cart_idx] = -np.inf

        ranked_idx = np.argsort(scores)[::-1]
        recommendations: list[dict[str, Any]] = []
        used = set(cart_item_ids)

        for idx in ranked_idx:
            item_id = self.index_item[int(idx)]
            if item_id in exclude_ids or item_id in used:
                continue
            if not np.isfinite(scores[idx]):
                continue

            reasons = self._top_reasons(cart_unique, int(idx))
            payload = self.serialize_item(item_id)
            payload["score"] = round(float(scores[idx]), 4)
            payload["reasons"] = reasons
            recommendations.append(payload)
            used.add(item_id)
            if len(recommendations) >= limit:
                return recommendations

        for item_id in self.popularity_order:
            if item_id in exclude_ids or item_id in used:
                continue
            payload = self.serialize_item(item_id)
            payload["score"] = 0.0
            payload["reasons"] = []
            recommendations.append(payload)
            used.add(item_id)
            if len(recommendations) >= limit:
                break
        return recommendations

    def _top_reasons(self, cart_item_ids: list[str], candidate_idx: int, top_n: int = 2) -> list[str]:
        assert self.B is not None
        contributions: list[tuple[float, str]] = []
        for item_id in cart_item_ids:
            src_idx = self.item_index[item_id]
            contribution = float(self.B[src_idx, candidate_idx])
            contributions.append((contribution, item_id))
        contributions.sort(reverse=True)
        reasons: list[str] = []
        for _, item_id in contributions[:top_n]:
            reasons.append(self.item_meta[item_id].item_name)
        return reasons


RECOMMENDER = GroceryRecommender(CATALOG_PATH, LINE_ITEMS_PATH)


class GroceryDemoHandler(BaseHTTPRequestHandler):
    server_version = "GroceryRecDemo/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_file(STATIC_DIR / "index.html")
            return
        if parsed.path.startswith("/static/"):
            rel_path = parsed.path.removeprefix("/static/")
            self._serve_file(STATIC_DIR / rel_path)
            return
        if parsed.path.startswith("/images/"):
            rel_path = parsed.path.removeprefix("/images/")
            self._serve_file(IMAGE_DIR / rel_path)
            return
        if parsed.path == "/api/bootstrap":
            self._json_response(
                {
                    "page_size": 5,
                    "catalog_size": len(RECOMMENDER.item_meta),
                    "trained_items": len(RECOMMENDER.item_index),
                }
            )
            return
        if parsed.path == "/api/feed":
            params = parse_qs(parsed.query)
            exclude_ids = self._extract_query_list(params, "exclude")
            limit = self._extract_query_int(params, "limit", default=5)
            self._json_response({"items": RECOMMENDER.feed(exclude_ids=exclude_ids, limit=limit)})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/recommendations":
            payload = self._read_json_body()
            cart_ids = self._normalize_item_ids(payload.get("cart_item_ids", []))
            exclude_ids = set(self._normalize_item_ids(payload.get("exclude_item_ids", [])))
            limit = int(payload.get("limit", 5))
            items = RECOMMENDER.recommend(cart_item_ids=cart_ids, exclude_ids=exclude_ids, limit=limit)
            self._json_response({"items": items})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length == 0:
            return {}
        raw_body = self.rfile.read(content_length)
        return json.loads(raw_body.decode("utf-8"))

    def _json_response(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _extract_query_list(self, params: dict[str, list[str]], key: str) -> set[str]:
        values = params.get(key, [])
        return set(value for value in values if value in RECOMMENDER.item_meta)

    def _extract_query_int(self, params: dict[str, list[str]], key: str, default: int) -> int:
        try:
            return max(1, int(params.get(key, [str(default)])[0]))
        except ValueError:
            return default

    def _normalize_item_ids(self, values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        return [value for value in values if isinstance(value, str) and value in RECOMMENDER.item_meta]


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 19 grocery recommendation demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8019)
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), GroceryDemoHandler)
    print(f"Grocery recommendation demo is running on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
