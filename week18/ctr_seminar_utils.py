from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import math
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


RANDOM_STATE = 42


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_data_root(start: Path | str = ".") -> Path:
    start = Path(start).resolve()
    for candidate in [start, *start.parents]:
        if all((candidate / name).exists() for name in ["interactions.csv", "items.csv", "users.csv"]):
            return candidate
    raise FileNotFoundError("Could not find interactions.csv/items.csv/users.csv")


def load_kion_data(data_root: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_root = Path(data_root)
    interactions = pd.read_csv(data_root / "interactions.csv", parse_dates=["last_watch_dt"])
    users = pd.read_csv(data_root / "users.csv")
    items = pd.read_csv(data_root / "items.csv")
    return interactions, users, items


def preprocess_kion(
    interactions: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interactions = interactions.copy()
    users = users.copy()
    items = items.copy()

    interactions["watched_pct"] = interactions["watched_pct"].fillna(0.0).astype(float)
    interactions["total_dur"] = interactions["total_dur"].fillna(0.0).astype(float)
    interactions["date"] = pd.to_datetime(interactions["last_watch_dt"]).dt.normalize()

    users["age"] = users["age"].fillna("unknown")
    users["income"] = users["income"].fillna("unknown")
    users["sex"] = users["sex"].fillna("unknown")
    users["kids_flg"] = users["kids_flg"].fillna(0).astype(int)

    items["content_type"] = items["content_type"].fillna("unknown")
    items["release_year"] = items["release_year"].fillna(0).astype(int)
    items["for_kids"] = items["for_kids"].fillna(0).astype(int)
    items["genres"] = items["genres"].fillna("unknown")
    items["directors"] = items["directors"].fillna("unknown")
    items["first_genre"] = items["genres"].map(lambda x: str(x).split(",")[0].strip() if str(x) else "unknown")
    items["first_director"] = items["directors"].map(lambda x: str(x).split(",")[0].strip() if str(x) else "unknown")
    items["release_year_bucket"] = pd.cut(
        items["release_year"],
        bins=[-1, 1980, 1990, 2000, 2010, 2020, 2100],
        labels=["<=1980", "1981-1990", "1991-2000", "2001-2010", "2011-2020", "2021+"],
    ).astype(str)
    return interactions, users, items


@dataclass
class SplitConfig:
    stage_days: int = 14
    min_history_days: int = 60
    positive_threshold: float = 50.0
    max_users_per_stage: int = 5000
    candidates_k: int = 80
    popular_fill_k: int = 30
    sequence_len: int = 30


@dataclass
class StageWindow:
    name: str
    history_end: pd.Timestamp
    target_start: pd.Timestamp
    target_end: pd.Timestamp


def make_stage_windows(interactions: pd.DataFrame, cfg: SplitConfig) -> list[StageWindow]:
    max_date = interactions["date"].max()
    test_end = max_date
    test_start = test_end - pd.Timedelta(days=cfg.stage_days - 1)
    val_end = test_start - pd.Timedelta(days=1)
    val_start = val_end - pd.Timedelta(days=cfg.stage_days - 1)
    train_end = val_start - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=cfg.stage_days - 1)

    return [
        StageWindow("train", train_start - pd.Timedelta(days=1), train_start, train_end),
        StageWindow("val", val_start - pd.Timedelta(days=1), val_start, val_end),
        StageWindow("test", test_start - pd.Timedelta(days=1), test_start, test_end),
    ]


class EASE:
    def __init__(self, l2: float = 500.0):
        self.l2 = l2
        self.B: np.ndarray | None = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.train_matrix: sp.csr_matrix | None = None

    def fit(self, interactions: pd.DataFrame) -> "EASE":
        df = interactions[["user_id", "item_id"]].drop_duplicates()
        user_idx = self.user_encoder.fit_transform(df["user_id"])
        item_idx = self.item_encoder.fit_transform(df["item_id"])
        X = sp.csr_matrix((np.ones(len(df)), (user_idx, item_idx)))
        G = (X.T @ X).toarray().astype(np.float64)
        diag = np.arange(G.shape[0])
        G[diag, diag] += self.l2
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diag, diag] = 0.0
        self.B = B
        self.train_matrix = X
        return self

    def recommend(
        self,
        history: pd.DataFrame,
        users: Sequence[int],
        top_k: int = 80,
        fill_items: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        assert self.B is not None and self.train_matrix is not None
        known_users = set(self.user_encoder.classes_)
        known_items = self.item_encoder.classes_
        fill_items = list(fill_items or [])

        history_pairs = (
            history[["user_id", "item_id"]]
            .drop_duplicates()
            .groupby("user_id")["item_id"]
            .agg(set)
            .to_dict()
        )

        rows = []
        for user in users:
            seen = history_pairs.get(user, set())
            if user in known_users:
                user_idx = self.user_encoder.transform([user])[0]
                scores = self.train_matrix[user_idx] @ self.B
                scores = np.asarray(scores).reshape(-1)
                seen_idx = []
                if seen:
                    seen_known = [x for x in seen if x in set(known_items)]
                    if seen_known:
                        seen_idx = self.item_encoder.transform(seen_known)
                        scores[seen_idx] = -np.inf
                current_k = min(top_k, len(scores))
                top_idx = np.argpartition(scores, -current_k)[-current_k:]
                top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
                candidates = [(int(known_items[i]), float(scores[i])) for i in top_idx if np.isfinite(scores[i])]
            else:
                candidates = []

            used = {item for item, _ in candidates}
            for item in fill_items:
                if len(candidates) >= top_k:
                    break
                if item in used or item in seen:
                    continue
                candidates.append((int(item), 0.0))
                used.add(item)

            for rank, (item_id, ease_score) in enumerate(candidates[:top_k], start=1):
                rows.append(
                    {
                        "user_id": int(user),
                        "item_id": int(item_id),
                        "ease_score": ease_score,
                        "ease_rank": rank,
                    }
                )

        return pd.DataFrame(rows, columns=["user_id", "item_id", "ease_score", "ease_rank"])


def _positive_pairs(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return (
        df.loc[df["watched_pct"] >= threshold, ["user_id", "item_id"]]
        .drop_duplicates()
        .assign(label=1)
    )


def _choose_stage_users(history: pd.DataFrame, target: pd.DataFrame, max_users: int) -> np.ndarray:
    hist_users = set(history["user_id"].unique())
    target_users = target["user_id"].value_counts().index.tolist()
    selected = [u for u in target_users if u in hist_users]
    return np.array(selected[:max_users], dtype=np.int64)


def _build_feature_tables(history: pd.DataFrame, items: pd.DataFrame) -> dict[str, pd.DataFrame]:
    items_meta = items[["item_id", "content_type", "first_genre", "first_director", "release_year", "release_year_bucket", "for_kids"]]
    hist = history.merge(items_meta, on="item_id", how="left")
    max_date = hist["date"].max()

    user_stats = hist.groupby("user_id").agg(
        user_hist_interactions=("item_id", "size"),
        user_hist_items=("item_id", "nunique"),
        user_avg_watch_pct=("watched_pct", "mean"),
        user_last_date=("date", "max"),
    )
    user_stats["user_days_since_last"] = (max_date - user_stats["user_last_date"]).dt.days.clip(lower=0)
    user_stats = user_stats.drop(columns=["user_last_date"]).reset_index()

    item_stats = hist.groupby("item_id").agg(
        item_hist_interactions=("user_id", "size"),
        item_hist_users=("user_id", "nunique"),
        item_avg_watch_pct=("watched_pct", "mean"),
        item_last_date=("date", "max"),
    )
    item_stats["item_days_since_last"] = (max_date - item_stats["item_last_date"]).dt.days.clip(lower=0)
    item_stats = item_stats.drop(columns=["item_last_date"]).reset_index()

    ui_stats = hist.groupby(["user_id", "item_id"]).agg(
        ui_hist_interactions=("date", "size"),
        ui_avg_watch_pct=("watched_pct", "mean"),
        ui_last_date=("date", "max"),
    )
    ui_stats["ui_days_since_last"] = (max_date - ui_stats["ui_last_date"]).dt.days.clip(lower=0)
    ui_stats = ui_stats.drop(columns=["ui_last_date"]).reset_index()

    ug_stats = hist.groupby(["user_id", "first_genre"]).agg(
        ug_hist_interactions=("date", "size"),
        ug_avg_watch_pct=("watched_pct", "mean"),
    ).reset_index()

    ud_stats = hist.groupby(["user_id", "first_director"]).agg(
        ud_hist_interactions=("date", "size"),
        ud_avg_watch_pct=("watched_pct", "mean"),
    ).reset_index()

    user_sequences = (
        hist.sort_values(["user_id", "date"])
        .groupby("user_id")["item_id"]
        .agg(list)
        .to_dict()
    )

    popularity = (
        hist.groupby("item_id")
        .size()
        .sort_values(ascending=False)
        .index
        .astype(int)
        .tolist()
    )

    return {
        "user_stats": user_stats,
        "item_stats": item_stats,
        "ui_stats": ui_stats,
        "ug_stats": ug_stats,
        "ud_stats": ud_stats,
        "user_sequences": user_sequences,
        "popular_items": popularity,
    }


def build_stage_dataset(
    interactions: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
    ease: EASE,
    stage: StageWindow,
    cfg: SplitConfig,
) -> pd.DataFrame:
    history = interactions.loc[interactions["date"] <= stage.history_end].copy()
    target = interactions.loc[(interactions["date"] >= stage.target_start) & (interactions["date"] <= stage.target_end)].copy()
    stage_users = _choose_stage_users(history, target, cfg.max_users_per_stage)
    history = history.loc[history["user_id"].isin(stage_users)].copy()
    target = target.loc[target["user_id"].isin(stage_users)].copy()

    tables = _build_feature_tables(history, items)
    candidates = ease.recommend(
        history=history,
        users=stage_users,
        top_k=cfg.candidates_k,
        fill_items=tables["popular_items"][: cfg.popular_fill_k],
    )

    positives = _positive_pairs(target, cfg.positive_threshold)
    dataset = candidates.merge(positives, on=["user_id", "item_id"], how="left")
    dataset["label"] = dataset["label"].fillna(0).astype(int)

    items_meta = items[
        ["item_id", "content_type", "first_genre", "first_director", "release_year", "release_year_bucket", "for_kids"]
    ].copy()
    dataset = dataset.merge(items_meta, on="item_id", how="left")
    dataset = dataset.merge(users, on="user_id", how="left")
    dataset = dataset.merge(tables["user_stats"], on="user_id", how="left")
    dataset = dataset.merge(tables["item_stats"], on="item_id", how="left")
    dataset = dataset.merge(tables["ui_stats"], on=["user_id", "item_id"], how="left")
    dataset = dataset.merge(tables["ug_stats"], on=["user_id", "first_genre"], how="left")
    dataset = dataset.merge(tables["ud_stats"], on=["user_id", "first_director"], how="left")

    dataset["history_items"] = dataset["user_id"].map(tables["user_sequences"])
    dataset["history_len"] = dataset["history_items"].map(lambda x: len(x) if isinstance(x, list) else 0)

    numeric_cols = [
        "ease_score",
        "ease_rank",
        "user_hist_interactions",
        "user_hist_items",
        "user_avg_watch_pct",
        "user_days_since_last",
        "item_hist_interactions",
        "item_hist_users",
        "item_avg_watch_pct",
        "item_days_since_last",
        "ui_hist_interactions",
        "ui_avg_watch_pct",
        "ui_days_since_last",
        "ug_hist_interactions",
        "ug_avg_watch_pct",
        "ud_hist_interactions",
        "ud_avg_watch_pct",
        "history_len",
        "release_year",
    ]
    for col in numeric_cols:
        dataset[col] = dataset[col].fillna(0.0)

    cat_cols = [
        "user_id",
        "item_id",
        "age",
        "income",
        "sex",
        "content_type",
        "first_genre",
        "first_director",
        "release_year_bucket",
        "for_kids",
        "kids_flg",
    ]
    for col in cat_cols:
        dataset[col] = dataset[col].fillna("unknown").astype(str)

    dataset["stage"] = stage.name
    return dataset.sort_values(["user_id", "ease_rank"]).reset_index(drop=True)


def evaluate_ranking(df: pd.DataFrame, score_col: str, k: int = 10) -> dict[str, float]:
    if df.empty:
        return {f"recall@{k}": 0.0, f"ndcg@{k}": 0.0}

    work = (
        df[["user_id", "label", score_col]]
        .sort_values(["user_id", score_col], ascending=[True, False], kind="mergesort")
    )
    topk = work.groupby("user_id", sort=False).head(k).copy()

    total_relevant = work.groupby("user_id", sort=False)["label"].sum()
    valid_users = total_relevant[total_relevant > 0].index
    if len(valid_users) == 0:
        return {f"recall@{k}": 0.0, f"ndcg@{k}": 0.0}

    hits = topk.groupby("user_id", sort=False)["label"].sum().reindex(valid_users, fill_value=0.0)
    recall = (hits / total_relevant.reindex(valid_users)).mean()

    topk["rank"] = topk.groupby("user_id", sort=False).cumcount() + 1
    topk["discount"] = 1.0 / np.log2(topk["rank"] + 1.0)
    topk["dcg_part"] = topk["label"] * topk["discount"]
    dcg = topk.groupby("user_id", sort=False)["dcg_part"].sum().reindex(valid_users, fill_value=0.0)

    ideal_counts = total_relevant.reindex(valid_users).clip(upper=k).astype(int)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    idcg = ideal_counts.map(lambda n: float(discounts[:n].sum()) if n > 0 else 0.0)
    ndcg = (dcg / idcg).replace([np.inf, -np.inf], 0.0).fillna(0.0).mean()

    return {
        f"recall@{k}": float(recall),
        f"ndcg@{k}": float(ndcg),
    }


def prepare_lgbm_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    cat_cols = [
        "age",
        "income",
        "sex",
        "content_type",
        "first_genre",
        "first_director",
        "release_year_bucket",
        "for_kids",
        "kids_flg",
    ]
    numeric_cols = [
        "ease_score",
        "ease_rank",
        "user_hist_interactions",
        "user_hist_items",
        "user_avg_watch_pct",
        "user_days_since_last",
        "item_hist_interactions",
        "item_hist_users",
        "item_avg_watch_pct",
        "item_days_since_last",
        "ui_hist_interactions",
        "ui_avg_watch_pct",
        "ui_days_since_last",
        "ug_hist_interactions",
        "ug_avg_watch_pct",
        "ud_hist_interactions",
        "ud_avg_watch_pct",
        "history_len",
        "release_year",
    ]

    def _encode(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, col: str) -> None:
        values = pd.concat([train[col], val[col], test[col]], axis=0).astype(str)
        encoder = LabelEncoder().fit(values)
        train[col] = encoder.transform(train[col].astype(str))
        val[col] = encoder.transform(val[col].astype(str))
        test[col] = encoder.transform(test[col].astype(str))

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    for col in cat_cols:
        _encode(train_df, val_df, test_df, col)

    feature_cols = numeric_cols + cat_cols
    return train_df, val_df, test_df, feature_cols, cat_cols


def fit_lgbm_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Sequence[str],
    cat_cols: Sequence[str],
) -> tuple[LGBMClassifier, LGBMRanker]:
    clf = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )
    clf.fit(
        train_df[list(feature_cols)],
        train_df["label"],
        eval_set=[(val_df[list(feature_cols)], val_df["label"])],
        eval_metric="binary_logloss",
        categorical_feature=list(cat_cols),
    )

    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )
    train_group = train_df.groupby("user_id").size().sort_index().to_numpy()
    val_group = val_df.groupby("user_id").size().sort_index().to_numpy()
    ranker.fit(
        train_df.sort_values("user_id")[list(feature_cols)],
        train_df.sort_values("user_id")["label"],
        group=train_group,
        eval_set=[(val_df.sort_values("user_id")[list(feature_cols)], val_df.sort_values("user_id")["label"])],
        eval_group=[val_group],
        eval_at=[5, 10, 20],
        categorical_feature=list(cat_cols),
    )
    return clf, ranker


class FeatureEncoder:
    def __init__(self, cat_cols: Sequence[str], num_cols: Sequence[str], seq_col: str = "history_items"):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.seq_col = seq_col
        self.cat_maps: dict[str, dict[str, int]] = {}
        self.item_map: dict[str, int] = {}

    def fit(self, frames: Iterable[pd.DataFrame]) -> "FeatureEncoder":
        frames = list(frames)
        for col in self.cat_cols:
            values = pd.concat([df[col].astype(str) for df in frames], axis=0).drop_duplicates().tolist()
            self.cat_maps[col] = {value: idx + 1 for idx, value in enumerate(values)}
        item_values = pd.concat([df["item_id"].astype(str) for df in frames], axis=0).drop_duplicates().tolist()
        for df in frames:
            seq_values = df[self.seq_col].dropna().tolist()
            for seq in seq_values:
                item_values.extend([str(x) for x in seq])
        item_values = pd.Series(item_values).drop_duplicates().tolist()
        self.item_map = {value: idx + 1 for idx, value in enumerate(item_values)}
        return self

    def transform_row(self, row: pd.Series, seq_len: int) -> dict[str, np.ndarray | int | float]:
        cat = np.array([self.cat_maps[col].get(str(row[col]), 0) for col in self.cat_cols], dtype=np.int64)
        num = np.array([float(row[col]) for col in self.num_cols], dtype=np.float32)
        item_id = self.item_map.get(str(row["item_id"]), 0)
        raw_seq = row[self.seq_col] if isinstance(row[self.seq_col], list) else []
        seq = [self.item_map.get(str(x), 0) for x in raw_seq[-seq_len:]]
        seq = [0] * (seq_len - len(seq)) + seq
        return {"cat": cat, "num": num, "item": item_id, "seq": np.array(seq, dtype=np.int64)}


class CTRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, encoder: FeatureEncoder, seq_len: int = 30):
        self.df = df.reset_index(drop=True)
        self.encoder = encoder
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        encoded = self.encoder.transform_row(row, self.seq_len)
        return {
            "cat": torch.tensor(encoded["cat"], dtype=torch.long),
            "num": torch.tensor(encoded["num"], dtype=torch.float32),
            "item": torch.tensor(encoded["item"], dtype=torch.long),
            "seq": torch.tensor(encoded["seq"], dtype=torch.long),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
        }


class CrossLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * self.w(x) + self.b + x


class DCNv2Lite(nn.Module):
    def __init__(self, cat_cardinalities: Sequence[int], num_dim: int, embed_dim: int = 16, cross_layers: int = 3):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, embed_dim) for card in cat_cardinalities])
        base_dim = embed_dim * len(cat_cardinalities) + num_dim
        self.cross_layers = nn.ModuleList([CrossLayer(base_dim) for _ in range(cross_layers)])
        self.deep = nn.Sequential(
            nn.Linear(base_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.out = nn.Linear(base_dim + 64, 1)

    def forward(self, cat: torch.Tensor, num: torch.Tensor, item: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        embs = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x0 = torch.cat(embs + [num], dim=1)
        x = x0
        for layer in self.cross_layers:
            x = layer(x0, x)
        deep = self.deep(x0)
        return self.out(torch.cat([x, deep], dim=1)).squeeze(1)


class FinalNetLite(nn.Module):
    def __init__(self, cat_cardinalities: Sequence[int], num_dim: int, embed_dim: int = 16):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, embed_dim) for card in cat_cardinalities])
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * len(cat_cardinalities), 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim * len(cat_cardinalities)),
            nn.Sigmoid(),
        )
        total_dim = embed_dim * len(cat_cardinalities) + num_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, cat: torch.Tensor, num: torch.Tensor, item: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        embs = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
        flat = torch.cat(embs, dim=1)
        gated = flat * self.gate(flat)
        features = torch.cat([flat, num], dim=1)
        return self.mlp(torch.cat([features, torch.cat([gated, num], dim=1)], dim=1)).squeeze(1)


class TransActLite(nn.Module):
    def __init__(self, cat_cardinalities: Sequence[int], item_vocab: int, num_dim: int, embed_dim: int = 32, seq_len: int = 30):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, embed_dim) for card in cat_cardinalities])
        self.item_embedding = nn.Embedding(item_vocab, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * (len(cat_cardinalities) + 2) + num_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.seq_len = seq_len

    def forward(self, cat: torch.Tensor, num: torch.Tensor, item: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        cat_embs = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
        target_item = self.item_embedding(item)
        pos = self.position_embedding(torch.arange(self.seq_len, device=seq.device))
        seq_emb = self.item_embedding(seq) + pos.unsqueeze(0)
        mask = seq.eq(0)
        encoded = self.encoder(seq_emb, src_key_padding_mask=mask)
        valid = (~mask).float().unsqueeze(-1)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        features = torch.cat(cat_embs + [target_item, pooled, num], dim=1)
        return self.mlp(features).squeeze(1)


def train_torch_model(
    model: nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    encoder: FeatureEncoder,
    epochs: int = 3,
    batch_size: int = 1024,
    lr: float = 1e-3,
    seq_len: int = 30,
    device: str | None = None,
) -> nn.Module:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loader = DataLoader(CTRDataset(train_df, encoder, seq_len), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CTRDataset(val_df, encoder, seq_len), batch_size=batch_size, shuffle=False)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_ndcg = -math.inf

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optim.zero_grad()
            logits = model(
                batch["cat"].to(device),
                batch["num"].to(device),
                batch["item"].to(device),
                batch["seq"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            optim.step()

        scores = predict_torch_model(model, val_df, encoder, batch_size=batch_size, seq_len=seq_len, device=device)
        ndcg = evaluate_ranking(val_df.assign(_score=scores), "_score")["ndcg@10"]
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_torch_model(
    model: nn.Module,
    df: pd.DataFrame,
    encoder: FeatureEncoder,
    batch_size: int = 2048,
    seq_len: int = 30,
    device: str | None = None,
) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    loader = DataLoader(CTRDataset(df, encoder, seq_len), batch_size=batch_size, shuffle=False)
    preds = []
    for batch in loader:
        logits = model(
            batch["cat"].to(device),
            batch["num"].to(device),
            batch["item"].to(device),
            batch["seq"].to(device),
        )
        preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds)


def build_leaderboard(scored_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_name, df in scored_frames.items():
        metrics = evaluate_ranking(df, "score")
        rows.append({"model": model_name, **metrics})
    return pd.DataFrame(rows).sort_values(["ndcg@10", "recall@10"], ascending=False).reset_index(drop=True)
