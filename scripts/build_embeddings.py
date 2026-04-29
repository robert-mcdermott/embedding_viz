#!/usr/bin/env python3
"""Build a clustered 3D embedding atlas from a word list."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import umap
from sklearn.cluster import MiniBatchKMeans


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "In-Search-of-Lost-Time-words.txt"
DEFAULT_OUTPUT = ROOT / "public" / "data" / "embedding-map.json"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

PALETTE = [
    "#ff5a67",
    "#22d3a6",
    "#ffd166",
    "#65a3ff",
    "#f97316",
    "#c084fc",
    "#2dd4bf",
    "#f43f8d",
    "#a3e635",
    "#fb7185",
    "#38bdf8",
    "#facc15",
    "#34d399",
    "#e879f9",
    "#f59e0b",
    "#60a5fa",
    "#ef4444",
    "#14b8a6",
    "#d9f99d",
    "#f472b6",
    "#93c5fd",
    "#fde047",
    "#4ade80",
    "#fb923c",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed words, project them to 3D, and write the browser data file."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--clusters", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_words(path: Path, max_words: int | None) -> list[str]:
    seen: set[str] = set()
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.strip()
        if not word:
            continue
        key = word.casefold()
        if key in seen:
            continue
        seen.add(key)
        words.append(word)
        if max_words is not None and len(words) >= max_words:
            break
    return words


def file_digest(words: list[str], model_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    for word in words:
        digest.update(b"\0")
        digest.update(word.encode("utf-8"))
    return digest.hexdigest()[:16]


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_or_create_embeddings(
    words: list[str],
    model_name: str,
    batch_size: int,
    device: str,
    cache_dir: Path,
    force: bool,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = file_digest(words, model_name)
    cache_path = cache_dir / f"{safe_name(model_name)}-{digest}.npy"

    if cache_path.exists() and not force:
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print(f"Loading embedding model {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        words,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    np.save(cache_path, embeddings)
    print(f"Cached embeddings at {cache_path}")
    return embeddings


def project_embeddings(embeddings: np.ndarray, seed: int) -> tuple[np.ndarray, umap.UMAP]:
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=28,
        min_dist=0.035,
        spread=2.4,
        metric="cosine",
        random_state=seed,
        low_memory=True,
        verbose=True,
    )
    projection = reducer.fit_transform(embeddings).astype(np.float32)

    center = np.median(projection, axis=0)
    projection = projection - center
    scale = float(np.percentile(np.linalg.norm(projection, axis=1), 98))
    projection = projection / max(scale, 1e-6) * 34.0
    return projection.astype(np.float32), reducer


def cluster_embeddings(
    embeddings: np.ndarray, words: list[str], cluster_count: int, seed: int
) -> tuple[np.ndarray, list[dict[str, object]]]:
    cluster_count = max(4, min(cluster_count, len(words)))
    kmeans = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=seed,
        batch_size=4096,
        n_init="auto",
        max_no_improvement=20,
    )
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_.astype(np.float32)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)

    clusters: list[dict[str, object]] = []
    for cluster_id in range(cluster_count):
        member_indices = np.flatnonzero(labels == cluster_id)
        if len(member_indices) == 0:
            representatives: list[str] = []
        else:
            scores = embeddings[member_indices] @ centers[cluster_id]
            top_members = member_indices[np.argsort(scores)[-8:]][::-1]
            representatives = [words[index] for index in top_members]

        clusters.append(
            {
                "id": cluster_id,
                "label": " / ".join(representatives[:3]) if representatives else f"Cluster {cluster_id + 1}",
                "terms": representatives,
                "count": int(len(member_indices)),
                "color": PALETTE[cluster_id % len(PALETTE)],
            }
        )

    clusters.sort(key=lambda item: int(item["id"]))
    return labels.astype(np.int16), clusters


def semantic_neighbors(reducer: umap.UMAP, limit: int = 6) -> list[list[list[float | int]]]:
    graph = reducer.graph_.tocsr()
    neighbors: list[list[list[float | int]]] = []
    for row in range(graph.shape[0]):
        start, end = graph.indptr[row], graph.indptr[row + 1]
        indices = graph.indices[start:end]
        weights = graph.data[start:end]
        order = np.argsort(weights)[-limit:][::-1]
        row_neighbors: list[list[float | int]] = []
        for item in order:
            neighbor_index = int(indices[item])
            if neighbor_index == row:
                continue
            row_neighbors.append([neighbor_index, round(float(weights[item]), 4)])
            if len(row_neighbors) >= limit:
                break
        neighbors.append(row_neighbors)
    return neighbors


def write_payload(
    output: Path,
    words: list[str],
    projection: np.ndarray,
    labels: np.ndarray,
    clusters: list[dict[str, object]],
    neighbors: list[list[list[float | int]]],
    model_name: str,
    embedding_dimensions: int,
    seed: int,
) -> None:
    points = []
    for index, word in enumerate(words):
        x, y, z = projection[index]
        points.append(
            {
                "w": word,
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "z": round(float(z), 4),
                "c": int(labels[index]),
                "n": neighbors[index],
            }
        )

    bounds = {
        "min": [round(float(value), 4) for value in projection.min(axis=0)],
        "max": [round(float(value), 4) for value in projection.max(axis=0)],
    }
    payload = {
        "meta": {
            "source": "In-Search-of-Lost-Time-words.txt",
            "model": model_name,
            "embeddingDimensions": embedding_dimensions,
            "projection": "UMAP 3D, cosine metric",
            "clustering": "MiniBatchKMeans on normalized embedding vectors",
            "seed": seed,
            "count": len(points),
            "bounds": bounds,
        },
        "clusters": clusters,
        "points": points,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(points):,} points to {output} ({size_mb:.2f} MB)")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-home"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    words = read_words(args.input, args.max_words)
    if not words:
        raise SystemExit(f"No words found in {args.input}")

    print(f"Loaded {len(words):,} words from {args.input}")
    device = choose_device(args.device)
    cache_dir = ROOT / ".cache" / "embeddings"
    embeddings = load_or_create_embeddings(
        words=words,
        model_name=args.model,
        batch_size=args.batch_size,
        device=device,
        cache_dir=cache_dir,
        force=args.force,
    )
    projection, reducer = project_embeddings(embeddings, args.seed)
    labels, clusters = cluster_embeddings(embeddings, words, args.clusters, args.seed)
    neighbors = semantic_neighbors(reducer)
    write_payload(
        output=args.output,
        words=words,
        projection=projection,
        labels=labels,
        clusters=clusters,
        neighbors=neighbors,
        model_name=args.model,
        embedding_dimensions=int(embeddings.shape[1]),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
