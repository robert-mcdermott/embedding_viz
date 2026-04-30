#!/usr/bin/env python3
"""Build clustered 3D embedding atlases from word lists or full text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import umap
from sklearn.cluster import MiniBatchKMeans


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORD_INPUT = ROOT / "In-Search-of-Lost-Time-words.txt"
DEFAULT_SENTENCE_INPUT = ROOT / "In-Search-of-Lost-Time.pdf"
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
    "#5eead4",
    "#a78bfa",
    "#fda4af",
    "#86efac",
    "#7dd3fc",
    "#fde68a",
    "#f0abfc",
    "#c4b5fd",
]

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "almost",
    "also",
    "although",
    "am",
    "among",
    "an",
    "and",
    "another",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "down",
    "each",
    "even",
    "ever",
    "every",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "like",
    "me",
    "more",
    "most",
    "my",
    "no",
    "not",
    "now",
    "of",
    "on",
    "one",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "up",
    "upon",
    "us",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}

ABBREVIATIONS = {
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Sr.",
    "Jr.",
    "St.",
    "Mt.",
    "M.",
    "MM.",
    "Mme.",
    "Mlle.",
    "Vol.",
    "vol.",
    "vols.",
    "No.",
    "etc.",
    "e.g.",
    "i.e.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed text units, project them to 3D, and write a browser dataset."
    )
    parser.add_argument("--unit", choices=("words", "sentences"), default="words")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--title", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--clusters", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-words", type=int, default=None, help="Deprecated alias for --max-items.")
    parser.add_argument("--min-sentence-words", type=int, default=4)
    parser.add_argument("--start-at", default=None, help="For text/PDF input, discard text before this marker.")
    parser.add_argument("--start-after", default=None, help="For text/PDF input, discard text through this marker.")
    parser.add_argument("--end-before", default=None, help="For text/PDF input, discard text from this marker onward.")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Extract and count items without embedding.")
    return parser.parse_args()


def default_input(unit: str) -> Path:
    return DEFAULT_SENTENCE_INPUT if unit == "sentences" else DEFAULT_WORD_INPUT


def read_words(path: Path, max_items: int | None) -> list[str]:
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
        if max_items is not None and len(words) >= max_items:
            break
    return words


def extract_text(path: Path) -> str:
    if path.suffix.casefold() != ".pdf":
        return path.read_text(encoding="utf-8")

    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        result = subprocess.run(
            [pdftotext, "-layout", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise SystemExit(
            "PDF input requires Poppler's pdftotext command or the optional pypdf package."
        ) from error

    reader = PdfReader(path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def normalize_text(text: str) -> str:
    text = text.replace("\x0c", "\n\n")
    text = re.sub(r"([A-Za-z])- *\n+ *([a-z])", r"\1\2", text)
    lines = [line.strip() for line in text.splitlines()]
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if re.fullmatch(r"\d+", line):
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def protect_sentence_boundaries(text: str) -> str:
    protected = text
    for abbr in sorted(ABBREVIATIONS, key=len, reverse=True):
        protected = protected.replace(abbr, abbr.replace(".", "<prd>"))
    protected = re.sub(r"\b([A-Z])\.", r"\1<prd>", protected)
    return protected


def split_sentences(text: str, min_words: int, max_items: int | None) -> list[str]:
    normalized = normalize_text(text)
    protected = protect_sentence_boundaries(normalized)
    raw_sentences = re.split(r"(?<=[.!?])(?:[\"')\]]+)?\s+(?=[\"'(\[]?[A-Z0-9])", protected)

    sentences: list[str] = []
    for sentence in raw_sentences:
        sentence = sentence.replace("<prd>", ".")
        sentence = re.sub(r"\s+", " ", sentence).strip()
        sentence = sentence.strip(" \t\n\r")
        if not sentence:
            continue
        words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", sentence)
        if len(words) < min_words:
            continue
        sentences.append(sentence)
        if max_items is not None and len(sentences) >= max_items:
            break
    return sentences


def trim_text(text: str, start_at: str | None, start_after: str | None, end_before: str | None) -> str:
    lowered = text.casefold()
    if start_at:
        index = lowered.find(start_at.casefold())
        if index == -1:
            raise SystemExit(f"Could not find --start-at marker: {start_at}")
        text = text[index:]
        lowered = text.casefold()
    if start_after:
        index = lowered.find(start_after.casefold())
        if index == -1:
            raise SystemExit(f"Could not find --start-after marker: {start_after}")
        text = text[index + len(start_after) :]
        lowered = text.casefold()
    if end_before:
        index = lowered.find(end_before.casefold())
        if index == -1:
            raise SystemExit(f"Could not find --end-before marker: {end_before}")
        text = text[:index]
    return text


def read_items(args: argparse.Namespace) -> tuple[list[str], Path]:
    input_path = args.input or default_input(args.unit)
    max_items = args.max_items if args.max_items is not None else args.max_words
    if args.unit == "words":
        return read_words(input_path, max_items), input_path

    text = trim_text(extract_text(input_path), args.start_at, args.start_after, args.end_before)
    return split_sentences(text, args.min_sentence_words, max_items), input_path


def file_digest(items: list[str], model_name: str, unit: str) -> str:
    digest = hashlib.sha256()
    digest.update(unit.encode("utf-8"))
    digest.update(model_name.encode("utf-8"))
    for item in items:
        digest.update(b"\0")
        digest.update(item.encode("utf-8"))
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
    items: list[str],
    unit: str,
    model_name: str,
    batch_size: int,
    device: str,
    cache_dir: Path,
    force: bool,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = file_digest(items, model_name, unit)
    cache_path = cache_dir / f"{safe_name(model_name)}-{unit}-{digest}.npy"

    if cache_path.exists() and not force:
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print(f"Loading embedding model {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        items,
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


def label_for_cluster(items: list[str], unit: str) -> str:
    if not items:
        return "Cluster"
    if unit == "words":
        return " / ".join(items[:3])

    tokens: list[str] = []
    for item in items:
        tokens.extend(
            token.casefold()
            for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", item)
            if token.casefold() not in STOPWORDS
        )
    top = [word for word, _ in Counter(tokens).most_common(3)]
    if top:
        return " / ".join(top)
    return " / ".join(short_text(item, 28) for item in items[:2])


def cluster_embeddings(
    embeddings: np.ndarray, items: list[str], unit: str, cluster_count: int, seed: int
) -> tuple[np.ndarray, list[dict[str, object]]]:
    cluster_count = max(4, min(cluster_count, len(items)))
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
            representatives = [items[index] for index in top_members]

        clusters.append(
            {
                "id": cluster_id,
                "label": label_for_cluster(representatives, unit),
                "terms": [short_text(item, 180) for item in representatives],
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


def short_text(value: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "..."


def default_title(source: Path, unit: str) -> str:
    base = source.stem.replace("-", " ").replace("_", " ").strip().title()
    return f"{base} {unit.title()}"


def write_payload(
    output: Path,
    items: list[str],
    unit: str,
    title: str,
    source: Path,
    projection: np.ndarray,
    labels: np.ndarray,
    clusters: list[dict[str, object]],
    neighbors: list[list[list[float | int]]],
    model_name: str,
    embedding_dimensions: int,
    seed: int,
) -> None:
    points = []
    for index, item in enumerate(items):
        x, y, z = projection[index]
        points.append(
            {
                "w": item,
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
            "title": title,
            "source": source.name,
            "unit": unit,
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
    print(f"Wrote {len(points):,} {unit} to {output} ({size_mb:.2f} MB)")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-home"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    items, source = read_items(args)
    if not items:
        raise SystemExit(f"No {args.unit} found in {source}")

    title = args.title or default_title(source, args.unit)
    if args.dry_run:
        print(f"Extracted {len(items):,} {args.unit} from {source}")
        for item in items[:5]:
            print(f"- {short_text(item, 240)}")
        return

    cluster_count = args.clusters if args.clusters is not None else (32 if args.unit == "sentences" else 24)
    print(f"Loaded {len(items):,} {args.unit} from {source}")
    device = choose_device(args.device)
    cache_dir = ROOT / ".cache" / "embeddings"
    embeddings = load_or_create_embeddings(
        items=items,
        unit=args.unit,
        model_name=args.model,
        batch_size=args.batch_size,
        device=device,
        cache_dir=cache_dir,
        force=args.force,
    )
    projection, reducer = project_embeddings(embeddings, args.seed)
    labels, clusters = cluster_embeddings(embeddings, items, args.unit, cluster_count, args.seed)
    neighbors = semantic_neighbors(reducer)
    write_payload(
        output=args.output,
        items=items,
        unit=args.unit,
        title=title,
        source=source,
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
