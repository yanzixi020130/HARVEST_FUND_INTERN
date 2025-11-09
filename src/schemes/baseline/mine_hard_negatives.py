"""
Script to mine hard negative chunk candidates for each query using FAISS ANN search.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


def setup_logging(log_file: Optional[str]) -> None:
    """Configure logging to console and optional file."""
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        handlers=handlers,
    )


def load_chunks(path: str) -> Tuple[List[str], List[str]]:
    """Load chunk ids and texts from a JSONL file."""
    chunk_ids: List[str] = []
    chunk_texts: List[str] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if "chunk_id" not in record or "text" not in record:
                raise ValueError(f"Missing fields in chunk record at {path}:{line_no}")
            chunk_ids.append(str(record["chunk_id"]))
            chunk_texts.append(str(record["text"]))
    if not chunk_ids:
        raise ValueError(f"No chunks loaded from {path}")
    return chunk_ids, chunk_texts


def load_positive_pairs(path: str) -> List[Dict[str, Any]]:
    """Load positive query-chunk pairs."""
    pairs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            for key in ("query_id", "query", "positive_chunk_id"):
                if key not in record:
                    raise ValueError(f"Missing '{key}' in pair at {path}:{line_no}")
            pairs.append(
                {
                    "query_id": str(record["query_id"]),
                    "query": str(record["query"]),
                    "positive_chunk_id": str(record["positive_chunk_id"]),
                }
            )
    if not pairs:
        raise ValueError(f"No positive pairs loaded from {path}")
    return pairs


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
) -> np.ndarray:
    """Encode texts into normalized float32 embeddings."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    return embeddings


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """L2 normalize embedding matrix."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build FAISS inner-product index from embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def mine_hard_candidates(
    chunks_path: str,
    pairs_path: str,
    output_path: str,
    model_name: str,
    top_k: int,
    batch_size: int,
    use_gpu: bool,
) -> None:
    chunk_ids, chunk_texts = load_chunks(chunks_path)
    LOGGER.info("Loaded %d chunks from %s", len(chunk_ids), chunks_path)

    model = SentenceTransformer(model_name)
    if use_gpu:
        model = model.to("cuda")

    chunk_embs = encode_texts(model, chunk_texts, batch_size)
    chunk_embs = l2_normalize(chunk_embs)
    index = build_faiss_index(chunk_embs)
    LOGGER.info("FAISS index built with dimension %d", chunk_embs.shape[1])

    LOGGER.info("Loading positive pairs from %s", pairs_path)
    pairs = load_positive_pairs(pairs_path)
    LOGGER.info("Loaded %d positive pairs from %s", len(pairs), pairs_path)

    queries = [pair["query"] for pair in pairs]
    query_embs = encode_texts(model, queries, batch_size)
    query_embs = l2_normalize(query_embs)

    search_k = min(top_k, len(chunk_ids))
    if search_k <= 0:
        raise ValueError("top_k must be positive and there must be at least one chunk.")
    _, indices = index.search(query_embs, search_k)

    total_candidates = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as writer:
        for row_idx, pair in enumerate(pairs):
            retrieved_ids = [chunk_ids[idx] for idx in indices[row_idx] if idx >= 0]
            candidate_neg_ids = [
                chunk_id for chunk_id in retrieved_ids if chunk_id != pair["positive_chunk_id"]
            ]
            total_candidates += len(candidate_neg_ids)
            record = {
                "query_id": pair["query_id"],
                "query": pair["query"],
                "positive_chunk_id": pair["positive_chunk_id"],
                "candidate_neg_ids": candidate_neg_ids,
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    avg_candidates = total_candidates / len(pairs)
    LOGGER.info("Average candidate negatives per query: %.2f", avg_candidates)
    LOGGER.info("Hard negative candidates saved to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine hard negative chunk candidates.")
    parser.add_argument(
        "--chunks_path",
        type=str,
        default="data_processed/chunks.jsonl",
        help="Path to chunks JSONL file.",
    )
    parser.add_argument(
        "--pairs_path",
        type=str,
        default="pairs/train_positive_pairs.jsonl",
        help="Path to positive pairs JSONL file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="scheme_data/baseline/hard_candidates.jsonl",
        help="Path to save mined candidates JSONL.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="SentenceTransformer model name for English text.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Number of nearest chunks to retrieve per query.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoding texts.",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Whether to run the encoder on GPU (cuda).",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/mine_hard_negatives.log",
        help="Optional path to write log output.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging(args.log_file)
    args = parse_args()
    mine_hard_candidates(
        chunks_path=args.chunks_path,
        pairs_path=args.pairs_path,
        output_path=args.output_path,
        model_name=args.model_name,
        top_k=args.top_k,
        batch_size=args.batch_size,
        use_gpu=args.use_gpu,
    )


if __name__ == "__main__":
    main()
