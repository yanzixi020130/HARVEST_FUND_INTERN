"""
Compute embedding-based quality metrics for mined hard negatives.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def init_logger(log_path: str) -> None:
    """Configure logging output to file and stdout."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    handlers = [
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    logging.info("Logger initialized. Writing logs to %s", log_path)


def load_chunks(chunks_path: str) -> Dict[str, str]:
    """Load chunk_id to text mapping."""
    chunk_texts: Dict[str, str] = {}
    with open(chunks_path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {chunks_path}:{line_no}") from exc
            if "chunk_id" not in record or "text" not in record:
                raise ValueError(f"Missing fields in chunk record at {chunks_path}:{line_no}")
            chunk_texts[str(record["chunk_id"])] = str(record["text"])
    if not chunk_texts:
        raise ValueError(f"No chunk data loaded from {chunks_path}")
    logging.info("Loaded %d chunks from %s", len(chunk_texts), chunks_path)
    return chunk_texts


def load_hard_negatives(path: str) -> List[Dict[str, Any]]:
    """Load hard negative records."""
    samples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            for key in ("query_id", "query", "positive_chunk_id", "negatives_hard"):
                if key not in record:
                    raise ValueError(f"Missing '{key}' in record at {path}:{line_no}")
            if not isinstance(record["negatives_hard"], list):
                raise ValueError(f"'negatives_hard' must be a list at {path}:{line_no}")
            samples.append(record)
    if not samples:
        raise ValueError(f"No hard negative samples loaded from {path}")
    logging.info("Loaded %d hard-negative records from %s", len(samples), path)
    return samples


def detect_device(use_gpu: bool) -> str:
    """Resolve device string based on availability and flag."""
    if not use_gpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    logging.warning("GPU requested but not available; falling back to CPU.")
    return "cpu"


def load_encoder(model_name_or_path: str, device: str) -> SentenceTransformer:
    """Load sentence transformer from local path or hub."""
    if os.path.isdir(model_name_or_path):
        logging.info("Loading encoder from local directory: %s", model_name_or_path)
        model = SentenceTransformer(model_name_or_path, device=device)
    else:
        logging.info(
            "Model path not found locally, loading pretrained encoder: %s",
            model_name_or_path,
        )
        model = SentenceTransformer(model_name_or_path, device=device)
    logging.info("Encoder ready on device: %s", device)
    return model


def encode_texts(
    model: SentenceTransformer,
    items: Dict[str, str],
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """Encode a dict of texts, return normalized embeddings keyed by original ids."""
    if not items:
        return {}
    keys = list(items.keys())
    texts = [items[key] for key in keys]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    embeddings = embeddings.astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    return {key: embeddings[idx] for idx, key in enumerate(keys)}


def compute_metrics(
    samples: List[Dict[str, Any]],
    query_embs: Dict[str, np.ndarray],
    chunk_embs: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Compute AvgSim, AvgMargin, NumQueries, NegativesPerQueryMean."""
    all_neg_sims: List[float] = []
    all_margins: List[float] = []
    query_with_negatives = 0
    total_negatives = 0

    for sample in samples:
        neg_entries = sample.get("negatives_hard", [])
        if not neg_entries:
            continue
        query_id = sample["query_id"]
        pos_id = sample["positive_chunk_id"]
        if query_id not in query_embs or pos_id not in chunk_embs:
            logging.warning("Missing embeddings for query %s or positive %s", query_id, pos_id)
            continue
        q_emb = query_embs[query_id]
        pos_emb = chunk_embs[pos_id]
        s_pos = float(np.dot(q_emb, pos_emb))

        neg_sims = []
        margins = []
        for neg_entry in neg_entries:
            neg_id = neg_entry.get("chunk_id")
            if not neg_id or neg_id not in chunk_embs:
                logging.warning("Negative chunk %s missing embedding for query %s", neg_id, query_id)
                continue
            neg_emb = chunk_embs[neg_id]
            s_neg = float(np.dot(q_emb, neg_emb))
            neg_sims.append(s_neg)
            margins.append(s_pos - s_neg)

        if neg_sims:
            all_neg_sims.extend(neg_sims)
            all_margins.extend(margins)
            total_negatives += len(neg_sims)
            query_with_negatives += 1

    if not all_neg_sims:
        logging.warning("No negative similarities computed; metrics will be zero.")
        avg_sim = 0.0
        avg_margin = 0.0
    else:
        avg_sim = float(np.mean(all_neg_sims))
        avg_margin = float(np.mean(all_margins))

    negatives_per_query = (
        total_negatives / query_with_negatives if query_with_negatives else 0.0
    )
    return {
        "AvgSim": avg_sim,
        "AvgMargin": avg_margin,
        "NumQueries": query_with_negatives,
        "NegativesPerQueryMean": negatives_per_query,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hard negative embeddings.")
    parser.add_argument(
        "--hard_neg_path",
        type=str,
        default="scheme_data/baseline/hard_negatives.jsonl",
        help="Path to hard negatives JSONL file.",
    )
    parser.add_argument(
        "--chunks_path",
        type=str,
        default="data_processed/chunks.jsonl",
        help="Path to chunk text JSONL file.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="models/baseline/bi_encoder",
        help="Local directory or pretrained model name.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="scheme_data/baseline/metrics_embedding.json",
        help="Output JSON path for metrics.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="reports/logs/compute_embedding_metrics.log",
        help="Log file path.",
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
        help="Use GPU/MPS if available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_logger(args.log_path)
    logging.info("Starting embedding metrics computation at %s", datetime.utcnow().isoformat())
    logging.info("Arguments: %s", vars(args))

    chunk_texts = load_chunks(args.chunks_path)
    hard_samples = load_hard_negatives(args.hard_neg_path)

    # Collect unique texts to encode.
    queries_to_encode: Dict[str, str] = {}
    chunks_to_encode: Dict[str, str] = {}
    for sample in hard_samples:
        queries_to_encode[sample["query_id"]] = sample["query"]
        pos_id = sample["positive_chunk_id"]
        if pos_id in chunk_texts:
            chunks_to_encode[pos_id] = chunk_texts[pos_id]
        else:
            logging.warning("Positive chunk %s missing from chunk file", pos_id)
        for neg in sample.get("negatives_hard", []):
            neg_id = neg.get("chunk_id")
            if neg_id and neg_id in chunk_texts:
                chunks_to_encode[neg_id] = chunk_texts[neg_id]
            elif neg_id:
                logging.warning("Negative chunk %s missing from chunk file", neg_id)

    device = detect_device(args.use_gpu)
    encoder = load_encoder(args.model_name_or_path, device=device)

    query_embs = encode_texts(encoder, queries_to_encode, args.batch_size)
    chunk_embs = encode_texts(encoder, chunks_to_encode, args.batch_size)

    metrics = compute_metrics(hard_samples, query_embs, chunk_embs)
    metrics.update(
        {
            "ModelUsed": args.model_name_or_path,
            "HardNegFile": args.hard_neg_path,
        }
    )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    logging.info("Metrics saved to %s", args.output_path)
    logging.info("Computed metrics: %s", metrics)


if __name__ == "__main__":
    main()
