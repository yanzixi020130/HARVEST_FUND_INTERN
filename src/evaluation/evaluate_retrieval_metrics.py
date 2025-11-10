"""
Evaluate retrieval metrics for the baseline bi-encoder on eval queries.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    FAISS_AVAILABLE = False


LOGGER = logging.getLogger(__name__)


def init_logger(log_path: str) -> None:
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
    logging.info("Logger initialized at %s", log_path)


def select_device(use_gpu: bool) -> str:
    if not use_gpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # pragma: no cover
        return "mps"
    logging.warning("GPU requested but unavailable, falling back to CPU.")
    return "cpu"


def load_model(model_path: str, device: str) -> SentenceTransformer:
    model = SentenceTransformer(model_path, device=device)
    logging.info("Loaded model from %s on device %s", model_path, device)
    return model


def load_chunks(chunks_path: str) -> Tuple[List[str], List[str]]:
    chunk_ids: List[str] = []
    chunk_texts: List[str] = []
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
                raise ValueError(f"Missing chunk_id/text at {chunks_path}:{line_no}")
            chunk_ids.append(str(record["chunk_id"]))
            chunk_texts.append(str(record["text"]))
    if not chunk_ids:
        raise ValueError(f"No chunks loaded from {chunks_path}")
    logging.info("Loaded %d chunks from %s", len(chunk_ids), chunks_path)
    return chunk_ids, chunk_texts


def encode_documents(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def build_index(doc_embeddings: np.ndarray, use_faiss: bool):
    if use_faiss:
        if not FAISS_AVAILABLE:
            raise RuntimeError("Faiss not installed but --use_faiss specified.")
        dim = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(doc_embeddings)
        logging.info("Built FAISS IndexFlatIP with dim %d", dim)
        return index
    logging.info("Skipping FAISS index; will use brute-force similarity.")
    return None


def load_eval_queries(eval_path: str) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    with open(eval_path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {eval_path}:{line_no}") from exc
            for key in ("query_id", "query", "positive_chunk_ids"):
                if key not in record:
                    raise ValueError(f"Missing '{key}' at {eval_path}:{line_no}")
            queries.append(record)
    if not queries:
        raise ValueError(f"No evaluation queries found in {eval_path}")
    logging.info("Loaded %d evaluation queries from %s", len(queries), eval_path)
    return queries


def encode_queries(
    model: SentenceTransformer,
    queries: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    embeddings = model.encode(
        queries,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def search_top_k(
    index,
    query_embs: np.ndarray,
    doc_embeddings: np.ndarray,
    top_k: int,
    use_faiss: bool,
) -> np.ndarray:
    if use_faiss:
        _, indices = index.search(query_embs, top_k)
        return indices
    # Brute-force cosine (dot product since normalized)
    sims = np.matmul(query_embs, doc_embeddings.T)
    top_indices = np.argpartition(-sims, kth=min(top_k, sims.shape[1] - 1), axis=1)[:, :top_k]
    # Ensure order by similarity
    sorted_indices = np.take_along_axis(
        top_indices,
        np.argsort(-np.take_along_axis(sims, top_indices, axis=1), axis=1),
        axis=1,
    )
    return sorted_indices


def compute_metrics_for_all(
    queries: List[Dict[str, Any]],
    retrieved_indices: np.ndarray,
    chunk_ids: Sequence[str],
    top_k: int,
) -> Dict[str, float]:
    precisions = []
    recalls = []
    f1_scores = []
    mrr_scores = []
    ndcg_scores = []

    for q_idx, query in enumerate(queries):
        positives = set(map(str, query["positive_chunk_ids"]))
        retrieved = [chunk_ids[idx] for idx in retrieved_indices[q_idx]]
        hits = [1 if doc_id in positives else 0 for doc_id in retrieved]
        hit_count = sum(hits)
        p_at_k = hit_count / top_k
        r_at_k = hit_count / max(1, len(positives))
        if hit_count == 0 or len(positives) == 0:
            f1 = 0.0
        else:
            f1 = (2 * p_at_k * r_at_k) / (p_at_k + r_at_k) if (p_at_k + r_at_k) else 0.0
        precisions.append(p_at_k)
        recalls.append(r_at_k)
        f1_scores.append(f1)

        mrr = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                mrr = 1.0 / rank
                break
        mrr_scores.append(mrr)

        # nDCG
        dcg = 0.0
        for i, hit in enumerate(hits, start=1):
            if hit:
                dcg += 1.0 / math.log2(i + 1)
        ideal_hits = min(len(positives), top_k)
        if ideal_hits == 0:
            idcg = 1.0
        else:
            idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    return {
        "Precision@{}".format(top_k): float(np.mean(precisions)),
        "Recall@{}".format(top_k): float(np.mean(recalls)),
        "F1@{}".format(top_k): float(np.mean(f1_scores)),
        "MRR@{}".format(top_k): float(np.mean(mrr_scores)),
        "nDCG@{}".format(top_k): float(np.mean(ndcg_scores)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics for bi-encoder.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/baseline/bi_encoder",
        help="Path to trained sentence-transformer model.",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default="data_eval/eval_pairs.jsonl",
        help="Evaluation JSONL file.",
    )
    parser.add_argument(
        "--chunks_path",
        type=str,
        default="data_processed/chunks.jsonl",
        help="Chunks JSONL corpus.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of documents to retrieve per query.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="scheme_data/baseline/metrics_retrieval.json",
        help="Output JSON path for metrics.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="reports/logs/evaluate_retrieval_metrics.log",
        help="Log file path.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoding.",
    )
    parser.add_argument(
        "--use_faiss",
        action="store_true",
        help="Use FAISS ANN index for retrieval.",
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
    logging.info("Evaluation run started at %s", datetime.utcnow().isoformat())
    logging.info("Arguments: %s", vars(args))

    device = select_device(args.use_gpu)
    model = load_model(args.model_path, device)

    chunk_ids, chunk_texts = load_chunks(args.chunks_path)
    doc_embeddings = encode_documents(model, chunk_texts, args.batch_size)
    index = build_index(doc_embeddings, args.use_faiss)

    queries = load_eval_queries(args.eval_path)
    query_texts = [q["query"] for q in queries]
    query_embs = encode_queries(model, query_texts, args.batch_size)

    retrieved_indices = search_top_k(index, query_embs, doc_embeddings, args.top_k, args.use_faiss)
    metrics = compute_metrics_for_all(queries, retrieved_indices, chunk_ids, args.top_k)
    metrics.update(
        {
            "TopK": args.top_k,
            "NumQueries": len(queries),
            "ModelPath": args.model_path,
            "EvalFile": args.eval_path,
        }
    )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    logging.info("Metrics saved to %s", args.output_path)
    logging.info("Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
