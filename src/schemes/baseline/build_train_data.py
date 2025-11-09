"""
Generate hard negative records and bi-encoder training data from mined candidates.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import CrossEncoder
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


def load_chunks(chunks_path: str) -> Dict[str, str]:
    """Load chunk_id -> text map from JSONL."""
    chunk_text_map: Dict[str, str] = {}
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
            chunk_text_map[str(record["chunk_id"])] = str(record["text"])
    if not chunk_text_map:
        raise ValueError(f"No chunks loaded from {chunks_path}")
    return chunk_text_map


def load_hard_candidates(path: str) -> List[Dict[str, Any]]:
    """Load hard candidate samples."""
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
            for key in ("query_id", "query", "positive_chunk_id", "candidate_neg_ids"):
                if key not in record:
                    raise ValueError(f"Missing '{key}' in candidate at {path}:{line_no}")
            if not isinstance(record["candidate_neg_ids"], list):
                raise ValueError(
                    f"'candidate_neg_ids' must be a list at {path}:{line_no}"
                )
            samples.append(
                {
                    "query_id": str(record["query_id"]),
                    "query": str(record["query"]),
                    "positive_chunk_id": str(record["positive_chunk_id"]),
                    "candidate_neg_ids": [str(cid) for cid in record["candidate_neg_ids"]],
                }
            )
    if not samples:
        raise ValueError(f"No candidate samples loaded from {path}")
    return samples


def build_cross_encoder(model_name: str, use_gpu: bool) -> CrossEncoder:
    """Instantiate a CrossEncoder, optionally on GPU."""
    model = CrossEncoder(model_name)
    if use_gpu:
        model = model.to("cuda")
    return model


def score_pairs(
    cross_encoder: CrossEncoder,
    pairs: List[Tuple[str, str]],
    batch_size: int,
) -> List[float]:
    """Score query-chunk pairs via CrossEncoder."""
    if not pairs:
        return []
    scores = cross_encoder.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    if hasattr(scores, "tolist"):
        return scores.tolist()
    return list(scores)


def process_query_sample(
    sample: Dict[str, Any],
    chunk_text_map: Dict[str, str],
    cross_encoder: CrossEncoder,
    batch_size: int,
    max_negatives: int,
    min_neg_score: float,
    score_margin: float,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, int]]]:
    """Process one sample to derive hard negatives and training entry."""
    query_id = sample["query_id"]
    query = sample["query"]
    pos_chunk_id = sample["positive_chunk_id"]
    candidate_neg_ids: List[str] = sample.get("candidate_neg_ids", [])

    stats = {
        "total_candidates": len(candidate_neg_ids),
        "valid_candidates": 0,
        "missing_chunks": 0,
        "filtered_pseudo": 0,
        "filtered_easy": 0,
        "filtered_margin": 0,
        "hard_negatives": 0,
    }

    pos_text = chunk_text_map.get(pos_chunk_id)
    if pos_text is None:
        LOGGER.warning("Positive chunk %s missing for query %s, skip sample.", pos_chunk_id, query_id)
        return None

    neg_pairs: List[Tuple[str, str]] = []
    valid_neg_ids: List[str] = []
    for neg_id in candidate_neg_ids:
        neg_text = chunk_text_map.get(neg_id)
        if neg_text is None:
            stats["missing_chunks"] += 1
            LOGGER.warning("Negative chunk %s missing for query %s, ignore candidate.", neg_id, query_id)
            continue
        valid_neg_ids.append(neg_id)
        neg_pairs.append((query, neg_text))
    stats["valid_candidates"] = len(valid_neg_ids)

    pairs = [(query, pos_text)] + neg_pairs
    scores = score_pairs(cross_encoder, pairs, batch_size=batch_size)
    if not scores:
        raise RuntimeError(f"CrossEncoder returned no scores for query {query_id}")
    score_pos = float(scores[0])
    neg_scores = [float(s) for s in scores[1:]]

    negatives_hard: List[Dict[str, Any]] = []
    for neg_id, score_neg in zip(valid_neg_ids, neg_scores):
        if score_neg >= score_pos:
            stats["filtered_pseudo"] += 1
            continue
        if score_neg < min_neg_score:
            stats["filtered_easy"] += 1
            continue
        if (score_pos - score_neg) < score_margin:
            stats["filtered_margin"] += 1
            continue
        negatives_hard.append({"chunk_id": neg_id, "score_neg": score_neg})

    negatives_hard.sort(key=lambda item: item["score_neg"], reverse=True)
    if max_negatives > 0:
        negatives_hard = negatives_hard[:max_negatives]

    stats["hard_negatives"] = len(negatives_hard)

    hard_record = {
        "query_id": query_id,
        "query": query,
        "positive_chunk_id": pos_chunk_id,
        "score_pos": score_pos,
        "negatives_hard": negatives_hard,
    }
    train_record = {
        "query_id": query_id,
        "query": query,
        "positive_chunk_id": pos_chunk_id,
        "negative_chunk_ids": [neg["chunk_id"] for neg in negatives_hard],
    }
    return hard_record, train_record, stats


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    """Write records to JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build training data with hard negatives.")
    parser.add_argument(
        "--hard_candidates_path",
        type=str,
        default="scheme_data/baseline/hard_candidates.jsonl",
        help="Path to mined hard candidate JSONL file.",
    )
    parser.add_argument(
        "--chunks_path",
        type=str,
        default="data_processed/chunks.jsonl",
        help="Path to chunks JSONL file.",
    )
    parser.add_argument(
        "--output_hard_path",
        type=str,
        default="scheme_data/baseline/hard_negatives.jsonl",
        help="Output path for hard negatives JSONL.",
    )
    parser.add_argument(
        "--output_train_path",
        type=str,
        default="scheme_data/baseline/train_with_hard_neg.jsonl",
        help="Output path for bi-encoder training JSONL.",
    )
    parser.add_argument(
        "--cross_encoder_model_name",
        type=str,
        default="BAAI/bge-reranker-base",
        help="Cross-encoder model name.",
    )
    parser.add_argument(
        "--max_negatives_per_query",
        type=int,
        default=5,
        help="Maximum number of hard negatives to keep per query.",
    )
    parser.add_argument(
        "--min_neg_score",
        type=float,
        default=0.4,
        help="Minimum score for a candidate to be kept as hard negative.",
    )
    parser.add_argument(
        "--score_margin",
        type=float,
        default=0.05,
        help="Required margin between positive and negative scores.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for cross-encoder scoring.",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU for the cross-encoder if available.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/build_train_data.log",
        help="Optional path to write log output.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging(args.log_file)
    args = parse_args()

    chunk_text_map = load_chunks(args.chunks_path)
    LOGGER.info("Loading hard negative candidates from %s", args.hard_candidates_path)
    samples = load_hard_candidates(args.hard_candidates_path)
    LOGGER.info("Loaded %d chunks and %d candidate samples.", len(chunk_text_map), len(samples))

    cross_encoder = build_cross_encoder(args.cross_encoder_model_name, args.use_gpu)

    hard_records: List[Dict[str, Any]] = []
    train_records: List[Dict[str, Any]] = []
    aggregate_stats = {
        "processed": 0,
        "skipped": 0,
        "total_candidates": 0,
        "valid_candidates": 0,
        "missing_chunks": 0,
        "filtered_pseudo": 0,
        "filtered_easy": 0,
        "filtered_margin": 0,
        "total_hard_negatives": 0,
    }

    for sample in tqdm(samples, desc="Scoring hard negatives"):
        result = process_query_sample(
            sample=sample,
            chunk_text_map=chunk_text_map,
            cross_encoder=cross_encoder,
            batch_size=args.batch_size,
            max_negatives=args.max_negatives_per_query,
            min_neg_score=args.min_neg_score,
            score_margin=args.score_margin,
        )
        if result is None:
            aggregate_stats["skipped"] += 1
            continue
        hard_record, train_record, stats = result
        hard_records.append(hard_record)
        train_records.append(train_record)

        aggregate_stats["processed"] += 1
        aggregate_stats["total_candidates"] += stats["total_candidates"]
        aggregate_stats["valid_candidates"] += stats["valid_candidates"]
        aggregate_stats["missing_chunks"] += stats["missing_chunks"]
        aggregate_stats["filtered_pseudo"] += stats["filtered_pseudo"]
        aggregate_stats["filtered_easy"] += stats["filtered_easy"]
        aggregate_stats["filtered_margin"] += stats["filtered_margin"]
        aggregate_stats["total_hard_negatives"] += stats["hard_negatives"]

    write_jsonl(args.output_hard_path, hard_records)
    write_jsonl(args.output_train_path, train_records)

    avg_candidates = (
        aggregate_stats["total_candidates"] / aggregate_stats["processed"]
        if aggregate_stats["processed"]
        else 0.0
    )
    avg_hard = (
        aggregate_stats["total_hard_negatives"] / aggregate_stats["processed"]
        if aggregate_stats["processed"]
        else 0.0
    )
    LOGGER.info(
        "Processed %d samples (skipped %d). Avg candidates/query: %.2f, avg hard negatives/query: %.2f",
        aggregate_stats["processed"],
        aggregate_stats["skipped"],
        avg_candidates,
        avg_hard,
    )
    LOGGER.info(
        "Filtered pseudo positives: %d, filtered easy negatives: %d, filtered by margin: %d, missing chunks: %d",
        aggregate_stats["filtered_pseudo"],
        aggregate_stats["filtered_easy"],
        aggregate_stats["filtered_margin"],
        aggregate_stats["missing_chunks"],
    )
    LOGGER.info(
        "Hard negatives saved to %s, training data saved to %s",
        args.output_hard_path,
        args.output_train_path,
    )


if __name__ == "__main__":
    main()
