"""
Split positive pairs by query_id into train/dev/test subsets and build eval file.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

LOGGER = logging.getLogger(__name__)


def load_pairs(path: str) -> List[Dict[str, Any]]:
    """Load positive pairs from JSONL."""
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
                    raise ValueError(f"Missing '{key}' in record at {path}:{line_no}")
            pairs.append(
                {
                    "query_id": str(record["query_id"]),
                    "query": str(record["query"]),
                    "positive_chunk_id": str(record["positive_chunk_id"]),
                }
            )
    if not pairs:
        raise ValueError(f"No pairs loaded from {path}")
    return pairs


def group_by_query(pairs: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group records by query_id."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in pairs:
        grouped[record["query_id"]].append(record)
    return grouped


def validate_ratios(train: float, dev: float, test: float) -> None:
    """Ensure ratios sum to 1 within tolerance."""
    total = train + dev + test
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0 (±0.01). Got {total:.4f}")


def split_query_ids(
    query_ids: Sequence[str],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[set[str], set[str], set[str]]:
    """Shuffle and split query ids according to ratios."""
    ids = list(query_ids)
    random.Random(seed).shuffle(ids)

    n = len(ids)
    train_ratio, dev_ratio, _ = ratios
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    train_ids = set(ids[:n_train])
    dev_ids = set(ids[n_train : n_train + n_dev])
    test_ids = set(ids[n_train + n_dev :])
    return train_ids, dev_ids, test_ids


def write_jsonl(path: str, records: Sequence[Dict[str, Any]]) -> None:
    """Write records to JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_eval_file(test_groups: Dict[str, List[Dict[str, Any]]], output_path: str) -> None:
    """Create eval file aggregating positive_chunk_ids for each query."""
    eval_records: List[Dict[str, Any]] = []
    for query_id, samples in test_groups.items():
        if not samples:
            continue
        query_text = samples[0]["query"]
        chunk_ids = [sample["positive_chunk_id"] for sample in samples]
        eval_records.append(
            {
                "query_id": query_id,
                "query": query_text,
                "positive_chunk_ids": chunk_ids,
            }
        )
    write_jsonl(output_path, eval_records)


def assign_records_to_split(
    grouped_pairs: Dict[str, List[Dict[str, Any]]],
    split_ids: Tuple[set[str], set[str], set[str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition grouped pairs into train/dev/test lists."""
    train_ids, dev_ids, test_ids = split_ids
    train_records: List[Dict[str, Any]] = []
    dev_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []
    for query_id, samples in grouped_pairs.items():
        if query_id in train_ids:
            train_records.extend(samples)
        elif query_id in dev_ids:
            dev_records.extend(samples)
        else:
            test_records.extend(samples)
    return train_records, dev_records, test_records


def log_split_stats(
    split_name: str,
    ids: set[str],
    records: Sequence[Dict[str, Any]],
    total_queries: int,
) -> None:
    """Log statistics for a split."""
    ratio = (len(ids) / total_queries) if total_queries else 0.0
    LOGGER.info(
        "%s split: %d queries (%.2f%%), %d samples",
        split_name,
        len(ids),
        ratio * 100,
        len(records),
    )
    if not records:
        LOGGER.warning("%s split has zero samples.", split_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split positive pairs into train/dev/test.")
    parser.add_argument(
        "--pairs_path",
        type=str,
        default="pairs/positive_pairs.jsonl",
        help="Input positive pairs JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pairs",
        help="Directory to store split JSONL files.",
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        default="data_eval/eval_pairs.jsonl",
        help="Output path for evaluation file built from test split.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio (by query count).",
    )
    parser.add_argument(
        "--dev_ratio",
        type=float,
        default=0.1,
        help="Dev split ratio (by query count).",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test split ratio (by query count).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )
    args = parse_args()
    validate_ratios(args.train_ratio, args.dev_ratio, args.test_ratio)

    pairs = load_pairs(args.pairs_path)
    grouped = group_by_query(pairs)
    query_ids = list(grouped.keys())
    LOGGER.info(
        "Loaded %d samples across %d unique queries.",
        len(pairs),
        len(query_ids),
    )

    split_ids = split_query_ids(
        query_ids=query_ids,
        ratios=(args.train_ratio, args.dev_ratio, args.test_ratio),
        seed=args.seed,
    )
    train_ids, dev_ids, test_ids = split_ids
    train_records, dev_records, test_records = assign_records_to_split(grouped, split_ids)

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train_positive_pairs.jsonl")
    dev_path = os.path.join(args.output_dir, "dev_positive_pairs.jsonl")
    test_path = os.path.join(args.output_dir, "test_positive_pairs.jsonl")

    write_jsonl(train_path, train_records)
    write_jsonl(dev_path, dev_records)
    write_jsonl(test_path, test_records)

    log_split_stats("Train", train_ids, train_records, len(query_ids))
    log_split_stats("Dev", dev_ids, dev_records, len(query_ids))
    log_split_stats("Test", test_ids, test_records, len(query_ids))

    test_groups = {qid: grouped[qid] for qid in test_ids}
    build_eval_file(test_groups, args.eval_output)
    LOGGER.info(
        "Wrote splits to %s and eval file to %s",
        args.output_dir,
        args.eval_output,
    )


if __name__ == "__main__":
    main()
