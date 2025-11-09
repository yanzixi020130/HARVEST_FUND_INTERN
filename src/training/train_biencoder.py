"""
Train a baseline bi-encoder on English hard-negative mining data.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


def init_logger(log_path: str) -> None:
    """Configure logging to both console and file."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline bi-encoder with hard negatives.")
    parser.add_argument(
        "--train_path",
        type=str,
        default="scheme_data/baseline/train_with_hard_neg.jsonl",
        help="Training data JSONL path.",
    )
    parser.add_argument(
        "--dev_path",
        type=str,
        default="pairs/dev_positive_pairs.jsonl",
        help="Optional dev positive pairs JSONL for monitoring.",
    )
    parser.add_argument(
        "--chunks_path",
        type=str,
        default="data_processed/chunks.jsonl",
        help="Chunk text JSONL path for resolving chunk ids.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="Pretrained sentence-transformer model name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/baseline/bi_encoder",
        help="Directory to save trained model.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="reports/logs/train_biencoder.log",
        help="Path to training log file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=256,
        help="Maximum sequence length for the model tokenizer.",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use CUDA if available.",
    )
    return parser.parse_args()


def load_chunks(chunks_path: str) -> Dict[str, str]:
    """Load chunk_id to text mapping."""
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
    logging.info("Loaded %d chunks from %s", len(chunk_text_map), chunks_path)
    return chunk_text_map


@dataclass
class TrainDataStats:
    total_samples: int = 0
    missing_positive: int = 0
    missing_negatives: int = 0
    expanded_examples: int = 0


def load_train_examples(
    train_path: str,
    chunk_text_map: Dict[str, str],
) -> Tuple[List[InputExample], TrainDataStats]:
    """Build SentenceTransformer InputExamples from the train JSONL."""
    examples: List[InputExample] = []
    stats = TrainDataStats()
    with open(train_path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {train_path}:{line_no}") from exc
            stats.total_samples += 1
            query = str(data["query"])
            pos_id = str(data["positive_chunk_id"])
            pos_text = chunk_text_map.get(pos_id)
            if pos_text is None:
                stats.missing_positive += 1
                logging.warning("Positive chunk %s missing for query %s", pos_id, data.get("query_id"))
                continue
            neg_ids = [str(cid) for cid in data.get("negative_chunk_ids", [])]
            valid_neg_count = 0
            for neg_id in neg_ids:
                if neg_id not in chunk_text_map:
                    stats.missing_negatives += 1
                    logging.warning("Negative chunk %s not found for query %s", neg_id, data.get("query_id"))
                else:
                    valid_neg_count += 1

            repeat = max(1, valid_neg_count)
            for _ in range(repeat):
                examples.append(InputExample(texts=[query, pos_text]))
            stats.expanded_examples += repeat
    if not examples:
        raise ValueError("No valid training examples were created.")
    logging.info(
        "Constructed %d InputExamples from %d samples (missing positives: %d, missing negatives: %d)",
        len(examples),
        stats.total_samples,
        stats.missing_positive,
        stats.missing_negatives,
    )
    return examples, stats


def prepare_model(model_name: str, max_len: int, use_gpu: bool) -> SentenceTransformer:
    """Load SentenceTransformer and optionally push to GPU."""
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_len
    if use_gpu and torch.cuda.is_available():
        model = model.to("cuda")
        logging.info("Using GPU for training.")
    else:
        logging.info("Training on CPU.")
    logging.info("Loaded model: %s", model_name)
    return model


def train(args: argparse.Namespace) -> None:
    """Main training workflow."""
    chunk_text_map = load_chunks(args.chunks_path)
    train_examples, stats = load_train_examples(args.train_path, chunk_text_map)

    os.makedirs(args.output_dir, exist_ok=True)
    model = prepare_model(args.model_name, args.max_len, args.use_gpu)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = max(1, int(0.1 * len(train_dataloader)))
    logging.info(
        "Starting training with %d batches, epochs=%d, batch_size=%d, warmup_steps=%d, lr=%.2e",
        len(train_dataloader),
        args.epochs,
        args.batch_size,
        warmup_steps,
        args.lr,
    )

    start_time = time.perf_counter()
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=args.epochs,
            warmup_steps=warmup_steps,
            output_path=args.output_dir,
            optimizer_params={"lr": args.lr},
            show_progress_bar=True,
        )
    except Exception as exc:
        logging.exception("Training failed due to an unhandled exception.")
        raise exc
    end_time = time.perf_counter()

    duration_minutes = (end_time - start_time) / 60.0
    logging.info("Training completed in %.2f minutes.", duration_minutes)
    logging.info("Model artifacts saved to %s", args.output_dir)
    logging.info(
        "Final stats — total samples: %d, expanded InputExamples: %d",
        stats.total_samples,
        stats.expanded_examples,
    )


def main() -> None:
    args = parse_args()
    init_logger(args.log_path)
    logging.info("Training configuration: %s", vars(args))
    logging.info("Run started at %s", datetime.utcnow().isoformat())
    try:
        train(args)
    except Exception:
        logging.exception("Bi-encoder training terminated with errors.")
        raise
    logging.info("Training finished successfully.")


if __name__ == "__main__":
    main()
