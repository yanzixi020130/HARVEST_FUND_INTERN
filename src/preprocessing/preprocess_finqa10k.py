"""Preprocess Financial-QA-10k data for downstream RAG usage."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


REQUIRED_COLUMNS = ["question", "answer", "context", "ticker", "filing"]
TEXT_COLUMNS = ["question", "answer", "context"]
MAX_CHARS = 500
MIN_CONTEXT_CHARS = 50
MIN_ANSWER_CHARS = 5
MIN_QUESTION_NON_WS_CHARS = 10
TARGET_MIN_CHUNK = 300


def project_root() -> Path:
    """Return repository root (two levels up from src/preprocessing)."""
    return Path(__file__).resolve().parent.parent.parent


def setup_logger(log_path: Path) -> logging.Logger:
    """Configure file and console loggers."""
    logger = logging.getLogger("preprocess_finqa10k")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def normalize_text(value: Any) -> str:
    """Apply whitespace normalization rules."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def length_without_whitespace(text: str) -> int:
    """Return length after removing whitespace."""
    return len(re.sub(r"\s+", "", text))


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load CSV and ensure required columns exist."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input data not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    for col in TEXT_COLUMNS:
        df[col] = df[col].apply(normalize_text)

    df["ticker"] = df["ticker"].apply(lambda x: "" if pd.isna(x) else str(x).strip())
    df["filing"] = df["filing"].apply(lambda x: "" if pd.isna(x) else str(x).strip())
    return df


def filter_records(df: pd.DataFrame) -> Dict[str, int]:
    """Filter invalid rows based on length rules, returning masks."""
    masks = {}

    masks["question"] = df["question"].apply(
        lambda text: length_without_whitespace(text) >= MIN_QUESTION_NON_WS_CHARS
    )
    masks["answer"] = df["answer"].apply(lambda text: len(text.strip()) >= MIN_ANSWER_CHARS)
    masks["context"] = df["context"].apply(
        lambda text: len(text.strip()) >= MIN_CONTEXT_CHARS
    )
    return masks


def split_sentences(text: str) -> List[str]:
    """Split text on sentence boundaries (., ?, !, newline)."""
    sentences: List[str] = []
    buffer: List[str] = []

    for char in text:
        buffer.append(char)
        if char in {".", "?", "!", "\n"}:
            sentence = "".join(buffer).strip()
            if sentence:
                sentences.append(sentence)
            buffer = []

    if buffer:
        sentence = "".join(buffer).strip()
        if sentence:
            sentences.append(sentence)

    return sentences or [text.strip()]


def chunk_context(text: str) -> List[str]:
    """Chunk context into ~300-500 character pieces."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= MAX_CHARS:
        return [text]

    sentences = split_sentences(text)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > MAX_CHARS:
            if current:
                chunks.append(current.strip())
                current = ""
            for start in range(0, len(sentence), MAX_CHARS):
                piece = sentence[start : start + MAX_CHARS].strip()
                if piece:
                    chunks.append(piece)
            continue

        if not current:
            current = sentence
            continue

        candidate_len = len(current) + 1 + len(sentence)
        if candidate_len <= MAX_CHARS:
            current = f"{current} {sentence}"
        elif len(current) >= TARGET_MIN_CHUNK:
            chunks.append(current.strip())
            current = sentence
        else:
            chunks.append(current.strip())
            current = sentence

    if current.strip():
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


def sanitize_metadata(value: str) -> Optional[str]:
    """Return sanitized metadata value or None."""
    value = value.strip()
    return value if value else None


def make_doc_id(ticker: Optional[str], filing: Optional[str], index: Any) -> str:
    """Create doc_id from ticker/filing or fallback."""
    if ticker and filing:
        ticker_clean = re.sub(r"\s+", "", ticker)
        filing_clean = re.sub(r"\s+", "", filing)
        return f"{ticker_clean}_{filing_clean}"
    return f"doc_{index}"


def main() -> None:
    root = project_root()
    data_path = root / "data_raw" / "Financial-QA-10k.csv"
    processed_dir = root / "data_processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "chunks.jsonl"

    log_dir = root / "reports" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir / "preprocess_finqa10k.log")

    try:
        df = load_dataset(data_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load dataset: %s", exc)
        raise SystemExit(1)

    original_count = len(df)
    masks = filter_records(df)

    reason_counts = {
        "question_too_short_or_empty": (~masks["question"]).sum(),
        "answer_too_short_or_empty": (~masks["answer"]).sum(),
        "context_too_short_or_empty": (~masks["context"]).sum(),
    }

    combined_mask = masks["question"] & masks["answer"] & masks["context"]
    filtered_df = df[combined_mask].copy()
    after_rule_count = len(filtered_df)

    filtered_df = filtered_df.drop_duplicates(subset=TEXT_COLUMNS)
    duplicates_removed = after_rule_count - len(filtered_df)
    reason_counts["duplicate_qac"] = duplicates_removed

    cleaned_count = len(filtered_df)
    dropped_total = original_count - cleaned_count

    if filtered_df.empty:
        logger.warning("No records remain after cleaning; writing empty output.")

    total_chunks = 0
    chunk_length_sum = 0

    with output_path.open("w", encoding="utf-8") as f:
        for row_index, row in filtered_df.iterrows():
            ticker_value = sanitize_metadata(row["ticker"])
            filing_value = sanitize_metadata(row["filing"])
            doc_id = make_doc_id(ticker_value, filing_value, row_index)
            context_chunks = chunk_context(row["context"])
            if not context_chunks:
                continue

            for chunk_idx, chunk_text in enumerate(context_chunks):
                record = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                    "text": chunk_text,
                    "metadata": {
                        "ticker": ticker_value,
                        "filing": filing_value,
                        "source": "Financial-QA-10k",
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1
                chunk_length_sum += len(chunk_text)

    avg_chunks_per_doc = total_chunks / cleaned_count if cleaned_count else 0.0
    avg_chunk_length = (
        chunk_length_sum / total_chunks if total_chunks else 0.0
    )

    logger.info("Original records: %d", original_count)
    logger.info("Cleaned records: %d", cleaned_count)
    logger.info("Discarded records: %d", dropped_total)
    logger.info("Discard reasons: %s", reason_counts)
    logger.info("Generated chunks: %d", total_chunks)
    logger.info("Average chunks per doc: %.2f", avg_chunks_per_doc)
    logger.info("Average chunk length: %.2f characters", avg_chunk_length)
    logger.info("Output written to: %s", output_path)


if __name__ == "__main__":
    main()
