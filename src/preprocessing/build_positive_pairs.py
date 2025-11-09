"""Build positive query-document pairs for Financial-QA-10k RAG training."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

REQUIRED_COLUMNS = ["question", "answer", "context", "ticker", "filing"]
MIN_QUERY_LEN = 5
MIN_ANSWER_LEN = 1
MIN_CONTEXT_LEN = 1
MIN_KEYWORD_LEN = 3


def project_root() -> Path:
    """Return repository root (one level up from src)."""
    return Path(__file__).resolve().parent.parent.parent


def setup_logger(log_path: Path) -> logging.Logger:
    """Create logger that writes to file and console."""
    logger = logging.getLogger("build_positive_pairs")
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
    """Normalize text by stripping, unifying whitespace, removing tabs."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def sanitize_metadata(value: Any) -> Optional[str]:
    """Convert metadata to trimmed string or None."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def make_doc_id(ticker: Optional[str], filing: Optional[str], index: Any) -> str:
    """Generate doc_id consistent with preprocessing script."""
    if ticker and filing:
        ticker_clean = re.sub(r"\s+", "", ticker)
        filing_clean = re.sub(r"\s+", "", filing)
        return f"{ticker_clean}_{filing_clean}"
    return f"doc_{index}"


def load_chunks(chunks_path: Path) -> Dict[str, List[Dict[str, str]]]:
    """Load chunks JSONL into doc_id -> list of chunk dicts."""
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    chunks_map: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with chunks_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {chunks_path}: {exc}"
                ) from exc
            doc_id = record.get("doc_id")
            chunk_id = record.get("chunk_id")
            text = record.get("text", "")
            if not doc_id or not chunk_id:
                continue
            chunks_map[doc_id].append({"chunk_id": chunk_id, "text": text})
    return chunks_map


def tokenize_answer(answer: str) -> List[str]:
    """Split answer into keywords, dropping very short tokens."""
    tokens = re.split(r"\W+", answer.lower())
    return [tok for tok in tokens if len(tok) >= MIN_KEYWORD_LEN]


def select_chunk(
    answer: str, chunks: List[Dict[str, str]]
) -> Tuple[Optional[Dict[str, str]], str]:
    """
    Select the best chunk using exact match, keyword overlap, or fallback.

    Returns (chunk_dict, match_type) where match_type in {"exact", "keyword", "fallback"}.
    """
    if not chunks:
        return None, "missing"

    answer_clean = answer.strip()
    if answer_clean:
        answer_lower = answer_clean.lower()
        for chunk in chunks:
            if answer_lower in chunk["text"].lower():
                return chunk, "exact"

    keywords = tokenize_answer(answer_clean)
    if keywords:
        best_chunk = None
        best_score = 0
        for chunk in chunks:
            text_lower = chunk["text"].lower()
            score = sum(1 for token in keywords if token in text_lower)
            if score > best_score:
                best_score = score
                best_chunk = chunk
        if best_chunk is not None and best_score > 0:
            return best_chunk, "keyword"

    return chunks[0], "fallback"


def main() -> None:
    root = project_root()
    csv_path = root / "data_raw" / "Financial-QA-10k.csv"
    chunks_path = root / "data_processed" / "chunks.jsonl"
    pairs_dir = root / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    output_path = pairs_dir / "positive_pairs.jsonl"

    log_dir = root / "reports" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir / "build_positive_pairs.log")

    try:
        chunks_map = load_chunks(chunks_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("加载 chunks 失败: %s", exc)
        raise SystemExit(1)

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except FileNotFoundError:
        logger.error("未找到 CSV 文件: %s", csv_path)
        raise SystemExit(1)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        logger.error("CSV 缺失必要列: %s", missing_cols)
        raise SystemExit(1)

    total_records = len(df)
    success_count = 0
    invalid_input_count = 0
    missing_chunk_count = 0
    exact_match_count = 0
    keyword_match_count = 0
    fallback_count = 0

    with output_path.open("w", encoding="utf-8") as out_file:
        for row_index, row in df.iterrows():
            question = normalize_text(row["question"])
            answer = normalize_text(row["answer"])
            context = normalize_text(row["context"])

            if (
                len(question) < MIN_QUERY_LEN
                or len(answer) < MIN_ANSWER_LEN
                or len(context) < MIN_CONTEXT_LEN
            ):
                invalid_input_count += 1
                continue

            ticker_value = sanitize_metadata(row["ticker"])
            filing_value = sanitize_metadata(row["filing"])
            doc_id = make_doc_id(ticker_value, filing_value, row_index)
            chunk_list = chunks_map.get(doc_id)

            if not chunk_list:
                missing_chunk_count += 1
                continue

            chunk, match_type = select_chunk(answer, chunk_list)
            if chunk is None:
                missing_chunk_count += 1
                continue

            if match_type == "exact":
                exact_match_count += 1
            elif match_type == "keyword":
                keyword_match_count += 1
            else:
                fallback_count += 1

            record = {
                "query_id": f"FQA_{row_index}",
                "query": question,
                "positive_doc_id": doc_id,
                "positive_chunk_id": chunk["chunk_id"],
                "answer": answer,
                "metadata": {
                    "ticker": ticker_value,
                    "filing": filing_value,
                    "source": "Financial-QA-10k",
                },
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            success_count += 1

    logger.info("总记录数: %d", total_records)
    logger.info("成功构建正样本数: %d", success_count)
    logger.info("无有效 query/answer/context 跳过: %d", invalid_input_count)
    logger.info("找不到 doc_id 或无 chunk 跳过: %d", missing_chunk_count)
    logger.info("完整 answer 精确命中数: %d", exact_match_count)
    logger.info("宽松匹配命中数: %d", keyword_match_count)
    logger.info("回退首 chunk 数: %d", fallback_count)
    logger.info("输出文件: %s", output_path)


if __name__ == "__main__":
    main()
