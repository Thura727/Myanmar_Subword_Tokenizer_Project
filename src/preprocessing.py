"""
Data preprocessing pipeline for Myanmar text corpora.

Steps:
  1. Remove HTML tags
  2. Remove unnecessary symbols
  3. Normalize Myanmar Unicode (NFC)
  4. Normalize whitespace
  5. Remove empty lines
  6. Deduplicate sentences
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import regex
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "myanmar_corpus.txt"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_corpus.txt"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_corpus.txt"
EVAL_PATH = PROJECT_ROOT / "data" / "processed" / "eval_corpus.txt"

# Myanmar Unicode block: U+1000–U+109F (plus extensions used in practice)
MYANMAR_CHAR = regex.compile(r"[\u1000-\u109F\uAA60-\uAA7F\uA9E0-\uA9FF]")
HTML_TAG = re.compile(r"<[^>]+>")
# Keep Myanmar letters, digits, basic Latin (for loanwords/URLs remnants),
# Myanmar punctuation, and common sentence punctuation.
ALLOWED = regex.compile(
    r"[^\u1000-\u109F\uAA60-\uAA7F\uA9E0-\uA9FF"
    r"0-9A-Za-z"
    r"\u104A-\u104F"  # Myanmar signs / punctuation
    r"\s.,!?;:'\"()\-\u2013\u2014\u104B]"
)


def remove_html(text: str) -> str:
    """Strip HTML/XML tags from text."""
    return HTML_TAG.sub(" ", text)


def normalize_unicode(text: str) -> str:
    """
    Normalize to Unicode NFC.

    Myanmar text often mixes NFC/NFD forms (e.g. stacked consonants and
    dependent vowels). NFC yields a canonical composed representation that
    SentencePiece can treat consistently.
    """
    return unicodedata.normalize("NFC", text)


def remove_unnecessary_symbols(text: str) -> str:
    """Remove characters outside the allowed Myanmar / punctuation set."""
    return ALLOWED.sub(" ", text)


def normalize_spaces(text: str) -> str:
    """Collapse runs of whitespace and strip leading/trailing spaces."""
    text = regex.sub(r"[\t\r\f\v]+", " ", text)
    text = regex.sub(r" +", " ", text)
    return text.strip()


def clean_line(line: str) -> str:
    """Apply the full cleaning pipeline to a single line."""
    line = remove_html(line)
    line = normalize_unicode(line)
    line = remove_unnecessary_symbols(line)
    line = normalize_spaces(line)
    return line


def is_valid_myanmar_sentence(line: str, min_myanmar_chars: int = 4) -> bool:
    """Keep lines that contain a minimum amount of Myanmar script."""
    if not line:
        return False
    myanmar_count = len(MYANMAR_CHAR.findall(line))
    return myanmar_count >= min_myanmar_chars


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def preprocess_corpus(
    input_path: Path | str = RAW_PATH,
    output_path: Path | str = PROCESSED_PATH,
    min_myanmar_chars: int = 4,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Preprocess a raw Myanmar corpus and write cleaned / train / eval files.

    Returns a statistics dictionary for logging / notebooks.
    """
    import random

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Raw corpus not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    seen: set[str] = set()
    cleaned: list[str] = []
    stats = {
        "raw_lines": len(raw_lines),
        "empty_removed": 0,
        "duplicates_removed": 0,
        "non_myanmar_removed": 0,
        "kept": 0,
        "train_lines": 0,
        "eval_lines": 0,
    }

    for line in tqdm(raw_lines, desc="Preprocessing"):
        cleaned_line = clean_line(line)
        if not cleaned_line:
            stats["empty_removed"] += 1
            continue
        if not is_valid_myanmar_sentence(cleaned_line, min_myanmar_chars):
            stats["non_myanmar_removed"] += 1
            continue
        if cleaned_line in seen:
            stats["duplicates_removed"] += 1
            continue
        seen.add(cleaned_line)
        cleaned.append(cleaned_line)

    stats["kept"] = len(cleaned)
    _write_lines(output_path, cleaned)

    # Deterministic train / eval split for held-out metrics
    rng = random.Random(seed)
    shuffled = list(cleaned)
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_ratio)) if shuffled else 0
    eval_lines = shuffled[:n_eval]
    train_lines = shuffled[n_eval:]
    stats["train_lines"] = len(train_lines)
    stats["eval_lines"] = len(eval_lines)
    _write_lines(TRAIN_PATH, train_lines)
    _write_lines(EVAL_PATH, eval_lines)

    print(f"Preprocessing complete → {output_path}")
    print(f"  train → {TRAIN_PATH} ({stats['train_lines']} lines)")
    print(f"  eval  → {EVAL_PATH} ({stats['eval_lines']} lines)")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess Myanmar corpus for SentencePiece training."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(RAW_PATH),
        help="Path to raw corpus text file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROCESSED_PATH),
        help="Path for cleaned corpus",
    )
    args = parser.parse_args()
    preprocess_corpus(args.input, args.output)


if __name__ == "__main__":
    main()
