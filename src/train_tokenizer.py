"""
Train SentencePiece BPE and Unigram tokenizers on cleaned Myanmar text.

Models trained:
  - BPE:      vocab sizes 4000, 8000, 16000
  - Unigram:  vocab sizes 4000, 8000, 16000

Character coverage is fixed at 0.9995 (recommended for non-Latin scripts).
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import sentencepiece as spm
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CORPUS = PROJECT_ROOT / "data" / "processed" / "train_corpus.txt"
CLEANED_CORPUS = PROJECT_ROOT / "data" / "processed" / "cleaned_corpus.txt"
CORPUS_PATH = TRAIN_CORPUS  # default; falls back to cleaned if missing
MODELS_DIR = PROJECT_ROOT / "models"

VOCAB_SIZES = (4000, 8000, 16000)
MODEL_TYPES = ("bpe", "unigram")
CHARACTER_COVERAGE = 0.9995


def train_sentencepiece(
    corpus_path: Path | str,
    model_prefix: Path | str,
    model_type: str,
    vocab_size: int,
    character_coverage: float = CHARACTER_COVERAGE,
) -> Path:
    """
    Train a single SentencePiece model.

    Parameters
    ----------
    corpus_path:
        Path to one-sentence-per-line training text.
    model_prefix:
        Output prefix (writes ``{prefix}.model`` and ``{prefix}.vocab``).
    model_type:
        ``"bpe"`` or ``"unigram"``.
    vocab_size:
        Target vocabulary size.
    character_coverage:
        Fraction of characters covered by the model (Myanmar-friendly default).

    Returns
    -------
    Path to the trained ``.model`` file.
    """
    corpus_path = Path(corpus_path)
    model_prefix = Path(model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model_type={model_type!r}")

    train_args = {
        "input": str(corpus_path),
        "model_prefix": str(model_prefix),
        "model_type": model_type,
        "vocab_size": vocab_size,
        "character_coverage": character_coverage,
        # Keep whitespace handling simple for Myanmar (spaces are rare).
        "normalization_rule_name": "identity",
        "remove_extra_whitespaces": True,
        # Hard / soft limit helps when corpus is modest relative to vocab.
        "hard_vocab_limit": False,
        # Reproducibility
        "seed_sentencepiece_size": 1_000_000,
        "shuffle_input_sentence": True,
        "input_sentence_size": 1_000_000,
        # Myanmar has no word boundaries; treat the line as the unit.
        "split_by_whitespace": True,
        "bos_id": -1,
        "eos_id": -1,
        "unk_id": 0,
        "pad_id": -1,
    }

    if (
        os.name == "nt"
        and (not str(corpus_path).isascii() or not str(model_prefix).isascii())
    ):
        with tempfile.TemporaryDirectory(prefix="sentencepiece_") as temporary_dir:
            temporary_dir = Path(temporary_dir)
            staged_corpus = temporary_dir / "corpus.txt"
            staged_prefix = temporary_dir / "model"
            shutil.copyfile(corpus_path, staged_corpus)
            train_args["input"] = str(staged_corpus)
            train_args["model_prefix"] = str(staged_prefix)
            spm.SentencePieceTrainer.train(**train_args)
            for extension in (".model", ".vocab"):
                shutil.copyfile(
                    Path(f"{staged_prefix}{extension}"),
                    Path(f"{model_prefix}{extension}"),
                )
    else:
        spm.SentencePieceTrainer.train(**train_args)

    model_path = Path(str(model_prefix) + ".model")
    return model_path


def train_all_models(
    corpus_path: Path | str = CORPUS_PATH,
    models_dir: Path | str = MODELS_DIR,
    vocab_sizes: tuple[int, ...] = VOCAB_SIZES,
    character_coverage: float = CHARACTER_COVERAGE,
) -> list[Path]:
    """Train the full BPE / Unigram grid and return model paths."""
    corpus_path = Path(corpus_path)
    if not corpus_path.exists() and CLEANED_CORPUS.exists():
        print(f"Train split missing; falling back to {CLEANED_CORPUS}")
        corpus_path = CLEANED_CORPUS
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Training corpus not found: {corpus_path}. "
            "Run preprocessing.py first."
        )

    trained: list[Path] = []
    jobs = [
        (model_type, vocab_size)
        for model_type in MODEL_TYPES
        for vocab_size in vocab_sizes
    ]

    for model_type, vocab_size in tqdm(jobs, desc="Training models"):
        prefix = models_dir / f"{model_type}_{vocab_size}"
        print(f"\n→ Training {model_type.upper()} vocab={vocab_size} …")
        path = train_sentencepiece(
            corpus_path=corpus_path,
            model_prefix=prefix,
            model_type=model_type,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
        )
        trained.append(path)
        print(f"  saved {path}")

    return trained


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SentencePiece BPE and Unigram Myanmar tokenizers."
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=str(CORPUS_PATH),
        help="Path to cleaned training corpus",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(MODELS_DIR),
        help="Directory for trained .model files",
    )
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=list(VOCAB_SIZES),
        help="Vocabulary sizes to train",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=CHARACTER_COVERAGE,
        help="SentencePiece character_coverage",
    )
    args = parser.parse_args()
    train_all_models(
        corpus_path=args.corpus,
        models_dir=args.models_dir,
        vocab_sizes=tuple(args.vocab_sizes),
        character_coverage=args.character_coverage,
    )
    print("\nAll models trained successfully.")


if __name__ == "__main__":
    main()
