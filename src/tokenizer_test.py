"""
Tokenization testing: encode sample Myanmar sentences with BPE and Unigram.

Writes human-readable comparisons to ``results/sample_outputs.txt``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SAMPLE_OUTPUT = RESULTS_DIR / "sample_outputs.txt"

# Canonical test sentences from the project brief
DEFAULT_SENTENCES = [
    "မြန်မာနိုင်ငံသည် ဖွံ့ဖြိုးတိုးတက်လာနေသည်။",
    "ကျန်းမာရေးဝန်ကြီးဌာနသည် ပြည်သူများအတွက် ဝန်ဆောင်မှုပေးနေသည်။",
]

# Additional morphology-focused compounds for error analysis
MORPHOLOGY_EXAMPLES = [
    "ကျန်းမာရေးဝန်ကြီးဌာန",
    "ဖွံ့ဖြိုးတိုးတက်မှု",
    "ပညာရေးစနစ်",
]

VOCAB_SIZES = (4000, 8000, 16000)


def load_processor(model_path: Path) -> spm.SentencePieceProcessor:
    """Load a trained SentencePiece model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    sp = spm.SentencePieceProcessor()
    if not sp.LoadFromSerializedProto(model_path.read_bytes()):
        raise ValueError(f"Failed to load model: {model_path}")
    return sp


def encode_pieces(sp: spm.SentencePieceProcessor, text: str) -> list[str]:
    """Return subword pieces for ``text``."""
    return list(sp.encode(text, out_type=str))


def format_comparison(
    sentence: str,
    bpe_tokens: list[str],
    unigram_tokens: list[str],
    vocab_size: int,
) -> str:
    """Pretty-print one sentence comparison."""
    lines = [
        "=" * 72,
        f"Vocabulary size: {vocab_size}",
        "-" * 72,
        f"Original sentence : {sentence}",
        f"BPE tokens        : {' | '.join(bpe_tokens)}",
        f"BPE token count   : {len(bpe_tokens)}",
        f"Unigram tokens    : {' | '.join(unigram_tokens)}",
        f"Unigram token count: {len(unigram_tokens)}",
        "",
    ]
    return "\n".join(lines)


def run_tests(
    models_dir: Path | str = MODELS_DIR,
    output_path: Path | str = SAMPLE_OUTPUT,
    sentences: list[str] | None = None,
    vocab_sizes: tuple[int, ...] = VOCAB_SIZES,
) -> str:
    """
    Tokenize test sentences with all BPE/Unigram models and save results.
    """
    models_dir = Path(models_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if sentences is None:
        sentences = DEFAULT_SENTENCES + MORPHOLOGY_EXAMPLES

    sections: list[str] = [
        "Myanmar Subword Tokenization – Sample Outputs",
        "BPE vs Unigram (SentencePiece)",
        "",
    ]

    for vocab_size in vocab_sizes:
        bpe = load_processor(models_dir / f"bpe_{vocab_size}.model")
        uni = load_processor(models_dir / f"unigram_{vocab_size}.model")
        for sentence in sentences:
            bpe_tokens = encode_pieces(bpe, sentence)
            uni_tokens = encode_pieces(uni, sentence)
            sections.append(
                format_comparison(sentence, bpe_tokens, uni_tokens, vocab_size)
            )

    report = "\n".join(sections)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved → {output_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test BPE and Unigram tokenizers on Myanmar sentences."
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(MODELS_DIR),
        help="Directory containing trained .model files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(SAMPLE_OUTPUT),
        help="Path for sample_outputs.txt",
    )
    args = parser.parse_args()
    run_tests(models_dir=args.models_dir, output_path=args.output)


if __name__ == "__main__":
    main()
