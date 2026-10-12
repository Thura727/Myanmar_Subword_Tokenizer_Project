"""
End-to-end pipeline:

  generate corpus → preprocess → train → test → evaluate → visualize → error analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python src/run_pipeline.py`` from project root or src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generate_corpus import write_corpus  # noqa: E402
from src.preprocessing import preprocess_corpus  # noqa: E402
from src.train_tokenizer import train_all_models  # noqa: E402
from src.tokenizer_test import run_tests  # noqa: E402
from src.evaluation import run_evaluation, write_error_analysis  # noqa: E402
from src.visualization import generate_all_graphs  # noqa: E402


def run_all(skip_generate: bool = False, vocab_sizes: tuple[int, ...] | None = None) -> None:
    if not skip_generate:
        print("=== 0. Generating raw corpus ===")
        write_corpus()

    print("\n=== 1. Preprocessing ===")
    preprocess_corpus()

    print("\n=== 2. Training SentencePiece models ===")
    kwargs = {}
    if vocab_sizes is not None:
        kwargs["vocab_sizes"] = vocab_sizes
    train_all_models(**kwargs)

    print("\n=== 3. Tokenization tests ===")
    run_tests()

    print("\n=== 4. Evaluation ===")
    run_evaluation()

    print("\n=== 5. Visualization ===")
    generate_all_graphs()

    print("\n=== 6. Error analysis ===")
    write_error_analysis()

    print("\nPipeline finished successfully.")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run full Myanmar tokenizer study.")
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip corpus generation (use existing raw file)",
    )
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Optional override of vocabulary sizes",
    )
    args = parser.parse_args()
    run_all(
        skip_generate=args.skip_generate,
        vocab_sizes=tuple(args.vocab_sizes) if args.vocab_sizes else None,
    )


if __name__ == "__main__":
    main()
