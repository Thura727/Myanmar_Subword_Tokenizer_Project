"""
Evaluation metrics for Myanmar SentencePiece tokenizers.

Metrics
-------
1. Compression ratio  = characters / tokens
2. OOV rate           = unknown pieces / total pieces
3. Average token length (characters per piece, excluding SentencePiece ▁)
4. Token count comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import sentencepiece as spm
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "eval_corpus.txt"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
COMPARISON_CSV = RESULTS_DIR / "comparison.csv"

VOCAB_SIZES = (4000, 8000, 16000)
MODEL_TYPES = ("bpe", "unigram")


def load_processor(model_path: Path) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    if not sp.LoadFromSerializedProto(model_path.read_bytes()):
        raise FileNotFoundError(f"Failed to load model: {model_path}")
    return sp


def strip_meta_marker(piece: str) -> str:
    """Remove SentencePiece whitespace marker ▁ for length statistics."""
    return piece.replace("▁", "")


def evaluate_model(
    sp: spm.SentencePieceProcessor,
    lines: list[str],
    model_name: str,
    vocab_size: int,
) -> dict:
    """
    Compute evaluation metrics on a list of corpus lines.

    OOV is approximated via the SentencePiece ``<unk>`` piece id (unk_id).
    """
    unk_id = sp.unk_id()
    total_chars = 0
    total_tokens = 0
    total_unk = 0
    token_char_lens: list[int] = []

    for line in lines:
        # Character count excludes whitespace for a fair Myanmar comparison
        # (Myanmar text is largely space-free; keeping spaces would inflate
        # the denominator inconsistently across mixed lines).
        chars = len(line.replace(" ", ""))
        ids = sp.encode(line, out_type=int)
        pieces = sp.encode(line, out_type=str)

        total_chars += chars
        total_tokens += len(ids)
        total_unk += sum(1 for i in ids if i == unk_id)
        for piece in pieces:
            cleaned = strip_meta_marker(piece)
            if cleaned:
                token_char_lens.append(len(cleaned))

    compression = (total_chars / total_tokens) if total_tokens else 0.0
    oov_rate = (total_unk / total_tokens) if total_tokens else 0.0
    avg_token_len = (
        sum(token_char_lens) / len(token_char_lens) if token_char_lens else 0.0
    )

    return {
        "Model": model_name,
        "Vocabulary size": vocab_size,
        "Actual vocab size": int(sp.get_piece_size()),
        "Token count": total_tokens,
        "Compression ratio": round(compression, 4),
        "OOV rate": round(oov_rate, 6),
        "Average token length": round(avg_token_len, 4),
    }


def run_evaluation(
    corpus_path: Path | str = CORPUS_PATH,
    models_dir: Path | str = MODELS_DIR,
    output_csv: Path | str = COMPARISON_CSV,
    max_lines: int | None = None,
    vocab_sizes: tuple[int, ...] = VOCAB_SIZES,
) -> pd.DataFrame:
    """
    Evaluate all trained models and write ``results/comparison.csv``.
    """
    corpus_path = Path(corpus_path)
    models_dir = Path(models_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if max_lines is not None:
        lines = lines[:max_lines]

    rows: list[dict] = []
    jobs = [
        (model_type, vocab_size)
        for model_type in MODEL_TYPES
        for vocab_size in vocab_sizes
    ]

    for model_type, vocab_size in tqdm(jobs, desc="Evaluating"):
        model_path = models_dir / f"{model_type}_{vocab_size}.model"
        sp = load_processor(model_path)
        row = evaluate_model(
            sp,
            lines,
            model_name=model_type.upper(),
            vocab_size=vocab_size,
        )
        rows.append(row)
        print(
            f"  {row['Model']:8s} vocab={vocab_size:5d}  "
            f"tokens={row['Token count']:8d}  "
            f"CR={row['Compression ratio']:.4f}  "
            f"OOV={row['OOV rate']:.6f}  "
            f"ATL={row['Average token length']:.4f}"
        )

    df = pd.DataFrame(rows)
    # Stable column order matching the project brief
    df = df[
        [
            "Model",
            "Vocabulary size",
            "Actual vocab size",
            "Token count",
            "Compression ratio",
            "OOV rate",
            "Average token length",
        ]
    ]
    df.to_csv(output_csv, index=False)
    print(f"\nComparison table → {output_csv}")
    return df


def error_analysis(
    models_dir: Path | str = MODELS_DIR,
    vocab_size: int = 8000,
) -> list[dict]:
    """
    Compare BPE vs Unigram on Myanmar compound words.

    A segmentation is considered "worse" when it produces more fragments
    (over-segmentation) for the same surface form, which tends to break
    morphological units in agglutinative / compounding Myanmar words.
    """
    models_dir = Path(models_dir)
    compounds = [
        # Required project compounds
        "ကျန်းမာရေးဝန်ကြီးဌာန",
        "ဖွံ့ဖြိုးတိုးတက်မှု",
        "ပညာရေးစနစ်",
        "ကျန်းမာရေး",
        "ပညာရေး",
        "စီးပွားရေး",
        "နိုင်ငံတော်",
        "ပြည်သူ့ကျန်းမာရေး",
        "သတင်းအချက်အလက်နည်းပညာ",
        "လူမှုကာကွယ်ရေး",
        # Additional cases where BPE often over-segments vs Unigram
        "သမဝါယမကောလိပ်",
        "အရှေ့တိုင်းဒဿနိကဗေဒ",
        "စားနပ်ရိက္ခာဖူလုံမှု",
        "ဂလိုဘယ်လိုက်ဇေးရှင်း",
        "ယူဂိုစလားဗီးယားနိုင်ငံ",
        "ဖေ့စ်ဘွတ်ခ်စာမျက်နှာ",
    ]

    bpe = load_processor(models_dir / f"bpe_{vocab_size}.model")
    uni = load_processor(models_dir / f"unigram_{vocab_size}.model")

    findings: list[dict] = []
    for word in compounds:
        bpe_toks = bpe.encode(word, out_type=str)
        uni_toks = uni.encode(word, out_type=str)
        verdict = "similar"
        if len(bpe_toks) > len(uni_toks):
            verdict = "BPE over-segments (worse)"
        elif len(uni_toks) > len(bpe_toks):
            verdict = "Unigram over-segments (worse)"

        findings.append(
            {
                "word": word,
                "bpe_tokens": bpe_toks,
                "unigram_tokens": uni_toks,
                "bpe_count": len(bpe_toks),
                "unigram_count": len(uni_toks),
                "verdict": verdict,
            }
        )
    return findings


def write_error_analysis(
    models_dir: Path | str = MODELS_DIR,
    output_path: Path | str | None = None,
    vocab_size: int = 8000,
) -> Path:
    """Append morphology error analysis to sample outputs / dedicated file."""
    models_dir = Path(models_dir)
    if output_path is None:
        output_path = RESULTS_DIR / "error_analysis.txt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    findings = error_analysis(models_dir, vocab_size=vocab_size)
    lines = [
        "Error Analysis: BPE vs Unigram on Myanmar Compound Words",
        f"Vocabulary size compared: {vocab_size}",
        "",
        "Criterion: over-segmentation (more pieces) is treated as worse for",
        "morphology preservation on closed compounds.",
        "",
    ]
    bpe_worse = 0
    uni_worse = 0
    for item in findings:
        if "BPE over-segments" in item["verdict"]:
            bpe_worse += 1
        elif "Unigram over-segments" in item["verdict"]:
            uni_worse += 1
        lines.extend(
            [
                f"Word: {item['word']}",
                f"  BPE     ({item['bpe_count']}): {' | '.join(item['bpe_tokens'])}",
                f"  Unigram ({item['unigram_count']}): "
                f"{' | '.join(item['unigram_tokens'])}",
                f"  Verdict: {item['verdict']}",
                "",
            ]
        )

    if bpe_worse > uni_worse:
        conclusion = (
            "Overall, Unigram preserves Myanmar compound morphology better "
            "on this sample (fewer over-segmentations)."
        )
    elif uni_worse > bpe_worse:
        conclusion = (
            "Overall, BPE preserves Myanmar compound morphology better "
            "on this sample (fewer over-segmentations)."
        )
    else:
        conclusion = (
            "On this sample, BPE and Unigram show similar morphology "
            "preservation; differences are case-specific."
        )

    lines.extend(
        [
            f"BPE-worse cases     : {bpe_worse}",
            f"Unigram-worse cases : {uni_worse}",
            "",
            "Conclusion:",
            conclusion,
            "",
            "Discussion note: Unigram's probabilistic objective often keeps",
            "high-probability multi-character units intact, which aligns with",
            "Myanmar compounding (e.g. ကျန်းမာရေး + ဝန်ကြီးဌာန). BPE merges",
            "by frequency pair counts and can split rare compounds more",
            "aggressively at smaller vocabularies.",
        ]
    )

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nError analysis → {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Myanmar SentencePiece BPE and Unigram models."
    )
    parser.add_argument("--corpus", type=str, default=str(CORPUS_PATH))
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--output", type=str, default=str(COMPARISON_CSV))
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Optional cap on evaluation lines (for quick smoke tests)",
    )
    parser.add_argument(
        "--error-analysis",
        action="store_true",
        help="Also write morphology error analysis",
    )
    args = parser.parse_args()
    run_evaluation(
        corpus_path=args.corpus,
        models_dir=args.models_dir,
        output_csv=args.output,
        max_lines=args.max_lines,
    )
    if args.error_analysis:
        write_error_analysis(models_dir=args.models_dir)


if __name__ == "__main__":
    main()
