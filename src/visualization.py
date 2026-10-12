"""
Matplotlib visualizations comparing BPE and Unigram Myanmar tokenizers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_CSV = PROJECT_ROOT / "results" / "comparison.csv"
GRAPHS_DIR = PROJECT_ROOT / "results" / "graphs"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )


def plot_compression_ratio(df: pd.DataFrame, out_dir: Path) -> Path:
    """BPE vs Unigram compression ratio across vocabulary sizes."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, marker in (("BPE", "o"), ("UNIGRAM", "s")):
        subset = df[df["Model"] == model].sort_values("Vocabulary size")
        ax.plot(
            subset["Vocabulary size"],
            subset["Compression ratio"],
            marker=marker,
            linewidth=2,
            label=model,
        )
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel("Compression ratio (characters / tokens)")
    ax.set_title("BPE vs Unigram – Compression Ratio")
    ax.legend()
    ax.set_xticks(sorted(df["Vocabulary size"].unique()))
    path = out_dir / "compression_ratio.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_oov_rate(df: pd.DataFrame, out_dir: Path) -> Path:
    """BPE vs Unigram OOV rate across vocabulary sizes."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, marker in (("BPE", "o"), ("UNIGRAM", "s")):
        subset = df[df["Model"] == model].sort_values("Vocabulary size")
        ax.plot(
            subset["Vocabulary size"],
            subset["OOV rate"],
            marker=marker,
            linewidth=2,
            label=model,
        )
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel("OOV rate")
    ax.set_title("BPE vs Unigram – OOV Rate")
    ax.legend()
    ax.set_xticks(sorted(df["Vocabulary size"].unique()))
    path = out_dir / "oov_rate.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_vocab_size_effect(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Vocabulary size effect on token count and average token length.

    Two-panel figure: token count (left) and average token length (right).
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for model, marker in (("BPE", "o"), ("UNIGRAM", "s")):
        subset = df[df["Model"] == model].sort_values("Vocabulary size")
        axes[0].plot(
            subset["Vocabulary size"],
            subset["Token count"],
            marker=marker,
            linewidth=2,
            label=model,
        )
        axes[1].plot(
            subset["Vocabulary size"],
            subset["Average token length"],
            marker=marker,
            linewidth=2,
            label=model,
        )

    axes[0].set_title("Vocabulary size → Token count")
    axes[0].set_xlabel("Vocabulary size")
    axes[0].set_ylabel("Total tokens on evaluation corpus")
    axes[0].legend()
    axes[0].set_xticks(sorted(df["Vocabulary size"].unique()))

    axes[1].set_title("Vocabulary size → Average token length")
    axes[1].set_xlabel("Vocabulary size")
    axes[1].set_ylabel("Average characters per token")
    axes[1].legend()
    axes[1].set_xticks(sorted(df["Vocabulary size"].unique()))

    fig.suptitle("Vocabulary Size Effect on Myanmar Tokenization", y=1.02)
    path = out_dir / "vocab_size_effect.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_token_count_bars(df: pd.DataFrame, out_dir: Path) -> Path:
    """Grouped bar chart of token counts (extra comparison view)."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    vocab_sizes = sorted(df["Vocabulary size"].unique())
    x = range(len(vocab_sizes))
    width = 0.35
    bpe_vals = [
        df[(df["Model"] == "BPE") & (df["Vocabulary size"] == v)][
            "Token count"
        ].values[0]
        for v in vocab_sizes
    ]
    uni_vals = [
        df[(df["Model"] == "UNIGRAM") & (df["Vocabulary size"] == v)][
            "Token count"
        ].values[0]
        for v in vocab_sizes
    ]
    ax.bar([i - width / 2 for i in x], bpe_vals, width, label="BPE")
    ax.bar([i + width / 2 for i in x], uni_vals, width, label="UNIGRAM")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(v) for v in vocab_sizes])
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel("Token count")
    ax.set_title("BPE vs Unigram – Token Count Comparison")
    ax.legend()
    path = out_dir / "token_count_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def generate_all_graphs(
    comparison_csv: Path | str = COMPARISON_CSV,
    graphs_dir: Path | str = GRAPHS_DIR,
) -> list[Path]:
    """Load comparison.csv and write all required graphs."""
    comparison_csv = Path(comparison_csv)
    graphs_dir = Path(graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    if not comparison_csv.exists():
        raise FileNotFoundError(
            f"Missing {comparison_csv}. Run evaluation.py first."
        )

    df = pd.read_csv(comparison_csv)
    # Normalize model names in case of mixed casing
    df["Model"] = df["Model"].str.upper()

    paths = [
        plot_compression_ratio(df, graphs_dir),
        plot_oov_rate(df, graphs_dir),
        plot_vocab_size_effect(df, graphs_dir),
        plot_token_count_bars(df, graphs_dir),
    ]
    print("Graphs saved:")
    for p in paths:
        print(f"  {p}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BPE vs Unigram comparison graphs."
    )
    parser.add_argument("--csv", type=str, default=str(COMPARISON_CSV))
    parser.add_argument("--graphs-dir", type=str, default=str(GRAPHS_DIR))
    args = parser.parse_args()
    generate_all_graphs(args.csv, args.graphs_dir)


if __name__ == "__main__":
    main()
