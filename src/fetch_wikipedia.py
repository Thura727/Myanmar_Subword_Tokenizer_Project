"""
Optional helper: download a Myanmar Wikipedia extract for the training corpus.

Usage:
  python3 -m src.fetch_wikipedia

Requires: ``pip install datasets``
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import regex

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_WIKI = PROJECT_ROOT / "data" / "external" / "wiki_my.txt"
MYANMAR = regex.compile(r"[\u1000-\u109F]")


def fetch_wikipedia(
    output_path: Path | str = EXTERNAL_WIKI,
    max_articles: int = 5000,
    max_lines: int = 80000,
    seed: int = 42,
) -> Path:
    from datasets import load_dataset

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.my",
        split="train",
        streaming=True,
    )

    collected: list[str] = []
    n_articles = 0
    for row in ds:
        text = row.get("text") or ""
        title = (row.get("title") or "").strip()
        if title:
            collected.append(title)
        for ln in text.splitlines():
            ln = ln.strip()
            for part in ln.replace("။", "။\n").splitlines():
                part = part.strip()
                if len(MYANMAR.findall(part)) >= 8:
                    collected.append(part)
        n_articles += 1
        if n_articles >= max_articles:
            break

    rng = random.Random(seed)
    rng.shuffle(collected)
    seen: set[str] = set()
    unique: list[str] = []
    for ln in collected:
        if ln in seen:
            continue
        seen.add(ln)
        unique.append(ln)
        if len(unique) >= max_lines:
            break

    output_path.write_text("\n".join(unique) + "\n", encoding="utf-8")
    print(
        f"Saved {len(unique)} lines from {n_articles} articles → {output_path}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Myanmar Wikipedia text.")
    parser.add_argument("--output", type=str, default=str(EXTERNAL_WIKI))
    parser.add_argument("--max-articles", type=int, default=5000)
    parser.add_argument("--max-lines", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    fetch_wikipedia(
        output_path=args.output,
        max_articles=args.max_articles,
        max_lines=args.max_lines,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
