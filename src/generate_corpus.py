"""
Build the raw Myanmar training corpus.

Primary source (preferred): Myanmar Wikipedia extract cached under
``data/external/wiki_my.txt``. If that file is missing, falls back to a
synthetic template corpus so the project remains runnable offline.

Also injects a small amount of noisy HTML / symbol lines so the
preprocessing pipeline can be demonstrated.
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "myanmar_corpus.txt"
EXTERNAL_WIKI = PROJECT_ROOT / "data" / "external" / "wiki_my.txt"

SUBJECTS = [
    "မြန်မာနိုင်ငံ",
    "ရန်ကုန်မြို့",
    "မန္တလေးမြို့",
    "နေပြည်တော်",
    "အစိုးရ",
    "ပြည်သူများ",
    "ကျောင်းသားများ",
    "ကျန်းမာရေးဝန်ကြီးဌာန",
    "ပညာရေးဝန်ကြီးဌာန",
    "တက္ကသိုလ်",
    "ဆေးရုံ",
    "စီးပွားရေးလုပ်ငန်းများ",
    "နိုင်ငံတော်",
    "ဒေသခံပြည်သူများ",
]

VERBS = [
    "ဖွံ့ဖြိုးတိုးတက်လာနေသည်",
    "ကြိုးပမ်းဆောင်ရွက်နေသည်",
    "ဝန်ဆောင်မှုပေးနေသည်",
    "ပူးပေါင်းဆောင်ရွက်နေသည်",
    "အကောင်အထည်ဖော်နေသည်",
    "တိုးတက်ကောင်းမွန်လာသည်",
]

OBJECTS = [
    "ပညာရေးစနစ်",
    "ကျန်းမာရေးစောင့်ရှောက်မှု",
    "စီးပွားရေးဖွံ့ဖြိုးတိုးတက်မှု",
    "လူမှုကာကွယ်ရေးအစီအစဉ်",
    "သတင်းအချက်အလက်နည်းပညာ",
    "သဘာဝပတ်ဝန်းကျင်ထိန်းသိမ်းရေး",
]

SEED_SENTENCES = [
    "မြန်မာနိုင်ငံသည် ဖွံ့ဖြိုးတိုးတက်လာနေသည်။",
    "ကျန်းမာရေးဝန်ကြီးဌာနသည် ပြည်သူများအတွက် ဝန်ဆောင်မှုပေးနေသည်။",
    "ပညာရေးစနစ်ကို ခေတ်မီအောင် ပြုပြင်ပြောင်းလဲသင့်သည်။",
    "ဖွံ့ဖြိုးတိုးတက်မှုသည် ပညာရေးနှင့် ကျန်းမာရေးအပေါ် မူတည်သည်။",
    "မြန်မာစာသည် သီးခြားစာလုံးခွဲခြားမှု မရှိသော ဘာသာစကားဖြစ်သည်။",
    "ဘိုက်အဆင့် စကားလုံးခွဲခြားနည်းသည် ရှားပါးစကားလုံးများအတွက် အသုံးဝင်သည်။",
    "ယူနီဂရမ်ဘာသာစကားပုံစံသည် ဖြစ်နိုင်ခြေအခြေခံ စကားလုံးခွဲခြားမှုကို အသုံးပြုသည်။",
    "ကျန်းမာရေးဝန်ကြီးဌာနကို မည်သို့ခွဲခြားမည်နည်း။",
    "ဖွံ့ဖြိုးတိုးတက်မှုကို မည်သို့ခွဲခြားမည်နည်း။",
    "ပညာရေးစနစ်ကို မည်သို့ခွဲခြားမည်နည်း။",
]


def synthetic_lines(rng: random.Random, n: int = 3000) -> list[str]:
    """Fallback / supplement sentences with morphology-focused compounds."""
    lines = list(SEED_SENTENCES)
    for subj, verb, obj in itertools.product(SUBJECTS, VERBS, OBJECTS):
        lines.append(f"{subj}သည် {verb}။")
        lines.append(f"{subj}သည် {obj} အတွက် {verb}။")
        lines.append(f"{obj}နှင့် ပတ်သက်၍ {subj}က ရှင်းလင်းပြောကြားခဲ့သည်။")
    compounds = [
        "ကျန်းမာရေးဝန်ကြီးဌာန",
        "ဖွံ့ဖြိုးတိုးတက်မှု",
        "ပညာရေးစနစ်",
        "ပြည်သူ့ကျန်းမာရေး",
        "သတင်းအချက်အလက်နည်းပညာ",
        "လူမှုကာကွယ်ရေး",
    ]
    for c in compounds:
        for subj in SUBJECTS:
            lines.append(f"{subj}သည် {c}ကို အရေးထား ဆောင်ရွက်နေသည်။")
            lines.append(f"{c}သည် နိုင်ငံ့ဖွံ့ဖြိုးရေးအတွက် အရေးကြီးသည်။")
    rng.shuffle(lines)
    return lines[:n]


def load_wiki_lines(path: Path, max_lines: int | None = None) -> list[str]:
    if not path.exists():
        return []
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                lines.append(ln)
            if max_lines is not None and len(lines) >= max_lines:
                break
    return lines


def inject_noise(lines: list[str], rng: random.Random) -> list[str]:
    """Insert blank / HTML / symbol noise for preprocessing demos."""
    noisy: list[str] = []
    extras = [
        "",
        "<div>12345</div>",
        "<p>ဤစာကြောင်းတွင် HTML တဂ်များ ပါဝင်သည်။</p>",
        "ကျန်းမာရေး@@@ဝန်ကြီးဌာန###သည် ပြည်သူများအတွက် ဝန်ဆောင်မှုပေးနေသည်!!!",
        "   မြန်မာနိုင်ငံသည်    ဖွံ့ဖြိုးတိုးတက်လာနေသည်။   ",
    ]
    for i, line in enumerate(lines):
        if i % 1200 == 0:
            noisy.append(rng.choice(extras))
        noisy.append(line)
    # Guarantee seed sentences + intentional duplicates for dedup demo
    noisy.extend(SEED_SENTENCES)
    noisy.extend(SEED_SENTENCES[:3])
    return noisy


def write_corpus(
    output_path: Path | str = RAW_PATH,
    wiki_path: Path | str = EXTERNAL_WIKI,
    max_wiki_lines: int = 80000,
    seed: int = 42,
) -> Path:
    """
    Write ``data/raw/myanmar_corpus.txt``.

    Prefer Wikipedia extract; always append synthetic morphology examples.
    """
    output_path = Path(output_path)
    wiki_path = Path(wiki_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    wiki = load_wiki_lines(wiki_path, max_lines=max_wiki_lines)
    synth = synthetic_lines(rng, n=4000)
    if wiki:
        print(f"Using Wikipedia extract: {len(wiki)} lines from {wiki_path}")
        combined = wiki + synth
    else:
        print("WARNING: Wikipedia extract missing; using synthetic corpus only.")
        print(f"  Place a Myanmar text file at {wiki_path} for best results.")
        combined = synth * 20  # expand fallback

    rng.shuffle(combined)
    combined = inject_noise(combined, rng)
    output_path.write_text("\n".join(combined) + "\n", encoding="utf-8")
    print(f"Wrote {len(combined)} lines → {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Myanmar raw corpus.")
    parser.add_argument("--output", type=str, default=str(RAW_PATH))
    parser.add_argument("--wiki", type=str, default=str(EXTERNAL_WIKI))
    parser.add_argument("--max-wiki-lines", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write_corpus(
        output_path=args.output,
        wiki_path=args.wiki,
        max_wiki_lines=args.max_wiki_lines,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
