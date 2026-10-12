# A Comparative Study of Myanmar Subword Tokenization Using SentencePiece BPE and Unigram Language Model

Academic mini project (Python) that trains custom Myanmar SentencePiece tokenizers and compares **BPE** vs **Unigram** across vocabulary sizes 4k / 8k / 16k.

## Features

- Myanmar text preprocessing (HTML removal, Unicode NFC, deduplication)
- SentencePiece BPE and Unigram training (`character_coverage=0.9995`)
- Tokenization demos on curated Myanmar sentences
- Metrics: compression ratio, OOV rate, average token length, token count
- Matplotlib comparison graphs
- Automatic morphology / compound-word error analysis
- Academic Markdown report + Jupyter demonstration notebook

## Project structure

```
Myanmar_Subword_Tokenizer_Project/
├── data/
│   ├── external/wiki_my.txt      # Myanmar Wikipedia extract (source)
│   ├── raw/myanmar_corpus.txt
│   └── processed/
│       ├── cleaned_corpus.txt
│       ├── train_corpus.txt
│       └── eval_corpus.txt
├── models/                      # *.model / *.vocab
├── src/
│   ├── generate_corpus.py
│   ├── fetch_wikipedia.py
│   ├── preprocessing.py
│   ├── train_tokenizer.py
│   ├── tokenizer_test.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── run_pipeline.py
├── results/
│   ├── comparison.csv
│   ├── sample_outputs.txt
│   ├── error_analysis.txt
│   └── graphs/
├── report/project_report.md
├── notebooks/demo.ipynb
└── requirements.txt
```

## Requirements

- Python 3.10+
- See `requirements.txt`

## Installation

```bash
cd Myanmar_Subword_Tokenizer_Project
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional (only if you need to re-download Wikipedia text):

```bash
pip install datasets
python3 -m src.fetch_wikipedia
```

## Quick start (full reproducible pipeline)

From the project root:

```bash
python3 -m src.run_pipeline
```

This will:

1. Build `data/raw/myanmar_corpus.txt` from the Wikipedia extract + morphology seeds  
2. Preprocess and create train/eval splits  
3. Train six SentencePiece models  
4. Run tokenization tests  
5. Evaluate metrics → `results/comparison.csv`  
6. Save graphs under `results/graphs/`  
7. Write morphology error analysis  

## Step-by-step commands

```bash
# 1) Corpus
python3 -m src.generate_corpus

# 2) Preprocess
python3 -m src.preprocessing

# 3) Train BPE + Unigram (4k/8k/16k)
python3 -m src.train_tokenizer

# 4) Sample tokenization outputs
python3 -m src.tokenizer_test

# 5) Evaluation + error analysis
python3 -m src.evaluation --error-analysis

# 6) Graphs
python3 -m src.visualization
```

## Jupyter notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

## Key metrics (definition)

| Metric | Formula / definition |
|--------|----------------------|
| Compression ratio | `characters / tokens` (spaces excluded from character count) |
| OOV rate | `count(<unk>) / total_tokens` |
| Average token length | mean characters per piece (SentencePiece `▁` stripped) |
| Token count | total pieces on the held-out eval corpus |

## Sample test sentences

- `မြန်မာနိုင်ငံသည် ဖွံ့ဖြိုးတိုးတက်လာနေသည်။`
- `ကျန်းမာရေးဝန်ကြီးဌာနသည် ပြည်သူများအတွက် ဝန်ဆောင်မှုပေးနေသည်။`

## Dataset note

The bundled extract under `data/external/wiki_my.txt` is sampled from [Wikimedia Wikipedia (`20231101.my`)](https://huggingface.co/datasets/wikimedia/wikipedia) via Hugging Face Datasets. Regenerating it requires network access and the `datasets` package.

## References

1. Kudo, T., & Richardson, J. (2018). *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing.* EMNLP. https://doi.org/10.18653/v1/D18-2012  
2. Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016.  
3. Kudo, T. (2018). *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates.* ACL 2018.  
4. Thu, Y. K., et al. *A Study on Myanmar Word Segmentation and Tokenization.*  
5. SentencePiece: https://github.com/google/sentencepiece  

## License

Project code is provided for academic / educational use. Wikipedia text remains under Wikimedia licensing terms.
