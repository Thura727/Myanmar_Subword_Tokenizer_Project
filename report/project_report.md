# A Comparative Study of Myanmar Subword Tokenization Using SentencePiece BPE and Unigram Language Model

**Academic Mini Project Report**

---

## 1. Abstract

Myanmar (Burmese) is written without explicit word boundaries, which makes tokenisation a non-trivial first step for neural NLP. This project trains custom SentencePiece tokenisers on a Myanmar Wikipedia–based corpus and systematically compares **Byte-Pair Encoding (BPE)** and the **Unigram language model** at vocabulary sizes 4,000, 8,000 and 16,000. We evaluate compression ratio, out-of-vocabulary (OOV) rate, average token length and token count on a held-out split, visualise vocabulary-size effects, and perform compound-word error analysis. Empirically, larger vocabularies increase compression for both algorithms; BPE tends to compress slightly more aggressively, while Unigram more often preserves morphologically coherent Myanmar compounds and loanword spans. The study provides a reproducible Python pipeline suitable for an NLP university mini-project presentation.

---

## 2. Introduction

Subword tokenisation mitigates open-vocabulary problems in neural text processing by representing rare words as compositions of frequent character sequences (Sennrich et al., 2016; Kudo, 2018). For Latin-script languages, whitespace already provides a weak word prior. Myanmar script, by contrast, concatenates syllables and compounds without spaces (Thu et al.), so tokenisers cannot rely on whitespace and must discover units from character statistics alone.

SentencePiece (Kudo & Richardson, 2018) trains directly from raw sentences and supports both BPE and Unigram objectives with language-independent normalisation. This project asks:

1. How do BPE and Unigram differ on Myanmar text under matched vocabulary budgets?
2. How does vocabulary size affect compression and OOV behaviour?
3. Which algorithm better respects Myanmar morphological compounds in qualitative error analysis?

---

## 3. Background

### 3.1 Myanmar writing system

Myanmar orthography is an abugida. Visual syllables combine consonants, medial, vowels and tones. Word-level spaces are optional and often absent in running text. Compound nouns such as `ကျန်းမာရေးဝန်ကြီးဌာန` (Ministry of Health) fuse multiple semantic stems into one continuous string. Classical Myanmar NLP therefore emphasised syllable segmentation before statistical modelling (Thu et al.).

### 3.2 Byte-Pair Encoding (BPE)

BPE starts from characters and iteratively merges the most frequent adjacent pair until a vocab budget is reached (Sennrich et al., 2016). Encoding is deterministic given the ordered merge table. Frequent collocations become single pieces; rare forms remain fragmented.

### 3.3 Unigram language model

The Unigram tokeniser maintains a seed vocabulary of candidates and prunes pieces to maximise a unigram language-model likelihood on the corpus (Kudo, 2018). At inference it selects a segmentation (often Viterbi) under piece probabilities, and optionally samples alternatives for subword regularisation.

### 3.4 SentencePiece

SentencePiece implements both algorithms as a C++ library with Python bindings, end-to-end training, and reversible detokenisation (Kudo & Richardson, 2018). Setting `character_coverage=0.9995` is recommended for languages with large character sets so that rare characters map to `<unk>` rather than exhausting the alphabet budget.

---

## 4. Related Work

- **Sennrich et al. (2016)** introduced BPE for NMT, showing gains on rare words via open-vocabulary subwords.
- **Kudo (2018)** proposed Unigram LM tokenisation and subword regularisation with multiple segmentations.
- **Kudo & Richardson (2018)** released SentencePiece as a practical, language-independent trainer/detokeniser used widely in multilingual pretrained models.
- **Thu et al.** surveyed Myanmar word segmentation and tokenisation challenges arising from the lack of spaces and the need for syllable-aware processing.

This project does not propose a new algorithm; it applies SentencePiece to Myanmar and reports a controlled BPE–Unigram comparison for pedagogy and local NLP practice.

---

## 5. Methodology

### 5.1 Preprocessing

Raw lines undergo:

1. HTML tag removal  
2. Removal of disallowed symbols  
3. Unicode NFC normalisation  
4. Whitespace normalisation  
5. Empty-line filtering  
6. Exact-line deduplication  
7. Myanmar-script minimum filter  

A 90/10 deterministic shuffle split yields `train_corpus.txt` and `eval_corpus.txt`.

### 5.2 Training protocol

| Setting | Value |
|---------|-------|
| Algorithms | BPE, Unigram |
| Vocabulary sizes | 4,000 / 8,000 / 16,000 |
| `character_coverage` | 0.9995 |
| Normalisation | `identity` (preserve Myanmar orthography) |
| Implementation | `sentencepiece` Python package |

Six models are written to `models/{bpe,unigram}_{vocab}.model`.

### 5.3 Evaluation metrics

On the held-out eval corpus:

- **Compression ratio** = non-space characters / tokens  
- **OOV rate** = `<unk>` tokens / tokens  
- **Average token length** = mean characters per piece (`▁` stripped)  
- **Token count** = total pieces  

Higher compression (with stable OOV) indicates more efficient packaging of Myanmar character sequences into vocabulary items.

### 5.4 Error analysis protocol

For curated compounds (and additional loanword / scholarly compounds), we treat **over-segmentation** (more pieces for the same surface form) as morphologically worse, then compare BPE vs Unigram at vocab=8,000.

---

## 6. Dataset

| Item | Value |
|------|-------|
| Primary source | Myanmar Wikipedia (`wikimedia/wikipedia`, config `20231101.my`) |
| Bundled extract | ~80,000 lines under `data/external/wiki_my.txt` |
| After cleaning | 80,746 unique lines |
| Train / eval | 72,672 / 8,074 |
| Training characters | ≈ 6.6M |
| Alphabet (coverage pass) | ≈ 130 symbols |

Synthetic morphology-focused sentences (health, education, development compounds) are mixed in so that project-critical forms appear in training.

---

## 7. Implementation

Python 3.10+ modules:

| Module | Role |
|--------|------|
| `generate_corpus.py` | Assemble raw corpus |
| `preprocessing.py` | Clean + split |
| `train_tokenizer.py` | SentencePiece grid training |
| `tokenizer_test.py` | Sample encodings → `sample_outputs.txt` |
| `evaluation.py` | Metrics CSV + error analysis |
| `visualization.py` | Matplotlib figures |
| `run_pipeline.py` | One-shot reproducible run |

Dependencies: `sentencepiece`, `pandas`, `numpy`, `matplotlib`, `regex`, `tqdm`, Jupyter.

---

## 8. Experiments

1. Train six models with identical coverage and corpus.  
2. Encode fixed demo sentences (project brief).  
3. Score all models on the held-out eval set.  
4. Plot compression, OOV, and vocabulary-size effects.  
5. Analyse compounds such as `ကျန်းမာရေးဝန်ကြီးဌာန`, `ဖွံ့ဖြိုးတိုးတက်မှု`, `ပညာရေးစနစ်`.

Hardware: standard CPU environment; full pipeline completes in minutes on the bundled corpus.

---

## 9. Results

### 9.1 Quantitative comparison

Results from `results/comparison.csv` (held-out evaluation):

| Model | Vocab | Actual vocab | Token count | Compression | OOV rate | Avg. token length |
|-------|------:|-------------:|------------:|------------:|---------:|------------------:|
| BPE | 4000 | 4000 | 183,348 | 3.6855 | 0.001876 | 3.6954 |
| BPE | 8000 | 8000 | 158,401 | 4.2659 | 0.002172 | 4.2743 |
| BPE | 16000 | 16000 | 139,997 | 4.8267 | 0.002457 | 4.8340 |
| Unigram | 4000 | 4000 | 190,944 | 3.5388 | 0.001802 | 3.9524 |
| Unigram | 8000 | 8000 | 163,911 | 4.1225 | 0.002099 | 4.4378 |
| Unigram | 16000 | 16000 | 144,602 | 4.6730 | 0.002379 | 4.9059 |

**Observations**

1. **Vocabulary size effect:** For both algorithms, larger vocabularies reduce token count and raise compression ratio and average token length — expected as longer pieces enter the vocabulary.  
2. **BPE vs Unigram:** At matched sizes, BPE yields higher compression (fewer tokens). Unigram shows slightly lower OOV and somewhat longer average pieces in characters, consistent with keeping high-probability multi-character units.  
3. **OOV remains low** (≈0.18–0.25%) thanks to `character_coverage=0.9995` and Myanmar’s compact alphabet relative to Chinese/Japanese.

### 9.2 Qualitative demo (vocab = 8,000)

Example: `ကျန်းမာရေးဝန်ကြီးဌာန`

- BPE: `▁ကျန်းမာရေး | ဝန်ကြီးဌာန`  
- Unigram: `▁ကျန်းမာရေး | ဝန်ကြီးဌာန`  

Both recover the transparent compound boundary between “health” and “ministry”. Differences appear more clearly on rarer loanwords and scholarly compounds (Section 11).

### 9.3 Figures

Saved under `results/graphs/`:

1. `compression_ratio.png` — BPE vs Unigram compression  
2. `oov_rate.png` — BPE vs Unigram OOV  
3. `vocab_size_effect.png` — token count & average length vs vocab  
4. `token_count_comparison.png` — grouped bars  

---

## 10. Discussion

Myanmar’s lack of whitespace forces SentencePiece to induce units from character co-occurrence alone. BPE’s greedy frequency merges favour corpus-frequent collocations and therefore stronger **compression**. Unigram’s likelihood objective prefers pieces that explain the corpus under a probabilistic model; this often aligns better with **human-interpretable stems** inside compounds and transliterations.

Neither algorithm is universally “better”:

- Prefer **BPE** when sequence length / embedding table efficiency is primary (higher compression).  
- Prefer **Unigram** when morphological transparency and optional subword sampling matter (Kudo, 2018).  

Vocabulary size is a strong lever: moving from 4k → 16k increases BPE compression from 3.69 to 4.83 characters/token on our eval set. Practitioners should tune vocab against downstream BLEU/accuracy, not compression alone.

Limitations: Wikipedia genre bias; synthetic seed sentences; intrinsic metrics only (no NMT/LM downstream task in this mini-project).

---

## 11. Error Analysis

Focus vocabulary: **8,000**.

### 11.1 Required compounds

| Word | BPE | Unigram | Note |
|------|-----|---------|------|
| ကျန်းမာရေးဝန်ကြီးဌာန | `ကျန်းမာရေး` + `ဝန်ကြီးဌာန` | same | Morphology-preserving |
| ဖွံ့ဖြိုးတိုးတက်မှု | `ဖွံ့ဖြိုး` \| `တိုးတက်` \| `မှု` | `ဖွံ့ဖြိုးတိုးတက်` \| `မှု` | Unigram keeps development stem longer |
| ပညာရေးစနစ် | single piece | single piece | Frequent education compound |
| လူမှုကာကွယ်ရေး | `လူမှုကာကွယ်` \| `ရေး` | single piece | **BPE over-segments** |

### 11.2 Cases where BPE is worse

| Word | BPE (fragments) | Unigram |
|------|-----------------|---------|
| သမဝါယမကောလိပ် | many short pieces | `သမဝါယမ` \| `ကောလိပ်` |
| အရှေ့တိုင်းဒဿနိကဗေဒ | splits scholarly stem | keeps `ဒဿနိကဗေဒ` |
| ဂလိုဘယ်လိုက်ဇေးရှင်း | character-ish splits | `ဂလိုဘယ်` \| `လိုက်ဇေးရှင်း` |
| ဖေ့စ်ဘွတ်ခ်စာမျက်နှာ | shattered loanword | `ဖေ့စ်ဘွတ်ခ်` \| `စာမျက်နှာ` |

**Interpretation.** On frequent native administrative compounds both systems behave similarly. On rarer compounds, Pali/Sanskrit scholarly forms, and English loanword transliterations, **Unigram preserves morphology better**, while BPE over-segments into less meaningful fragments. This matches Unigram’s bias toward high-probability longer pieces (Kudo, 2018).

---

## 12. Conclusion

This project delivered a complete, reproducible comparison of SentencePiece **BPE** and **Unigram** for Myanmar subword tokenisation. Larger vocabularies improve compression for both methods; BPE compresses slightly more, while Unigram more often respects Myanmar compound and loanword structure in qualitative analysis. Future work should measure downstream impact on Myanmar neural MT or language modelling and explore hybrid syllable-aware pre-tokenisation.

---

## 13. References

1. Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. https://doi.org/10.18653/v1/D18-2012  

2. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016)*.  

3. Kudo, T. (2018). Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates. *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL 2018)*.  

4. Thu, Y. K., et al. A Study on Myanmar Word Segmentation and Tokenization.  

5. Google. SentencePiece. https://github.com/google/sentencepiece  

6. Wikimedia Foundation. Wikipedia dataset (`20231101.my`) via Hugging Face Datasets. https://huggingface.co/datasets/wikimedia/wikipedia  
