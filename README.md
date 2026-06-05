# N-gram Language Model

A from-scratch implementation of N-gram language models (Unigram, Bigram, Trigram) with three probability estimation techniques, evaluated across three distinct text corpora.

## Overview

This project implements and compares N-gram language models built entirely using Python standard libraries and `nltk` — no deep learning frameworks. The goal is to understand how smoothing techniques affect model quality across domains with varying vocabulary and structure.

### What's implemented

**Models**
- Unigram, Bigram, and Trigram language models

**Smoothing / Estimation Techniques**
- Maximum Likelihood Estimation (MLE)
- Laplace (Add-1) Smoothing
- Good-Turing Smoothing

**Evaluation**
- Perplexity on three out-of-domain test corpora

## Dataset

Training data: Wikipedia text corpus (`wiki_train.csv`) — not included in this repo due to size (~211MB). Download from [Hugging Face: wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) or the course data source.

Test sets (included):

| File | Domain |
|------|--------|
| `wiki_test.csv` | Wikipedia (in-domain) |
| `arxiv_test.csv` | Scientific abstracts (ArXiv) |
| `financial_test.csv` | Financial news |

## Preprocessing Pipeline

- Lowercasing
- Number normalization (`num`, `year`, `percent` tokens)
- Punctuation removal
- Lemmatization (WordNet)
- OOV handling via `<UNK>` token (min frequency threshold = 2)
- Sentence boundary markers `<s>` and `</s>`

## Project Structure

```
N-gram_Language_Model/
├── data/
│   ├── raw/                  # Test CSVs (wiki_train.csv excluded)
│   └── processed/            # Placeholder for intermediate outputs
├── notebooks/
│   └── ngram_language_model.ipynb   # Full implementation and results
├── reports/
│   ├── Report.pdf            # Written report
│   └── figures/
├── requirements.txt
└── README.md
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK resources (run once)
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# Open the notebook
jupyter notebook notebooks/ngram_language_model.ipynb
```

> **Note:** Place `wiki_train.csv` in `data/raw/` before running Section A. The notebook uses relative paths.

## Results Summary

Perplexity scores (lower = better) across all model-smoothing combinations on three test sets. Key findings:
- Trigram MLE collapses on out-of-domain data (zero-probability problem without smoothing)
- Good-Turing outperforms Laplace on sparse n-grams
- Financial corpus shows highest perplexity across all models due to domain shift from Wikipedia training

## Dependencies

- Python 3.x
- `pandas`
- `nltk`
- Standard library: `collections`, `re`, `math`, `time`
