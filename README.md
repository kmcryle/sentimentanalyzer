# Predicting Audience Voting Outcomes Through Code-Switched Sentiment Analysis with Fine-Tuned BERT-Based Transformers and LSTM Networks

This is a working prototype built for an undergraduate thesis, showcasing how data analytics and NLP techniques can be applied to sentiment analysis on code-switched (Taglish) text — using real Reddit discussions about Pinoy Big Brother as the case study.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C)]()
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E)]()

> 🔗 **Live demo:** https://taglish-sentiment-analyzer.streamlit.app/

---

## Try It

**Requirements:** Python 3.10+

```bash
git clone <this-repo-url>
cd <repo-folder>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run sentiment_app.py
```

Opens at `http://localhost:8501` with three tabs:

| Tab | What it does |
|---|---|
| **Overview** | Study background, the Big Four duos, key findings |
| **Model Evaluation** | Accuracy/F1 comparison, code-switching impact, sentiment distribution, voting alignment |
| **Sentiment Analyzer** | Type a Taglish sentence, pick a model, get a live prediction — or run all four at once |

> Model files (especially `model.safetensors` for the transformer models) are large. If you clone this repo and they look empty or tiny, make sure Git LFS is set up (`git lfs install && git lfs pull`) — or check whether the models are hosted externally (e.g. Hugging Face Hub) instead.

## What's Inside

```
.
├── images/                     # Duo photos shown in the app (AzVer, BreKa, CharEs, RaWi)
├── models/
│   ├── lstm/
│   │   ├── lstm_sentiment_model.pt
│   │   ├── lstm_tokenizer.pkl
│   │   └── label_encoder.pkl
│   ├── mbert/                  # HuggingFace format (config.json, model.safetensors, tokenizer files)
│   ├── roberta/                # HuggingFace format
│   └── xlmr/                   # HuggingFace format
├── sentiment_app.py             # Main Streamlit app
├── upload_models.py             # Pushes trained model artifacts into models/
├── requirements.txt
├── .streamlit/                  # Streamlit config (theme, settings)
└── README.md
```

Training happens separately, in Google Colab — this repo only holds the **finished artifacts** (model weights, tokenizers, label encoders) plus the app that serves them, so it runs standalone without needing to retrain anything.

```
Reddit (PRAW scraping) → cleaning + duo-tagging + manual annotation
        → training/fine-tuning in Colab (LSTM · mBERT · RoBERTa · XLM-RoBERTa)
        → artifacts exported and uploaded into models/ (see upload_models.py)
        → sentiment_app.py serves them here
```

## The Research Behind It

PBB fandom discussions on Reddit are almost entirely **Taglish** — code-switched Tagalog-English — which most sentiment analysis work doesn't account for. This thesis set out to answer four questions:

1. How do a trained LSTM and pre-trained transformers (mBERT, RoBERTa, XLM-RoBERTa) compare on sentiment about the Big Four duos — **AzVer, BreKa, CharEs, RaWi**?
2. How does code-switching affect each model's performance?
3. What's the distribution of predicted sentiment (positive/neutral/negative) per duo?
4. Do those predicted sentiments line up with the official Big Night rankings?

**Dataset:** 141,921 Reddit entries (2,334 posts + 139,587 comments) from **r/pinoybigbrother**, collected via **PRAW** between June 7 and July 5, 2025 — the reveal of the final duos through the Big Night finale — then manually annotated for sentiment. A secondary benchmark of ~36,801 English-only Reddit comments (2019 Indian General Elections, Kaggle) was used to isolate the effect of code-switching from other platform effects. Raw data isn't included in this repository — only the trained artifacts and the app are checked in.

**Models compared:**

| Model | Type | Notes |
|---|---|---|
| **LSTM** | Trained from scratch | Bi-directional LSTM + attention, GloVe embeddings — the baseline |
| **mBERT** | Fine-tuned transformer | Multilingual BERT, pre-trained on 104 languages |
| **RoBERTa** | Fine-tuned transformer | Optimized BERT pre-training (dynamic masking, larger batches) |
| **XLM-RoBERTa** | Fine-tuned transformer | Cross-lingual transformer, built for exactly this kind of text |

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| LSTM | 0.84 | 0.84 | 0.84 | 0.85 |
| mBERT | 0.85 | 0.88 | 0.85 | 0.86 |
| RoBERTa | 0.86 | 0.86 | 0.86 | 0.86 |
| **XLM-RoBERTa** | **0.89** | **0.89** | **0.89** | **0.89** |

- Pre-trained transformers beat the from-scratch LSTM baseline, with **XLM-RoBERTa** performing best overall.
- Every model does worse on code-switched Taglish than on pure English — LSTM drops the most, XLM-RoBERTa is the most resilient.
- **Neutral** sentiment is consistently the hardest class for every model, due to mixed or ambiguous phrasing.
- All four models predicted the same ranking (RaWi 1st, BreKa 2nd, CharEs 3rd, AzVer 4th), correctly calling 3rd and 4th place — but all missed the actual 1st/2nd swap (BreKa won, RaWi came 2nd), likely because Reddit sentiment can't see the show's unlimited paid-voting mechanic.

## Retraining From Scratch

The training notebooks (one per model) run in Google Colab and aren't part of this repo — each covers data loading, tokenization, hyperparameter search, final training, and evaluation (classification report, confusion matrix, ROC-AUC). To rerun training yourself, you'd need:

- The cleaned, annotated dataset (`FINAL_cleaned_annotations_RESOLVED1.csv` — not included here for privacy/size reasons)
- A GPU runtime (Colab's free tier is enough at this dataset size)

## Limitations

- Labels come from manual annotation on a single-subreddit dataset — results may not generalize beyond r/pinoybigbrother or this specific PBB season.
- Reddit sentiment reflects a Taglish/English-speaking, terminally-online subset of fans, not the full voting population — the show's actual vote allows unlimited paid votes via app, which Reddit sentiment can't capture.
- The "Neutral" class remains the weakest across all models and is the main source of misclassification.

## The Thesis

**Predicting Audience Voting Outcomes Through Code-Switched Sentiment Analysis with Fine-Tuned BERT-Based Transformers and LSTM Networks**

- **Authors:** Kim Caryl H. Esperanza · Louella Josephine A. Ng · Lourence Anne Q. Resquid
- **Advisers:** Mary Jane C. Samonte · Madhavi Devaraj
- **Program:** BS Data Science, School of Information Technology, Mapua University
- **Year:** 2026


