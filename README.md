# Part 3: NLP and Sequence Modeling

## Overview

This project builds a complete NLP pipeline on a customer support text classification dataset. The goal is to understand how raw text is transformed into numerical representations that machine learning models can work with, and to compare traditional vectorization approaches against sequence-based deep learning architectures. The dataset contains 1,500 customer support messages labelled across three sentiment classes: positive, negative, and neutral.

## Dataset

| Property | Value |
|---|---|
| File | `customer_support_text_classification.csv` |
| Records | 1,500 |
| Columns | ticket_id, channel, customer_message, sentiment_label, word_count, urgent_flag |
| Target | `sentiment_label` (positive / negative / neutral) |
| Avg. message length | ~12.7 words |
| Class balance | Well balanced — neutral: 524, negative: 497, positive: 479 |

## Folder Structure

```
part-3-nlp-sequence-modeling/
│
├── README.md
├── notebook.py
├── requirements.txt
└── results/
    ├── dataset_overview.png
    ├── baseline_confusion_matrices.png
    ├── baseline_comparison.csv
    ├── model_evaluation.png
    ├── model_evaluation.csv
    └── sample_predictions.txt
```



## Tasks Covered

| Task | Description |
|---|---|
| Task 1 | Dataset understanding — shape, class distribution, average text length, sample records |
| Task 2 | Text preprocessing — lowercasing, special character removal, tokenization, stopword removal |
| Task 3 | Text vectorization — Bag of Words and TF-IDF with bigrams, with explanation of why vectorization is needed |
| Task 4 | Baseline models — Multinomial Naive Bayes (BoW) and Logistic Regression (TF-IDF), evaluated with accuracy and macro F1 |
| Task 5 | Sequence model — full LSTM architecture design + MLP neural network implemented with 5 hyperparameter experiments |
| Task 6 | Reflection — RNN limitations, LSTM gating, attention mechanisms, and Transformer importance in modern NLP |

## Key Results

Both baseline models (Naive Bayes and Logistic Regression) achieve high accuracy on this well-structured dataset, with TF-IDF bigrams providing the richer feature space. The MLP neural network experiments confirm that a deeper architecture with an appropriate learning rate performs best. The full LSTM architecture design demonstrates how sequence models would handle the same task with preserved word-order information.
