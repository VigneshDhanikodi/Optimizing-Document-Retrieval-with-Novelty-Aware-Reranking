<div align="center">

# Neural Re-Ranking & Novelty-Aware Retrieval

### Transformer-based reranking with query routing, uncertainty calibration, passage compression, and soft-label fine-tuning

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FF6F00?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)
![MS MARCO](https://img.shields.io/badge/Dataset-MS%20MARCO%20v1.1-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

> **Project Description:**
> This project explores advanced reranking strategies for information retrieval by improving a transformer-based cross-encoder with four complementary novelties: query-type routing, uncertainty-aware inference, passage compression, and soft-label fine-tuning. The goal is to produce more relevant, reliable, and diverse search rankings on MS MARCO-style query-document pairs.

</div>

---

## 📋 Table of Contents

| #  | Section                                                                          |
| -- | -------------------------------------------------------------------------------- |
| 1  | [Overview](#overview)                                                            |
| 2  | [Problem Statement](#problem-statement)                                          |
| 3  | [Key Contributions](#key-contributions)                                          |
| 4  | [System Architecture](#system-architecture)                                      |
| 5  | [Dataset](#dataset)                                                              |
| 6  | [Baseline Model](#baseline-model)                                                |
| 7  | [Novelty 1: Query-Type Routing](#novelty-1-query-type-routing)                   |
| 8  | [Novelty 2: Uncertainty-Aware Reranking](#novelty-2-uncertainty-aware-reranking) |
| 9  | [Novelty 3: Passage Compression](#novelty-3-passage-compression)                 |
| 10 | [Novelty 4: Soft-Label Fine-Tuning](#novelty-4-soft-label-fine-tuning)           |
| 11 | [Evaluation Metrics](#evaluation-metrics)                                        |
| 12 | [Implementation Details](#implementation-details)                                |
| 13 | [Results Artifacts](#results-artifacts)                                          |
| 14 | [Quickstart](#quickstart)                                                        |
| 15 | [Project Structure](#project-structure)                                          |
| 16 | [Limitations](#limitations)                                                      |
| 17 | [Future Work](#future-work)                                                      |
| 18 | [License](#license)                                                              |

---

## Overview

This notebook presents a research-oriented reranking pipeline designed to improve the quality of retrieval results in a neural information retrieval setting. It starts from a strong cross-encoder baseline and then introduces four independent improvements aimed at different failure modes commonly seen in reranking systems: query heterogeneity, calibration issues, long-passage noise, and sparse binary supervision.

The core idea is simple: a single reranker is often not enough. Different query types benefit from different ranking behavior, some predictions are less certain than others, passages may contain unnecessary text, and binary labels do not fully capture graded relevance. This notebook addresses each of these issues directly.

---

## Problem Statement

Traditional reranking models often perform well on average, but they still face several limitations:

* A single model treats all queries as if they have the same structure and difficulty.
* Raw reranker scores are often poorly calibrated.
* Long passages may contain irrelevant sentences that dilute relevance signals.
* Binary labels such as `is_selected` do not reflect the full spectrum of relevance.

This project asks a practical question:
**Can reranking quality improve when the pipeline becomes query-aware, uncertainty-aware, content-aware, and label-aware at the same time?**

---

## Key Contributions

This notebook implements four novelties:

1. **Adaptive Query-Type Routing with Ensemble Rerankers**
   Queries are classified into types such as numeric, person, location, description, entity, and other, then routed to specialist rerankers.

2. **Uncertainty-Aware Reranking with Confidence Calibration**
   MC Dropout is used at inference time to estimate uncertainty, and uncertain predictions are down-weighted.

3. **Dynamic Passage Compression Before Reranking**
   A bi-encoder selects the most query-relevant sentences from long passages before they are passed to the cross-encoder.

4. **Soft Label Fine-Tuning Using Answer Overlap**
   ROUGE-L overlap between passage text and gold answers is used to generate soft relevance labels for training.

---

## System Architecture

```text
Query + Passage
      ↓
Initial Retrieval / Candidate Pairs
      ↓
Baseline Cross-Encoder Scoring
      ↓
+-----------------------------+
| Novelty Enhancements Layer   |
+-----------------------------+
      ↓
1) Query-Type Routing
2) Uncertainty Calibration
3) Passage Compression
4) Soft-Label Fine-Tuning
      ↓
Final Relevance Scores
      ↓
Ranking Output + Metrics + Plots
```

The notebook is organized as a modular experimental pipeline so that each novelty can be evaluated independently and compared fairly against the baseline.

---

## Dataset

| Feature           | Description                                  |
| ----------------- | -------------------------------------------- |
| Dataset           | MS MARCO v1.1                                |
| Task              | Passage reranking / information retrieval    |
| Input             | Query–passage pairs                          |
| Supervision       | Binary labels + answer text                  |
| Main split used   | Train / Validation                           |
| Evaluation subset | Validation subset for faster experimentation |

MS MARCO is a strong benchmark for reranking because it contains realistic search queries and noisy candidate passages. The notebook also leverages the `answers` field to build soft relevance targets.

---

## Baseline Model

The baseline reranker is:

```python
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This model is used as the default cross-encoder for scoring query-passage pairs. It serves as the reference point for all comparisons in the notebook.

The baseline pipeline:

* forms query-passage pairs,
* scores them using the cross-encoder,
* ranks candidates by the predicted relevance score,
* evaluates ranking quality with standard IR metrics.

---

## Novelty 1: Query-Type Routing

Not all queries should be treated equally. A “who is” question behaves differently from a “how many” query or a location-based query. This novelty introduces a lightweight query classifier that assigns each query to a type.

### Query types supported

* numeric
* person
* location
* description
* entity
* other

### Routing logic

Each query type is mapped to a specialist reranker:

* some types use the general model,
* some are routed to a larger model,
* some are routed to an ELECTRA-based reranker.

### Why this matters

This allows the reranking system to adapt to query structure instead of using a single uniform scoring strategy for everything.

### Extra enhancement

The notebook also learns a **weighted ensemble** across multiple rerankers using logistic regression on validation scores, which adds a second layer of improvement beyond hard routing.

---

## Novelty 2: Uncertainty-Aware Reranking

Cross-encoder scores are often overconfident. This novelty adds confidence estimation using **MC Dropout**.

### How it works

* dropout layers are kept active during inference,
* the model is run multiple times,
* the mean score becomes the prediction,
* the standard deviation becomes an uncertainty estimate.

### Confidence adjustment

High-uncertainty items are penalized by subtracting uncertainty from their rerank score.

### Why this matters

This makes the reranker more conservative when the model is unsure, which can improve ranking stability and calibration.

### Additional metric

The notebook computes:

* **Expected Calibration Error (ECE)**

and compares baseline calibration against MC Dropout calibration.

---

## Novelty 3: Passage Compression

Long passages often contain noisy or irrelevant sentences. This novelty compresses each passage before reranking.

### Method

* split the passage into sentences,
* encode the query and each sentence using a sentence-transformer,
* compute cosine similarity,
* keep the top relevant sentences only.

### Output

The compressed passage is shorter but more query-focused.

### Why this matters

Cross-encoders are sensitive to input noise. Reducing irrelevant content can help the reranker focus on the most useful evidence while also lowering inference cost.

### Experiment support

The notebook includes an ablation study over different numbers of retained sentences.

---

## Novelty 4: Soft-Label Fine-Tuning

Binary labels are too coarse for ranking tasks. This novelty converts the supervision signal into a **soft relevance target**.

### Method

* compute ROUGE-L overlap between the passage and the gold answer(s),
* blend that score with the original binary label,
* fine-tune the reranker using a KL-divergence-based objective.

### Why this matters

This approach captures graded relevance instead of forcing every example into a strict 0/1 label. It is especially useful when some passages are partially relevant but not fully selected.

### Resulting effect

The model becomes more sensitive to degrees of relevance, which is more aligned with ranking behavior.

---

## Evaluation Metrics

The notebook evaluates all variants with ranking and calibration metrics.

| Metric      | Meaning                                           |
| ----------- | ------------------------------------------------- |
| Precision@K | Fraction of relevant items in the top-k results   |
| MRR         | How early the first relevant result appears       |
| NDCG@K      | Ranking quality with position-aware gain          |
| ECE         | Calibration error between confidence and accuracy |

These metrics provide a balanced view of retrieval quality, not just top-score accuracy.

---

## Implementation Details

### Core libraries

* `torch`
* `transformers`
* `sentence-transformers`
* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`
* `rouge-score`

### Environment

* Google Colab
* GPU runtime recommended
* T4 GPU used for experiments

### Experimental structure

The notebook is divided into:

* installation
* dataset setup
* baseline ranking
* novelty experiments
* evaluation
* visualization
* model saving

---

## Results Artifacts

The notebook generates and saves the following outputs:

* `calibration_plot.png`
  Reliability diagram for baseline vs calibrated reranker

* `novelty_comparison.png`
  Bar chart comparing baseline and novel methods

* `novelty_results_summary.csv`
  Summary table of all experiment metrics

* `soft_label_reranker/`
  Saved fine-tuned model and tokenizer

These artifacts make the notebook easy to reproduce, present, and extend.

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rerankers-novelty.git
cd rerankers-novelty
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

```bash
jupyter notebook
```

### 4. Run in Colab

If you use Colab, switch to:

* **Runtime → Change runtime type → GPU**

---

## Project Structure

```text
rerankers-novelty/
├── rerankers_novelty.ipynb
├── soft_label_reranker/
├── calibration_plot.png
├── novelty_comparison.png
├── novelty_results_summary.csv
├── requirements.txt
└── README.md
```

---

## Limitations

* The notebook is computationally expensive for large-scale reranking.
* MC Dropout increases inference cost because it requires multiple forward passes.
* Passage compression depends on sentence segmentation quality.
* Soft-label fine-tuning requires answer text to be available.
* Query-type routing currently uses lightweight heuristics rather than a fully trained classifier.

---

## Future Work

Possible extensions include:

* replacing rule-based routing with a learned query classifier,
* using larger rerankers or late-interaction models,
* adding hybrid retrieval with BM25 + dense embeddings,
* evaluating on more benchmarks beyond MS MARCO,
* extending the calibration method to temperature scaling or conformal prediction,
* integrating the reranker into a full RAG pipeline.

---

## License

MIT License

---

## Summary

This notebook is a strong foundation for a research-style reranking project. It combines:

* strong baseline reranking,
* query-aware routing,
* uncertainty calibration,
* query-focused compression,
* and soft-label supervision.

That makes it suitable for:

* portfolio presentation,
* research experimentation,
* and further development into a full ranking or RAG system.
