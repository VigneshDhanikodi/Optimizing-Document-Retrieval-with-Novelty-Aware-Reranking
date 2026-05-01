# 🔍 Novelty-Aware Soft Label Reranker

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9D371.svg)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Environment](https://img.shields.io/badge/Environment-Colab_T4_GPU-F9AB00.svg)](https://colab.research.google.com/)

> **A fine-tuned cross-encoder reranking model designed to optimize for document novelty and soft-label relevance matching.**

## 📌 Overview

Traditional information retrieval systems often rely on hard binary labels (relevant vs. irrelevant) and prioritize exact term matching, which can lead to redundant or overly generic search results. 

This project tackles that limitation by training a **Novelty-Aware Reranker** using **soft labels**. By fine-tuning a transformer-based cross-encoder on a large-scale dataset, this model learns fine-grained relevance scores. It not only fetches highly relevant documents but also promotes content *novelty* and diversity within the top-k retrieved results.

---

## 🔑 Key Features

* **Soft-Label Distillation:** Trains on continuous relevance scores rather than binary targets, allowing the model to capture nuanced document-query semantic relationships.
* **Novelty Optimization:** Evaluates and reranks candidate documents to ensure the final output reduces redundancy.
* **Large-Scale Data Processing:** Capable of ingesting and streaming massive `.parquet` dataset splits directly from the Hugging Face hub (175M+ training examples).
* **Automated Evaluation Pipeline:** Dynamically computes precision, novelty metrics, and generates a comprehensive `novelty_results_summary.csv` for post-run analysis.

---

## 🏗️ System Architecture & Pipeline

[ Initial Retrieval (BM25 / Bi-Encoder) ] 
       │
       ▼
[ Candidate Document Pool ] + [ User Query ]
       │
       ├──> Tokenized & Concatenated (Query [SEP] Document)
       ▼
[ Soft Label Cross-Encoder (Fine-tuned Transformer) ]
       │── 1. Attention modeling across query & document
       │── 2. Soft-label loss evaluation (KL Divergence / MSE)
       └── 3. Novelty penalty application
       │
       ▼
[ Final Ranked Output (Top-K Novel & Relevant Docs) ]


---

## 📦 Dataset Details

The model leverages a massive dataset, streamed efficiently using PyTorch and Hugging Face `datasets`:

| Split | Size (Approx) | Format |
| :--- | :--- | :--- |
| **Train** | 175M records | `.parquet` |
| **Validation** | 21.4M records | `.parquet` |
| **Test** | 20.5M records | `.parquet` |

*Data preprocessing utilizes advanced tokenization schemas (`vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`) to handle complex queries.*

---

## ⚙️ Technology Stack

* **Hardware Targeting:** Nvidia T4 GPU (Optimized for Google Colab environments)
* **Deep Learning:** PyTorch
* **NLP Architecture:** Hugging Face `transformers`, `datasets`
* **Data Manipulation:** Pandas (CSV generation), NumPy
* **Output Formats:** Safetensors (`model.safetensors`) for fast, secure weight loading.

---

## 🚀 Quickstart

### 1. Environment Setup
Clone the repository and install the dependencies:

git clone [https://github.com/your-username/novelty-soft-reranker.git](https://github.com/your-username/novelty-soft-reranker.git)
cd novelty-soft-reranker
pip install transformers datasets torch pandas safetensors


### 2. Loading the Fine-Tuned Model
Once you have run the training pipeline, the model will be saved locally. You can load it for inference like this:

python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
model_path = "./soft_label_reranker/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Example query-document pair
query = "Recent advancements in edge AI"
document = "Deploying YOLO models on Jetson Orin Nano hardware without cloud dependency."

# Tokenize and score
inputs = tokenizer(query, document, return_tensors="pt", truncation=True)
with torch.no_grad():
    logits = model(**inputs).logits
    score = logits.squeeze().item()

print(f"Relevance/Novelty Score: {score}")

---

## 📁 Project Structure


novelty-soft-reranker/
├── rerankers_novelty.ipynb       # Main Colab training and evaluation notebook
├── soft_label_reranker/          # Exported model directory
│   ├── config.json               
│   ├── model.safetensors         # Fine-tuned weights
│   ├── tokenizer_config.json     
│   └── vocab.txt                 
├── data/                         # Data directory (if downloading locally)
└── novelty_results_summary.csv   # Automatically generated evaluation metrics


---

## 📊 Evaluation & Results

After execution, the notebook automatically evaluates the model on the test split. The results are aggregated and exported.

Check the **`novelty_results_summary.csv`** file in the root directory for detailed metrics on:
* Mean Reciprocal Rank (MRR)
* Normalized Discounted Cumulative Gain (nDCG)
* Novelty & Diversity indices across the Top-10 retrieved documents.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
```
