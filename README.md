# rnn-sentiment-analysis-imdb
Sentiment analysis of 50K IMDB movie reviews using an RNN in PyTorch, with NLP text pre-processing and TF-IDF features. Achieves 85.16% test accuracy.
# 🎬 IMDB Movie Review Sentiment Analysis using RNN

A Natural Language Processing (NLP) project that classifies IMDB movie reviews as **positive** or **negative** using a **Recurrent Neural Network (RNN)** built with PyTorch.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Limitations and Future Work](#-limitations-and-future-work)
- [Author](#-author)

---

## 📌 Overview

Understanding customer opinion from text is one of the most common tasks in NLP. In this project, raw movie reviews are cleaned, converted into numerical features using **TF-IDF**, and passed to an RNN that predicts the sentiment of each review.

**Key highlights**

- Complete text pre-processing pipeline (lowercase, URL / HTML / punctuation removal, stopwords, stemming)
- TF-IDF vectorization with 5,000 features
- RNN classifier implemented in PyTorch with `DataLoader` mini-batch training
- **85.16%** accuracy on unseen test data

---

## 📂 Dataset

**IMDB Dataset of 50K Movie Reviews** (available on Kaggle), saved as `IMDB Dataset.csv`.

| Property | Value |
|---|---|
| Total reviews | 50,000 |
| Reviews after removing duplicates | 49,582 |
| Columns | `review` (text), `sentiment` (positive / negative) |
| Missing values | 0 |
| Train / Test split | 39,665 / 9,917 (80% / 20%) |

---

## 🔄 Project Workflow

```
Load IMDB Data
      ↓
Check Nulls and Remove Duplicates
      ↓
Text Pre-processing
      ↓
Label Encoding
      ↓
TF-IDF Vectorization
      ↓
Train/Test Split
      ↓
PyTorch DataLoaders
      ↓
RNN Training
      ↓
Evaluation
```

**Text pre-processing steps**

| Step | Purpose |
|---|---|
| Lowercasing | Treat `Movie` and `movie` as the same word |
| Remove URLs | Links carry no sentiment |
| Remove punctuation | Keep only letters, numbers and spaces |
| Remove HTML tags | Clean tags such as `<br />` |
| Remove stopwords | Drop very common words (the, is, and ...) |
| Stemming (Porter) | Reduce words to their root (`running → run`) |
| Label encoding | `negative → 0`, `positive → 1` |
| TF-IDF (5,000 features) | Convert text into numeric vectors |

---

## 🧠 Model Architecture

| Component | Details |
|---|---|
| Input size | 5,000 (TF-IDF features) |
| RNN layer | `nn.RNN`, 128 hidden units, 1 layer |
| Output layer | `nn.Linear(128, 1)` followed by Sigmoid |
| Loss function | Binary Cross-Entropy (`BCELoss`) |
| Optimizer | Adam |
| Batch size | 64 |
| Epochs | 10 |

---

## 📊 Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **85.16%** |

The model correctly classifies about 85 out of every 100 unseen reviews.

---

## 🛠 Tech Stack

- **Language:** Python 3
- **Data handling:** Pandas
- **NLP:** NLTK, Regular Expressions, TF-IDF (scikit-learn)
- **Deep learning:** PyTorch
- **Visualization:** Matplotlib
- **Environment:** Jupyter Notebook

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/alhinarabia/imdb-sentiment-analysis-rnn.git
cd imdb-sentiment-analysis-rnn
```

### 2. Install the dependencies
```bash
pip install pandas numpy matplotlib nltk scikit-learn torch jupyter
```

### 3. Add the dataset
Download **IMDB Dataset.csv** from Kaggle and place it in the project root folder (next to the notebook).

### 4. Run the notebook
```bash
jupyter notebook RNN_for_sentimentanalysis.ipynb
```
The notebook downloads the required NLTK resources (`punkt`, `punkt_tab`, `stopwords`) automatically on first run.

---

## 📁 Project Structure

```
.
├── RNN_for_sentimentanalysis.ipynb   # Complete analysis, training and evaluation
├── IMDB Dataset.csv                  # Dataset (download separately)
└── README.md
```

---

## 🔭 Limitations and Future Work

- Each review is passed to the RNN as one TF-IDF vector (a single time step), so the model does not read the words in order. Using word embeddings with real word sequences would let the RNN use its full strength.
- Compare with **LSTM** and **GRU** models.
- Improve stopword removal so that it works on whole words only.
- Add a validation set, learning curves and hyperparameter tuning.
- Deploy the model as a simple web app (Streamlit or Flask).

---

## 👤 Author

**Alhin Arabia**

[![GitHub](https://img.shields.io/badge/GitHub-alhinarabia-181717?logo=github)](https://github.com/alhinarabia)

If you found this project useful, consider giving it a ⭐ on GitHub!
