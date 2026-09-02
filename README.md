# Custom Sparse Mixture-of-Experts (MoE) for Nuanced Sentiment Analysis on Persian Restaurant Reviews

## 📄 Abstract

State-of-the-art Natural Language Processing (NLP) often demands massive computational resources. Mixture-of-Experts (MoE) architectures address this by activating specialized sub-networks conditionally per token, thereby keeping computational costs constant while scaling capacity.

This repository presents a comprehensive, from-scratch implementation of a **Sparse Mixture-of-Experts (MoE)** network in PyTorch for Persian sentiment analysis. The project is built on a dataset of **50,000 Persian restaurant reviews** collected from SnappFood using a custom data pipeline (`DataCollector.py`). In addition to the model implementation, the project now includes a **Flask-based local web interface** that allows users to interact with the trained model through a modern browser UI.

---

## 1. Introduction & Motivation

Persian sentiment analysis—particularly in food-delivery contexts—poses significant linguistic challenges. Reviews frequently contain colloquial expressions, mixed sentiment targets, informal spelling, and high syntactic diversity.

Instead of processing every token through identical weights, this project adopts a **Sparse MoE architecture**, where a gating network dynamically routes each token to specialized Feed-Forward Neural Networks (Experts). This reduces parameter interference while enabling semantic specialization without increasing computation per token.

---

## 2. Dataset & Preprocessing Pipeline

The empirical foundation of this project is a dataset containing **70,000 Persian text reviews** collected from SnappFood.

### 2.1 Data Collection

`DataCollector.py` automates the collection process by handling repeated requests, rate limiting, and converting raw JSON responses into a structured dataset.

### 2.2 Text Normalization

Persian text requires careful preprocessing due to RTL formatting and orthographic variations.

- **Right-to-Left Rendering:** `python-bidi` and `arabic_reshaper` ensure correct visual rendering during debugging and logging.
- **Cleaning:** URLs, excessive punctuation, and non-essential noise are removed while preserving meaningful emojis and sentiment-bearing tokens.
- **Token Preparation:** Text is normalized before entering the training pipeline.

---

## 3. Architecture & Methodology

The implemented model consists of four primary components:

- Embedding Layer
- Custom Gating Network
- Parallel Expert Networks
- Classification Head

![MoE Architecture Design](Mixture of Experts.jpg)

### 3.1 Sparse MoE Layer

For an input representation <math value="x"/>, the MoE output is:

<math block value="y=\\sum_{i=1}^{N}G(x)_iE_i(x)"/>

where:

- <math value="N"/> is the number of experts.
- <math value="E_i(x)"/> is the output of the i-th expert.
- <math value="G(x)_i"/> is the routing weight assigned by the gating network.

To enforce sparsity, only the **Top-k experts** are activated:

<math block value="G(x)=\\text{Softmax}(\\text{TopK}(W_gx,k))"/>

---

### 3.2 Load-Balancing Auxiliary Loss

A common failure mode of MoE systems is **router collapse**, where only a few experts receive nearly all tokens.

To prevent this, an auxiliary load-balancing loss is combined with the classification loss:

<math block value="L_{total}=L_{CE}+\\alpha L_{aux}"/>

The auxiliary objective encourages more uniform expert utilization across each batch, improving both training stability and expert specialization.

---

## 4. Interpretability & Expert Specialization

One distinguishing feature of this implementation is its ability to inspect **expert specialization**.

During validation, the `discover_expert_specialties` routine records which tokens are routed to each expert, allowing qualitative analysis of the semantic roles learned by individual experts.

---

## 5. Local Web Interface

The project includes a **Flask-powered local web application** that provides an intuitive interface for testing the sentiment analysis model.

### Features

- Modern browser-based interface
- Persian RTL support
- Local inference without Streamlit
- Simple deployment with a single command

After launching the application, users can enter Persian restaurant reviews directly from the browser and receive sentiment predictions from the trained MoE model.

---

## 6. Repository Structure

```text
SnappFood-MoE-Sentiment-Analysis/
│
├── app.py                  # Flask local web application
├── DataCollector.py        # SnappFood data collection pipeline
├── MainNotebook.ipynb      # Training and experimentation notebook
├── requirements.txt
├── templates/              # HTML templates for the Flask interface
├── static/                 # CSS, JavaScript, and assets (if present)
├── Mixture of Experts.jpg
└── README.md
```

---

## 7. Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/sadraaa444-glitch/SnappFood-MoE-Sentiment-Analysis.git
cd SnappFood-MoE-Sentiment-Analysis
pip install -r requirements.txt
```

---

## 8. Running the Project

### Step 1 — Prepare the Dataset (Optional)

If the dataset has not already been created, run:

```bash
python DataCollector.py
```

### Step 2 — Launch the Local Web Application

Start the Flask application with:

```bash
python app.py
```

Then open the displayed local address (typically `http://127.0.0.1:5000`) in your browser.

---

## 9. Training

The complete training pipeline—including preprocessing, model implementation, custom MoE layers, training loops, and routing analysis—is available in:

`MainNotebook.ipynb`

---

## 10. Technologies Used

- Python
- PyTorch
- Flask
- HTML/CSS
- python-bidi
- arabic_reshaper
- Jupyter Notebook

---

*Developed by Sadra Amiri*
