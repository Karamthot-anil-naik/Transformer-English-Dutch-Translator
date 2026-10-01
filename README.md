# Transformer English → Dutch Translator

A **Neural Machine Translation (NMT)** project that translates English sentences into Dutch using a **Transformer-based Encoder-Decoder architecture**.

This project demonstrates the fundamental concepts behind modern Transformer models, including **tokenization, embeddings, positional encoding, self-attention, multi-head attention, encoder-decoder attention, and sequence-to-sequence translation**.

---

## 📌 Project Overview

The goal of this project is to build a Transformer-based translation system that learns to translate English sentences into Dutch.

The model takes an English sentence as input and generates the corresponding Dutch translation.

```text
English Sentence
       ↓
   Tokenization
       ↓
   Embedding
       ↓
Positional Encoding
       ↓
Transformer Encoder
       ↓
Transformer Decoder
       ↓
Linear + Softmax
       ↓
Dutch Translation
```

---

## 🚀 Features

* English → Dutch translation
* Transformer Encoder-Decoder architecture
* Tokenization and vocabulary processing
* Word embeddings
* Positional encoding
* Multi-head self-attention
* Encoder-decoder attention
* Feed-forward neural networks
* Sequence-to-sequence learning
* Training and validation
* Sentence-level translation inference

---

## 🧠 Transformer Architecture

The project follows the standard Transformer architecture introduced for sequence-to-sequence tasks.

```text
                    ENGLISH INPUT
                         │
                         ▼
                   Tokenization
                         │
                         ▼
                  Input Embedding
                         │
                         ▼
                Positional Encoding
                         │
                         ▼
              ┌─────────────────────┐
              │   TRANSFORMER       │
              │      ENCODER        │
              │                     │
              │  Self-Attention     │
              │        ↓            │
              │  Feed Forward       │
              │        ↓            │
              │  Add & Norm         │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   TRANSFORMER       │
              │      DECODER        │
              │                     │
              │ Masked Self-Attn.   │
              │        ↓            │
              │ Cross Attention     │
              │        ↓            │
              │ Feed Forward        │
              │        ↓            │
              │  Add & Norm         │
              └──────────┬──────────┘
                         │
                         ▼
                  Linear Layer
                         │
                         ▼
                      Softmax
                         │
                         ▼
                 DUTCH OUTPUT
```

---

## 🔄 Working Process

### 1. Data Preparation

The English-Dutch sentence pairs are loaded and prepared for training.

Each training example contains:

```text
English Sentence → Dutch Sentence
```

The dataset is processed into sequences that can be understood by the neural network.

---

### 2. Tokenization

The input sentences are converted into individual tokens.

Special tokens can be used to represent different sequence conditions:

```text
<sos> → Start of Sentence
<eos> → End of Sentence
<pad> → Padding
<unk> → Unknown Token
```

Example:

```text
English:
I love mathematics

Tokens:
I | love | mathematics
```

---

### 3. Embedding

Each token is converted into a numerical vector using an embedding layer.

```text
Token
  ↓
Embedding Layer
  ↓
Numerical Vector
```

These embeddings allow the Transformer to work with numerical representations of words.

---

### 4. Positional Encoding

Unlike RNNs, Transformers do not inherently process tokens according to their sequential position.

Therefore, positional information is added to the token embeddings.

```text
Token Embedding
       +
Positional Encoding
       ↓
Transformer Input
```

This allows the model to understand the position of each word within a sentence.

---

### 5. Transformer Encoder

The encoder processes the complete English input sequence.

The encoder contains:

* Multi-Head Self-Attention
* Feed-Forward Neural Network
* Residual Connections
* Layer Normalization

Self-attention allows the model to understand relationships between different words in the input sentence.

---

### 6. Transformer Decoder

The decoder generates the Dutch translation based on:

* Previously generated Dutch tokens
* Information from the encoder

The decoder uses:

* Masked Self-Attention
* Encoder-Decoder Cross Attention
* Feed-Forward Neural Network
* Residual Connections
* Layer Normalization

---

### 7. Output Prediction

The decoder output is passed through a linear layer followed by a probability distribution over the Dutch vocabulary.

```text
Decoder Output
      ↓
Linear Layer
      ↓
Softmax
      ↓
Predicted Dutch Token
```

The process continues until the model generates the end-of-sentence token.

---

# 🔤 Translation Examples

The trained model can generate translations such as:

| English Input         | Dutch Translation    |
| --------------------- | -------------------- |
| I love mathematics    | Ik hou van wiskunde  |
| calculus is great     | Calculus is geweldig |
| mathematics is useful | Wiskunde is nuttig   |
| I love calculus       | Ik hou van calculus  |
| mathematics is great  | Wiskunde is geweldig |

### Example 1

```text
English:
I love mathematics

Dutch:
Ik hou van wiskunde
```

### Example 2

```text
English:
calculus is great

Dutch:
Calculus is geweldig
```

### Example 3

```text
English:
mathematics is useful

Dutch:
Wiskunde is nuttig
```

### Example 4

```text
English:
I love calculus

Dutch:
Ik hou van calculus
```

### Example 5

```text
English:
mathematics is great

Dutch:
Wiskunde is geweldig
```

---

## 🛠️ Technologies Used

* **Python**
* **PyTorch**
* **Natural Language Processing (NLP)**
* **Deep Learning**
* **Transformer Architecture**
* **Sequence-to-Sequence Learning**
* **Tokenization**
* **NumPy**
* **Matplotlib**
* **Jupyter Notebook**

---

> Update the structure if your actual project uses different filenames or folders.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Karamthot-anil-naik/transformer-english-dutch-translator.git
```

Navigate to the project:

```bash
cd transformer-english-dutch-translator
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

If the project uses Jupyter Notebook:

```bash
jupyter notebook
```

Open the Transformer notebook and run the cells sequentially.

For a Python training script:

```bash
python src/train.py
```

---

## 📊 Results

The project successfully demonstrates English-to-Dutch translation using the trained Transformer model.

Sample translations:

```text
I love mathematics
        ↓
Ik hou van wiskunde

mathematics is useful
        ↓
Wiskunde is nuttig

I love calculus
        ↓
Ik hou van calculus
```

### Evaluation

The model can be evaluated using metrics such as:

* Training Loss
* Validation Loss
* BLEU Score
* Translation Examples

> Add numerical metric values only when they are actually obtained from your model.

---

## 📚 Key Concepts Demonstrated

This project provides practical exposure to:

* Natural Language Processing
* Neural Machine Translation
* Sequence-to-Sequence Learning
* Transformer Architecture
* Encoder-Decoder Architecture
* Self-Attention
* Multi-Head Attention
* Masked Attention
* Cross Attention
* Positional Encoding
* Word Embeddings
* Tokenization
* Vocabulary Building
* Teacher Forcing
* Model Training
* Model Inference

---

## 🎯 Learning Outcomes

Through this project, I gained practical understanding of how **Transformer architectures are applied to Neural Machine Translation**.

The project helped demonstrate the complete workflow from:

```text
Raw Text
   ↓
Tokenization
   ↓
Vocabulary
   ↓
Embeddings
   ↓
Positional Encoding
   ↓
Transformer Encoder
   ↓
Transformer Decoder
   ↓
Prediction
   ↓
Dutch Translation
```

---

## 🔮 Future Improvements

* Implement Beam Search decoding
* Improve tokenization using subword tokenization
* Train using a larger English-Dutch parallel dataset
* Add BLEU score evaluation
* Add a Streamlit user interface
* Deploy the translation application
* Support additional language pairs
* Experiment with pretrained Transformer models
* Improve translation quality through hyperparameter tuning

---

## 💻 GitHub Setup

Initialize Git:

```bash
git init
```

Add project files:

```bash
git add .
```

Create the first commit:

```bash
git commit -m "Initial commit - Transformer English Dutch Translator"
```

Rename the branch:

```bash
git branch -M main
```

Connect the GitHub repository:

```bash
git remote add origin https://github.com/Karamthot-anil-naik/transformer-english-dutch-translator.git
```

Push the project:

```bash
git push -u origin main
```

---

## 👨‍💻 Author

**Karamthot Anil Naik**

B.Tech Computer Science Engineering

GitHub: [Karamthot-anil-naik](https://github.com/Karamthot-anil-naik)

---

## ⭐ Conclusion

The **Transformer English → Dutch Translator** demonstrates the application of Transformer-based deep learning to Neural Machine Translation.

The project covers the complete translation pipeline, including **tokenization, embeddings, positional encoding, attention mechanisms, encoder-decoder processing, training, and inference**.

It provides a practical implementation of core Transformer concepts used in modern NLP and language translation systems.
