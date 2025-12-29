# DeepPHQ  
**Evaluating Textual Granularity for PHQ-8 Depression Severity Prediction**

Predicting depression severity (PHQ-8) from clinical interview transcripts using deep learning, with a focus on how **input granularity and model architecture influence robustness and behavior**.

---

## 🧠 Motivation

Automatic depression assessment from language is a promising but fragile area of clinical NLP.  
While many prior works report performance improvements, it remains unclear **what linguistic structure models actually rely on**.

This project is motivated by a central question:

> **Does increasing textual context (word → sentence → dialogue) meaningfully improve PHQ-8 prediction, or are models largely invariant to input granularity?**

Rather than optimizing a single model for performance, this work emphasizes **controlled experimental comparison** to better understand model behavior in sensitive mental health settings.

---

## 🔍 Research Question

**How does textual granularity affect depression severity prediction across different neural architectures?**

Specifically, this project examines:
- Word-level, sentence-level, and dialogue-level text representations
- Architecture-dependent sensitivity to contextual scope
- Stability and robustness under reduced or fragmented input context

---

## 📊 Dataset

- **DAIC-WOZ** clinical interview dataset  
- Semi-structured interviews between participants and a virtual interviewer  
- Each participant annotated with a **PHQ-8 score (0–24)**  
- **Text-only modality** (audio and video intentionally excluded)

Only **participant speech** is retained to isolate depressive linguistic signals.

---

## 🧩 Input Construction

Each interview is transformed into **three parallel input representations**:

- **Word-level**: randomly sampled individual tokens  
- **Sentence-level**: randomly sampled full sentences  
- **Dialogue-level**: contiguous multi-sentence segments (~512 tokens)

All inputs are length-matched to avoid confounding effects from sequence length.  
Balanced sampling is applied to mitigate PHQ-8 label imbalance.

---

## 🧠 Models Evaluated

Four neural architectures commonly used in clinical NLP are evaluated under identical conditions:

| Model | Purpose |
|------|--------|
| **Transformer (CORAL)** | Long-range context modeling with ordinal regression |
| **TextCNN** | Local lexical and phrase-level pattern extraction |
| **LSTM** | Sequential baseline sensitive to ordering |
| **Hierarchical Attention RNN (HAN)** | Explicit word–sentence–dialogue structure |

Each model is tuned on dialogue-level inputs and evaluated unchanged across all granularities.

---

## 🧪 Training & Evaluation

- Framework: **PyTorch**
- Optimizer: **AdamW**, gradient clipping
- Evaluation metric: **Participant-level MSE**
- PHQ-8 modeled using **CORAL ordinal regression** for the Transformer

---

## 📈 Key Findings

- Performance differences across input granularities are **surprisingly small**
- Dialogue-level context does not consistently outperform sentence- or word-level inputs
- Transformers and HAN models remain robust under reduced context
- LSTMs show greater sensitivity to input granularity

**Main takeaway:**

> Within DAIC-WOZ, textual granularity alone does not fundamentally determine PHQ-8 prediction performance.

---

## 🧠 What This Project Emphasizes

- Controlled experimental design over metric chasing  
- Architecture-agnostic comparison  
- Understanding *why* models behave as they do in clinical NLP  
- Awareness of dataset bias and evaluation limitations  

---

## 🔗 Resources

- **Code**: https://github.com/RemMyFav/DeepPHQ  
- **Report**: Included in this repository

---

## 📜 License

MIT License
