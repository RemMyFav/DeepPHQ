# DeepPHQ

**Predicting depression severity (PHQ-8) from clinical interview transcripts using Deep Learning**

This repository contains the code and report materials for the CS 7643 Deep Learning course project at Georgia Tech.

---

## 🧠 Overview
**Goal:**  
To predict participants’ PHQ-8 depression scores using transcribed text from the [DAIC-WOZ dataset](https://dcapswoz.ict.usc.edu/wwwdaicwoz/).  

**Approach:**  
We experiment with multiple neural architectures—**LSTM**, **RNN**, and **Transformer**—to explore how linguistic patterns relate to psychological assessments.  
The task is framed as both regression and classification to evaluate model robustness.

---

## 🧩 Methods
- Preprocessing of interview transcripts (tokenization, cleaning, truncation)
- Embedding initialization with pretrained GloVe/BERT vectors
- Model training with dropout + Adam optimizer  
- Evaluation metrics: MSE, MAE (regression) and Accuracy, F1 (classification)

---

## 📊 Dataset
- **DAIC-WOZ** clinical dialogue dataset  
- Each participant’s transcript paired with a **PHQ-8 score**  
- Only text modality used (no audio/video)

---

## 🧪 Results & Analysis
Comparative experiments between LSTM and Transformer models show that contextual embeddings significantly improve PHQ prediction accuracy.  
Detailed results and plots will be included in the final report.

---

## 👥 Team
- **Haike Yu** –
- **Mun Sun Bin** 
- **Edbert Wang**
- **Chengyuan Yao**
  
---

## ⚙️ Environment
- Python 3.10  
- PyTorch 2.x  
- HuggingFace Transformers  
- NumPy / Pandas / Matplotlib  

---

## 📜 License
This project is released under the [MIT License](LICENSE).

---
