# 📩 Spam Email Classification

This project is a web application that classifies whether an email is **Spam** or **Not Spam** using basic feature extraction and a machine learning model.

---

##  Project Definition

The goal is to build a lightweight and interpretable spam classifier that:
- Accepts **raw email text** as input
- Extracts features based on **word frequency**, **character usage**, and **capital letter patterns**
- Predicts whether the email is **Spam** or **Not Spam**
- Shows an explanation of how the model made its prediction using **SHAP bar plots**

---

##  Dataset Used

- **Dataset**: [UCI Spambase Dataset](https://archive.ics.uci.edu/ml/datasets/spambase)  
  It contains 4,601 email messages labeled as spam or not spam.
- Features are derived from:
  - Frequency of 48 specific words (e.g., `money`, `email`, `remove`, etc.)
  - Frequency of 6 special characters (e.g., `!`, `$`)
  - 3 statistics about capital letter usage (average length, longest run, total count)

---

##  Model Used

- **Algorithm**: [XGBoost Classifier]  
- **Preprocessing**:
  - Custom feature extraction (not bag-of-words or deep embeddings)
  - StandardScaler for normalization

---

##  Website Link

 **Live App**: [https://ApurvaKondekar2004-Spam-Email-Classification.streamlit.app](https://hrtoq7soeavzuswlgcysjk.streamlit.app/)  

---

##  Example

**Input Email:**
Receive free internet by clicking the link!!!


**Prediction:**
- 🛑 Spam  
- **Confidence:** 92.4%

**SHAP Explanation:**
- Features like `free`, `internet`, and `!` strongly influenced the model towards Spam.

