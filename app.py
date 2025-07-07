import streamlit as st
import numpy as np
import pickle
import re
import shap
import matplotlib.pyplot as plt


with open('xgb_best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

word_features = ['make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet',
                 'order', 'mail', 'receive', 'will', 'people', 'report', 'addresses', 'free',
                 'business', 'email', 'you', 'credit', 'your', 'font', '000', 'money', 'hp',
                 'hpl', 'george', '650', 'lab', 'labs', 'telnet', '857', 'data', '415', '85',
                 'technology', '1999', 'parts', 'pm', 'direct', 'cs', 'meeting', 'original',
                 'project', 're', 'edu', 'table', 'conference']

char_features = [';', '(', '[', '!', '$', '#']
extra_features = ["avg_cap_run", "longest_cap_run", "total_cap_run"]
feature_names = word_features + char_features + extra_features

def extract_features(email_text):
    email_text_lower = email_text.lower()
    words = re.findall(r'\b\w+\b', email_text_lower)
    n_words = len(words)

    word_freqs = [100 * words.count(word) / n_words if n_words > 0 else 0 for word in word_features]
    total_chars = len(email_text_lower)
    char_freqs = [100 * email_text_lower.count(char) / total_chars if total_chars > 0 else 0 for char in char_features]

    cap_runs = re.findall(r'[A-Z]+', email_text)
    run_lengths = [len(run) for run in cap_runs]
    avg = np.mean(run_lengths) if run_lengths else 0
    longest = max(run_lengths) if run_lengths else 0
    total = sum(run_lengths) if run_lengths else 0

    return np.array(word_freqs + char_freqs + [avg, longest, total]).reshape(1, -1)

# Streamlit App
st.set_page_config(page_title="Spam Email Classifier", layout="centered")
st.title("📩 Spam Email Classifier")

email_text = st.text_area(" Enter email text to classify:")


if st.button("🔍 Predict"):
    if email_text.strip() == "":
        st.warning("Please enter some email text.")
    else:
        # Feature processing and prediction
        features = extract_features(email_text)
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0]

        st.markdown(f"### 🧠 Prediction: {'🛑 **Spam**' if prediction == 1 else '✅ **Not Spam**'}")
        st.markdown(f"**Confidence:** {max(proba) * 100:.2f}%")

        # SHAP Explanation
        st.subheader("📊 Feature Impact Explanation (SHAP)")
        st.markdown("""
            🔎 **SHAP Explanation Note**  

            - 🔵 Blue bars: Features that push the prediction **toward Not Spam**
            - 🔴 Red bars: Features that push it **toward Spam**
            - The longer the bar, the more influence that feature had on the final result.

                """)

        explainer = shap.Explainer(model)
        shap_values = explainer(scaled)
        shap_values.feature_names = feature_names
        
        st.subheader("📊 Feature Impact Explanation (SHAP Bar Plot)")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)
