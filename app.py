import streamlit as st
import pickle
import numpy as np
import os
from lime.lime_text import LimeTextExplainer

# st.write("Current working directory:", os.getcwd())
# st.write("Files present:", os.listdir())
# ========= Load Model ========= #
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
model = pickle.load(open("model/rf_model.pkl", "rb"))

# Wrapper for LIME → converts text before prediction
def predict_proba_wrapper(texts):
    X = tfidf.transform(texts)
    return model.predict_proba(X)

class_names = ['REAL', 'FAKE']
explainer = LimeTextExplainer(class_names=class_names)

st.set_page_config(page_title="Fake News Classifier", layout="wide")

st.title("📰 Fake News Detection App")
st.write("Enter a news article below, and the model will classify it.")

text = st.text_area("Paste news text here:", height=200)

if st.button("Predict"):
    if text.strip() == "":
        st.warning("⚠ Please enter some text first.")
    else:
        # Predict
        X = tfidf.transform([text])
        pred_proba = model.predict_proba(X)[0]
        pred = np.argmax(pred_proba)

        st.subheader(f"🔍 Prediction: **{class_names[pred]}**")
        st.write(f"Confidence: `{pred_proba[pred]*100:.2f}%`")

        # ============ LIME Explanation ============ #
        exp = explainer.explain_instance(text, predict_proba_wrapper, num_features=10)

        st.subheader("🧠 Why did the model predict that?")
        st.write(exp.as_list())

        st.subheader("🔎 Word Importance Visualization (LIME)")
        st.components.v1.html(exp.as_html(), height=500, scrolling=True)
