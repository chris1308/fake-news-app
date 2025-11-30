import streamlit as st
import pickle
import numpy as np
from lime.lime_text import LimeTextExplainer

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
        # ============ Prediction ============ #
        X = tfidf.transform([text])
        pred_proba = model.predict_proba(X)[0]
        pred_class = class_names[np.argmax(pred_proba)]
        confidence = pred_proba[np.argmax(pred_proba)] * 100

        st.markdown(f"""
        ### 🔍 Model Prediction  
        **Result:** <span style="font-size:26px;font-weight:bold;color:#4CAF50">{pred_class}</span>  
        **Confidence Score:** `{confidence:.2f}%`
        """, unsafe_allow_html=True)

        # Horizontal probability bar
        st.progress(float(confidence/100))

        st.write("---")

        # ============ Explainability Section ============ #
        st.subheader("🧠 Why did the model predict this? (LIME Analysis)")
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(text, predict_proba_wrapper, num_features=10)

        # Display important words as colored indicators
        weights = exp.as_list()
        st.markdown("### 🔑 Top contributing words")

        for word, weight in weights:
            color = "#4CAF50" if weight > 0 else "#FF4C4C"
            st.markdown(
                f"""
                <div style="margin:6px 0;padding:8px;border-radius:6px;background-color:{color}20;">
                    <b>{word}</b> — <span style="color:{color};font-weight:bold">{weight:.4f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("---")

        col1, col2 = st.columns([1.3, 1])

        # Column 1: LIME HTML Chart
        with col1:
            st.markdown("### 🔎 Word Importance Chart")
            st.components.v1.html(exp.as_html(), height=500, scrolling=True)

        # Column 2: Raw weigts + summary
        with col2:
            pos_words = [w for w in weights if w[1] > 0]
            neg_words = [w for w in weights if w[1] < 0]

            st.markdown("""
            ### 📌 Quick Summary
            - 🟢 Positive evidence → pushes classification toward **REAL**
            - 🔴 Negative evidence → pushes classification toward **FAKE**
            """)

            st.markdown(f"**🟢 Supporting words:** `{len(pos_words)}`") 
            st.markdown(f"**🔴 Opposing words:** `{len(neg_words)}`")
