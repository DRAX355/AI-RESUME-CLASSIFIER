# app.py

import streamlit as st
import joblib
import re
import fitz  # PyMuPDF
import numpy as np
import matplotlib.pyplot as plt

# ----------- Define Stopwords Manually (faster, avoids download) -----------
stop_words = set([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','any','both','each','few','more',
    'most','other','some','such','no','nor','not','only','own','same','so',
    'than','too','very','s','t','can','will','just','don','should','now'
])

# ----------- Resume Cleaning Function -----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\S*@\S*\s?', '', text)  # remove emails
    text = re.sub(r'http\S+', '', text)     # remove URLs
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove non-alphabetic
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# ----------- Extract Text from PDF -----------
def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

# ----------- Load Model & Vectorizer (cached) -----------
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("resume_classifier_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

# ----------- Streamlit UI -----------
st.set_page_config(page_title="Resume Classifier", page_icon="📄")
st.title("🤖 Resume Job Category Classifier")
st.markdown("Upload your resume (PDF or TXT), and get the top 3 predicted job categories!")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "txt"])

if uploaded_file:
    with st.spinner("Extracting and classifying..."):
        if uploaded_file.type == "application/pdf":
            raw_text = extract_text_from_pdf(uploaded_file)
        else:
            raw_text = uploaded_file.read().decode("utf-8")

        cleaned_text = clean_text(raw_text)
        
        if cleaned_text.strip():
            model, vectorizer = load_model_and_vectorizer()
            vectorized_input = vectorizer.transform([cleaned_text])

            probabilities = model.predict_proba(vectorized_input)[0]
            all_categories = list(zip(model.classes_, probabilities))

            # Top 3 categories for progress bars
            top3_indices = np.argsort(probabilities)[::-1][:3]
            top3_categories = [(model.classes_[i], probabilities[i]) for i in top3_indices]

            st.subheader("🔍 Top 3 Predicted Categories:")

            for category, prob in top3_categories:
                st.write(f"**{category}** : {prob*100:.2f}%")
                st.progress(prob)

            # Horizontal bar chart for all categories
            labels = [cat for cat, _ in all_categories]
            sizes = [prob*100 for _, prob in all_categories]

            sorted_indices = np.argsort(sizes)
            labels_sorted = [labels[i] for i in sorted_indices]
            sizes_sorted = [sizes[i] for i in sorted_indices]

            fig, ax = plt.subplots(figsize=(8, len(labels_sorted)*0.4))
            bars = ax.barh(labels_sorted, sizes_sorted, color='skyblue')
            ax.set_xlabel("Probability (%)")
            ax.set_title("All Category Probabilities")

            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{width:.2f}%", va='center')

            st.subheader("📊 Category Probability Distribution (All Classes)")
            st.pyplot(fig)

        else:
            st.warning("Resume text appears empty after cleaning. Please upload a different file.")

        st.info("Ensure your resume is well-formatted and contains sufficient text for classification.")
