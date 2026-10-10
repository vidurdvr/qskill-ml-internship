import os
import sys
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.preprocess import clean_text

MODEL_PATH = os.path.join(CURRENT_DIR, "outputs", "best_spam_pipeline.joblib")
METRICS_PATH = os.path.join(CURRENT_DIR, "outputs", "metrics_summary.csv")
TOP_WORDS_PATH = os.path.join(CURRENT_DIR, "outputs", "top_spam_words.csv")

st.set_page_config(
    page_title="SMS Spam Detector — QSkill AI/ML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model artifact not found. Please train the model first by running `src/train.py`.")
        return None
    return joblib.load(MODEL_PATH)

pipeline = load_model()

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .spam-box {
        background-color: #FEE2E2;
        border-left: 6px solid #EF4444;
        padding: 1rem 1.2rem;
        border-radius: 0.5rem;
        color: #991B1B;
        font-weight: 600;
        font-size: 1.2rem;
    }
    .ham-box {
        background-color: #D1FAE5;
        border-left: 6px solid #10B981;
        padding: 1rem 1.2rem;
        border-radius: 0.5rem;
        color: #065F46;
        font-weight: 600;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🛡️ SMS Spam Mail Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time NLP Classification Pipeline • QSkill AI/ML Internship</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Project Details")
st.sidebar.markdown("""
**Dataset:** SMS Spam Collection (UCI)
**Architecture:** BoW + Logistic Regression
**Tuning:** 5-Fold Stratified Cross-Validation
**Test Precision:** 100.0% (Zero False Positives)
**Test F1-Score:** 94.7%
""")

st.sidebar.subheader("Quick Presets")
preset = st.sidebar.radio(
    "Choose a sample message to test:",
    [
        "None (Write Custom)",
        "Spam: Lottery Prize Winner",
        "Spam: Urgent Bank Account Alert",
        "Ham: Team Project Meeting",
        "Ham: Casual Friend Catchup"
    ]
)

preset_map = {
    "Spam: Lottery Prize Winner": "WINNER! You have been selected to receive a £1,000 cash prize or a free holiday! Call 09061701461 to claim your reward immediately.",
    "Spam: Urgent Bank Account Alert": "URGENT: Your mobile account has been credited with 500 bonus points. Reply CLAIM to activate before midnight!",
    "Ham: Team Project Meeting": "Hey team, the slide deck for tomorrow's sprint review is updated in the shared folder. Let's do a quick dry run at 10 AM.",
    "Ham: Casual Friend Catchup": "Hey! Are we still grabbing coffee after work today? Let me know when you finish your shift."
}

default_text = preset_map.get(preset, "")

# Main Content Layout
col1, col2 = st.columns([1.2, 0.8], gap="large")

with col1:
    st.subheader("✉️ Message Input")
    user_input = st.text_area(
        "Enter or paste the SMS / Email text below:",
        value=default_text,
        height=140,
        placeholder="Type a message here..."
    )
    
    classify_btn = st.button("🔍 Classify Message", type="primary", use_container_width=True)

    if classify_btn or (user_input and preset != "None (Write Custom)"):
        if not user_input.strip():
            st.warning("Please enter a message to analyze.")
        elif pipeline is None:
            st.error("Model is not loaded.")
        else:
            cleaned = clean_text(user_input)
            pred_idx = pipeline.predict([cleaned])[0]
            probas = pipeline.predict_proba([cleaned])[0]
            
            spam_prob = float(probas[1])
            ham_prob = float(probas[0])
            is_spam = pred_idx == 1
            
            st.markdown("### Prediction Result")
            if is_spam:
                st.markdown(
                    f'<div class="spam-box">🚨 VERDICT: SPAM (Spam Probability: {spam_prob*100:.1f}%)</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="ham-box">✅ VERDICT: HAM / LEGITIMATE (Ham Probability: {ham_prob*100:.1f}%)</div>',
                    unsafe_allow_html=True
                )
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Confidence Bars
            st.write("**Prediction Probabilities:**")
            col_spam, col_ham = st.columns(2)
            with col_spam:
                st.metric("Spam Risk", f"{spam_prob*100:.2f}%")
                st.progress(spam_prob)
            with col_ham:
                st.metric("Legitimate (Ham) Confidence", f"{ham_prob*100:.2f}%")
                st.progress(ham_prob)
                
            with st.expander("🛠️ Preprocessing Pipeline Breakdown", expanded=True):
                st.write("**Raw Text:**", user_input)
                st.write("**Cleaned & Tokenized Text (NLTK stopwords/punct removed):**", f"`{cleaned}`")

with col2:
    st.subheader("📊 Model Performance & Insights")
    
    tab1, tab2 = st.tabs(["Top Spam Words", "Benchmark Results"])
    
    with tab1:
        if os.path.exists(TOP_WORDS_PATH):
            words_df = pd.read_csv(TOP_WORDS_PATH).tail(10)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sns.barplot(
                x='Weight', y='Feature', data=words_df,
                palette='Reds_r', ax=ax, hue='Feature', legend=False
            )
            ax.set_title("Top 10 Strongest Spam Predictors", fontsize=11, weight='bold')
            ax.set_xlabel("Learned Feature Weight", fontsize=9)
            ax.set_ylabel("Token / N-Gram", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Run training to view feature weights.")
            
    with tab2:
        if os.path.exists(METRICS_PATH):
            metrics_df = pd.read_csv(METRICS_PATH)
            st.dataframe(
                metrics_df[['Model', 'Feature Extractor', 'Test Precision', 'Test Recall', 'Test F1-Score']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Benchmark summary not yet generated.")

st.markdown("---")
st.caption("Developed for QSkill AI/ML Internship • October 2026")
