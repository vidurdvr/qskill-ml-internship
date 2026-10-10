import os
import sys
import argparse
import joblib

# Ensure local imports work regardless of CWD
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.preprocess import clean_text

MODEL_PATH = os.path.join(CURRENT_DIR, "outputs", "best_spam_pipeline.joblib")

def classify_message(message: str, model_path: str = MODEL_PATH):
    """
    Classify a single SMS / email message as SPAM or HAM.
    Returns a dictionary containing label, probabilities, and cleaned text.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            f"Please run 'python src/train.py' first to train and export the pipeline."
        )
    
    pipeline = joblib.load(model_path)
    cleaned = clean_text(message)
    
    # Predict
    pred_label_idx = pipeline.predict([cleaned])[0]
    label = "SPAM" if pred_label_idx == 1 else "HAM"
    
    # Probabilities
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba([cleaned])[0]
        ham_prob = float(probas[0])
        spam_prob = float(probas[1])
    else:
        spam_prob = float(pred_label_idx)
        ham_prob = 1.0 - spam_prob
        
    confidence = spam_prob if label == "SPAM" else ham_prob
    
    return {
        "original_message": message,
        "cleaned_text": cleaned,
        "label": label,
        "spam_probability": spam_prob,
        "ham_probability": ham_prob,
        "confidence": confidence
    }

def main():
    parser = argparse.ArgumentParser(
        description="SMS / Email Spam Detector CLI — Predict whether a message is SPAM or HAM."
    )
    parser.add_argument(
        "--message", "-m",
        type=str,
        help="Input text message to classify.",
        required=False
    )
    args = parser.parse_args()

    if args.message:
        raw_message = args.message
    else:
        print("=" * 60)
        print("          SMS / Email Spam Classifier (CLI) ")
        print("=" * 60)
        raw_message = input("\nEnter message text: ").strip()
        if not raw_message:
            print("No message provided. Exiting.")
            return

    res = classify_message(raw_message)
    
    print("\n" + "-" * 50)
    print(f"VERDICT:            [{res['label']}]")
    print(f"Confidence:         {res['confidence'] * 100:.2f}%")
    print(f"Spam Probability:   {res['spam_probability'] * 100:.2f}%")
    print(f"Ham Probability:    {res['ham_probability'] * 100:.2f}%")
    print(f"Cleaned Text:       \"{res['cleaned_text']}\"")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
