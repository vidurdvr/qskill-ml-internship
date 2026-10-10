import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)

# Ensure local imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.data_loader import load_spam_data
from src.preprocess import clean_text

OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Set high-quality visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

def perform_eda(df: pd.DataFrame):
    """Generate and save EDA plots: class balance and message length distribution."""
    print("\n--- 1. Exploratory Data Analysis ---")
    df['char_length'] = df['message'].astype(str).apply(len)
    df['word_count'] = df['message'].astype(str).apply(lambda s: len(s.split()))
    
    # 1. Class distribution plot
    plt.figure(figsize=(7, 4.5))
    ax = sns.countplot(
        x='label', 
        data=df, 
        palette={'ham': '#4C72B0', 'spam': '#C44E52'},
        hue='label',
        legend=False
    )
    plt.title("SMS Class Distribution (Ham vs. Spam)", fontsize=14, weight='bold', pad=12)
    plt.xlabel("Class Label", fontsize=12)
    plt.ylabel("Number of Messages", fontsize=12)
    
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f'{height:,} ({height/total*100:.1f}%)',
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='bottom', fontsize=11, xytext=(0, 4),
            textcoords='offset points'
        )
    plt.tight_layout()
    class_dist_path = os.path.join(OUTPUTS_DIR, "class_distribution.png")
    plt.savefig(class_dist_path)
    plt.close()
    print(f"Saved: {class_dist_path}")

    # 2. Message length distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(
        data=df, x='char_length', hue='label', bins=60, kde=True,
        palette={'ham': '#4C72B0', 'spam': '#C44E52'}, ax=axes[0],
        element="step", common_norm=False
    )
    axes[0].set_title("Character Length Distribution", fontsize=13, weight='bold')
    axes[0].set_xlabel("Message Character Length", fontsize=11)
    axes[0].set_ylabel("Density / Count", fontsize=11)
    axes[0].set_xlim(0, 300)

    sns.histplot(
        data=df, x='word_count', hue='label', bins=40, kde=True,
        palette={'ham': '#4C72B0', 'spam': '#C44E52'}, ax=axes[1],
        element="step", common_norm=False
    )
    axes[1].set_title("Word Count Distribution", fontsize=13, weight='bold')
    axes[1].set_xlabel("Message Word Count", fontsize=11)
    axes[1].set_ylabel("Density / Count", fontsize=11)
    axes[1].set_xlim(0, 60)

    plt.suptitle("SMS Length Analysis by Class", fontsize=15, weight='bold', y=1.02)
    plt.tight_layout()
    length_dist_path = os.path.join(OUTPUTS_DIR, "message_length_distribution.png")
    plt.savefig(length_dist_path)
    plt.close()
    print(f"Saved: {length_dist_path}")

def run_training_pipeline():
    """Execute complete ML pipeline with CV, GridSearch, Evaluation, and Serialization."""
    df = load_spam_data()
    perform_eda(df)
    
    print("\n--- 2. Text Preprocessing ---")
    df['clean_message'] = df['message'].apply(clean_text)
    # Binary target: ham = 0, spam = 1
    df['target'] = (df['label'] == 'spam').astype(int)
    
    # 80/20 Stratified Train/Test Split
    X = df['clean_message']
    y = df['target']
    raw_texts = df['message']
    
    X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
        X, y, raw_texts, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Train set: {len(X_train)} samples ({y_train.sum()} spam, {len(y_train) - y_train.sum()} ham)")
    print(f"Test set:  {len(X_test)} samples ({y_test.sum()} spam, {len(y_test) - y_test.sum()} ham)")

    # Define model candidate pipelines with BoW and TF-IDF
    feature_extractors = {
        'BoW': CountVectorizer(ngram_range=(1, 2)),
        'TF-IDF': TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    }

    classifiers = {
        'Multinomial Naive Bayes': (MultinomialNB(), {'clf__alpha': [0.01, 0.1, 0.5, 1.0, 2.0]}),
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            {'clf__C': [0.1, 1.0, 5.0, 10.0]}
        ),
        'Linear SVM': (
            CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=2000, dual='auto')),
            {'clf__estimator__C': [0.1, 1.0, 5.0, 10.0]}
        )
    }

    print("\n--- 3. 5-Fold Stratified Cross-Validation & Hyperparameter Tuning ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    trained_models = {}

    for vec_name, vectorizer in feature_extractors.items():
        for clf_name, (clf, param_grid) in classifiers.items():
            model_key = f"{clf_name} ({vec_name})"
            print(f"Tuning {model_key}...")
            
            pipe = Pipeline([
                ('vec', vectorizer),
                ('clf', clf)
            ])
            
            grid = GridSearchCV(
                pipe,
                param_grid=param_grid,
                cv=cv,
                scoring='f1',
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else y_pred
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)
            avg_prec = average_precision_score(y_test, y_proba)
            
            results.append({
                'Model': clf_name,
                'Feature Extractor': vec_name,
                'Best Parameters': str(grid.best_params_),
                'CV F1 Score': grid.best_score_,
                'Test Accuracy': acc,
                'Test Precision': prec,
                'Test Recall': rec,
                'Test F1-Score': f1,
                'Test ROC-AUC': roc_auc,
                'Test PR-AUC': avg_prec
            })
            
            trained_models[model_key] = {
                'pipeline': best_model,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'f1': f1,
                'precision': prec,
                'recall': rec
            }

    results_df = pd.DataFrame(results).sort_values(by='Test F1-Score', ascending=False).reset_index(drop=True)
    summary_csv = os.path.join(OUTPUTS_DIR, "metrics_summary.csv")
    results_df.to_csv(summary_csv, index=False)
    print("\n--- Model Benchmark Summary ---")
    print(results_df[['Model', 'Feature Extractor', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score']].to_string(index=False))
    print(f"\nSaved benchmark metrics: {summary_csv}")

    # Determine best model based on F1-Score and Precision (Spam false positives are costly)
    best_model_name = results_df.iloc[0]['Model'] + f" ({results_df.iloc[0]['Feature Extractor']})"
    best_entry = trained_models[best_model_name]
    best_pipeline = best_entry['pipeline']
    
    print(f"\n>>> Best Performing Pipeline: {best_model_name} <<<")
    
    # Save the best pipeline
    model_artifact_path = os.path.join(OUTPUTS_DIR, "best_spam_pipeline.joblib")
    joblib.dump(best_pipeline, model_artifact_path)
    print(f"Saved best model artifact: {model_artifact_path}")

    # 4. Detailed Evaluation Visualizations
    print("\n--- 4. Visual Evaluation & Curves ---")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_entry['y_pred'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['HAM (Legit)', 'SPAM'],
        yticklabels=['HAM (Legit)', 'SPAM'],
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix: {best_model_name}", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved: {cm_path}")

    # Precision-Recall Curves comparison
    plt.figure(figsize=(8, 6))
    for model_key, data in trained_models.items():
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, data['y_proba'])
        ap = average_precision_score(y_test, data['y_proba'])
        plt.plot(recall_vals, precision_vals, lw=2, label=f"{model_key} (AP = {ap:.3f})")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve Comparison (Imbalanced SMS Data)", fontsize=13, weight='bold', pad=12)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    pr_path = os.path.join(OUTPUTS_DIR, "precision_recall_curve.png")
    plt.savefig(pr_path)
    plt.close()
    print(f"Saved: {pr_path}")

    # 5. Top Spam Indicating Words Feature Analysis
    print("\n--- 5. Feature Importance / Top Spam Words ---")
    # For linear models or naive bayes, extract learned weights
    # We will build a dedicated TF-IDF + Logistic Regression / Naive Bayes extractor for clean feature weights
    vec = best_pipeline.named_steps['vec']
    clf = best_pipeline.named_steps['clf']
    feature_names = np.array(vec.get_feature_names_out())

    if hasattr(clf, 'coef_'):
        weights = clf.coef_[0]
    elif hasattr(clf, 'feature_log_prob_'):
        # Difference in log probabilities between spam and ham
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    elif hasattr(clf, 'calibrated_classifiers_'):
        # CalibratedClassifierCV
        weights = np.mean([c.estimator.coef_[0] for c in clf.calibrated_classifiers_], axis=0)
    else:
        weights = np.zeros(len(feature_names))

    top_n = 20
    top_spam_indices = np.argsort(weights)[-top_n:]
    top_ham_indices = np.argsort(weights)[:top_n]

    top_spam_df = pd.DataFrame({
        'Feature': feature_names[top_spam_indices],
        'Weight': weights[top_spam_indices]
    }).sort_values('Weight', ascending=True)

    top_spam_csv = os.path.join(OUTPUTS_DIR, "top_spam_words.csv")
    top_spam_df.to_csv(top_spam_csv, index=False)

    plt.figure(figsize=(9, 7))
    sns.barplot(
        x='Weight', y='Feature', data=top_spam_df,
        palette='Reds_r', hue='Feature', legend=False
    )
    plt.title(f"Top {top_n} Strongest Spam-Indicating Features", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Model Feature Weight / Log-Odds", fontsize=11)
    plt.ylabel("N-Gram / Token", fontsize=11)
    plt.tight_layout()
    top_words_path = os.path.join(OUTPUTS_DIR, "top_spam_words.png")
    plt.savefig(top_words_path)
    plt.close()
    print(f"Saved: {top_words_path}")

    # 6. Error Analysis: Misclassified Samples
    print("\n--- 6. Error Analysis on Test Set ---")
    test_analysis_df = pd.DataFrame({
        'Original Message': raw_test.values,
        'Cleaned Message': X_test.values,
        'True Label': ['SPAM' if y == 1 else 'HAM' for y in y_test.values],
        'Predicted Label': ['SPAM' if y == 1 else 'HAM' for y in best_entry['y_pred']],
        'Spam Probability': best_entry['y_proba']
    })
    
    errors_df = test_analysis_df[test_analysis_df['True Label'] != test_analysis_df['Predicted Label']].copy()
    errors_path = os.path.join(OUTPUTS_DIR, "error_analysis.csv")
    errors_df.to_csv(errors_path, index=False)
    print(f"Total test errors: {len(errors_df)} out of {len(y_test)} ({len(errors_df)/len(y_test)*100:.2f}%)")
    print(f"Saved error analysis: {errors_path}")
    
    # Print sample false positives and false negatives
    fps = errors_df[(errors_df['True Label'] == 'HAM') & (errors_df['Predicted Label'] == 'SPAM')]
    fns = errors_df[(errors_df['True Label'] == 'SPAM') & (errors_df['Predicted Label'] == 'HAM')]
    
    print(f"\nFalse Positives (Legitimate marked as Spam, Costly!): {len(fps)}")
    for _, row in fps.head(3).iterrows():
        print(f" - [P(Spam)={row['Spam Probability']:.2f}] {row['Original Message']}")
        
    print(f"\nFalse Negatives (Spam missed): {len(fns)}")
    for _, row in fns.head(3).iterrows():
        print(f" - [P(Spam)={row['Spam Probability']:.2f}] {row['Original Message']}")

    print("\n==========================================")
    print("Spam Detector Training & Evaluation Complete!")
    print("==========================================")

if __name__ == "__main__":
    run_training_pipeline()
