# QSkill AI/ML Internship Projects

This repository contains two machine learning projects developed for the **QSkill AI/ML Internship**:

1. **[Spam Mail Detector](spam-detector/)** — An end-to-end NLP classification pipeline that identifies SMS spam with high precision and recall, featuring text preprocessing, TF-IDF / BoW comparison, hyperparameter-tuned classifiers, comprehensive evaluation, model serialization, CLI inference, and a Streamlit demo.
2. **[California House Price Prediction](house-price-prediction/)** — A regression pipeline modeling housing values using the California Housing dataset, featuring exploratory data analysis, domain-driven feature engineering, cross-validated ensemble models (Random Forest, Gradient Boosting, Linear models), residual analysis, and standalone prediction capabilities.

---

## Repository Structure

```
qskill-ml-internship/
├── .gitignore
├── README.md                      # Top-level overview
├── requirements.txt               # Pinned dependencies
├── spam-detector/                 # Project 1: SMS Spam Classification
│   ├── README.md                  # Project documentation & benchmark results
│   ├── notebooks/                 # Step-by-step EDA & experiments notebook
│   ├── src/                       # Source code (preprocessing, training, loader)
│   ├── outputs/                   # Model artifacts & plots
│   ├── predict.py                 # CLI inference script
│   └── app.py                     # Streamlit web demo
└── house-price-prediction/        # Project 2: California Housing Regression
    ├── README.md                  # Project documentation & benchmark results
    ├── notebooks/                 # Step-by-step EDA & experiments notebook
    ├── src/                       # Source code (feature engineering, training, loader)
    ├── outputs/                   # Model artifacts & plots
    └── predict.py                 # CLI inference script
```

---

## Quick Start

### 1. Environment Setup

Create and activate a clean Python virtual environment (Python 3.10+):

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Activate on macOS/Linux:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Running Project 1: Spam Detector

```bash
# Run CLI inference
python spam-detector/predict.py --message "Congratulations! You have won a $1,000 Walmart Gift Card. Click here to claim."

# Launch Streamlit web application
streamlit run spam-detector/app.py
```

### 3. Running Project 2: House Price Prediction

```bash
# Run CLI inference with sample inputs
python house-price-prediction/predict.py --MedInc 4.5 --HouseAge 25 --AveRooms 5.5 --AveBedrms 1.1 --Population 1200 --AveOccup 2.8 --Latitude 37.88 --Longitude -122.23
```

---

## Tech Stack

- **Language:** Python 3.10+
- **Machine Learning:** `scikit-learn`, `numpy`, `pandas`, `scipy`
- **Natural Language Processing:** `nltk`
- **Visualization:** `matplotlib`, `seaborn`
- **Model Deployment & App:** `joblib`, `streamlit`
