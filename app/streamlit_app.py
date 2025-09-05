import streamlit as st
import sys
import os
import pandas as pd

# Ensure the parent folder is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.preprocess import preprocess_text
    from src.baseline import train_baseline, evaluate_baseline
except ModuleNotFoundError:
    st.error("Could not find the src module. Check folder structure and __init__.py files.")
    raise

from src.preprocess import preprocess_text
from src.baseline import train_baseline, evaluate_baseline

st.title("ICU Outcome Prediction")

uploaded_file = st.file_uploader("Upload ICU Data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    # Preprocess the data
    X_train, X_test, y_train, y_test, vectorizer = preprocess_text(df)
    
    # Train and evaluate baseline model
    model = train_baseline(X_train, y_train)
    auc = evaluate_baseline(model, X_test, y_test)

    st.write(f"Baseline Model AUROC: {auc:.2f}")
