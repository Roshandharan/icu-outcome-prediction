import streamlit as st
import sys
import os
import pandas as pd

# Add the parent directory to Python path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
