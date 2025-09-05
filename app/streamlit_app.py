import streamlit as st
import pandas as pd
from src.preprocess import preprocess_text
from src.baseline import train_baseline, evaluate_baseline

st.title("ICU Outcome Prediction")

uploaded_file = st.file_uploader("Upload ICU Data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    X_train, X_test, y_train, y_test, vectorizer = preprocess_text(df)
    model = train_baseline(X_train, y_train)
    auc = evaluate_baseline(model, X_test, y_test)

    st.write(f"Baseline Model AUROC: {auc:.2f}")