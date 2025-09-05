# ICU Outcome Prediction

Multimodal machine learning pipeline predicting patient outcomes from time-series vitals and clinical notes.

## Highlights
- Baseline: TF-IDF + Logistic Regression → 78% AUROC
- NLP: BioClinicalBERT embeddings → 85% AUROC
- Time-series: LSTM / Transformer → +6–8% AUROC improvement
- Automated retraining with GitHub Actions (30% faster)
- Deployed with Streamlit for interactive patient risk exploration

## Installation
```bash
git clone https://github.com/<your-username>/icu-outcome-prediction.git
cd icu-outcome-prediction
pip install -r requirements.txt
```

## Run Streamlit App
```bash
cd app
streamlit run streamlit_app.py
```

## Repository Structure
- `src/` → ML pipeline
- `notebooks/` → Experiments
- `app/` → Streamlit dashboard
- `.github/workflows/ci.yml` → CI/CD automation