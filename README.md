# ICU Outcome Prediction

A **multimodal machine learning pipeline** designed to predict patient outcomes in intensive care units (ICUs) using **time-series vitals** and **clinical notes**. This repository combines NLP, deep learning, and automated deployment to provide an interpretable and actionable patient risk prediction system.

![ICU Outcome Prediction Pipeline](path/to/your/image.png)

---

## Highlights

### Predictive Modeling

- **Baseline Model**
  - Features: Bag-of-words and **TF-IDF** representations of clinical notes.
  - Model: Logistic Regression with L2 regularization.
  - Performance: **78% AUROC**, providing a transparent baseline for comparison.

- **NLP-Based Model**
  - Features: **BioClinicalBERT embeddings** extracted from unstructured clinical notes.
  - Model: Dense neural network classifier.
  - Performance: **85% AUROC**, capturing nuanced clinical context from free-text notes.

- **Time-Series Model**
  - Features: Patient vitals (heart rate, blood pressure, respiratory rate, oxygen saturation) sampled at high frequency.
  - Models: 
    - **LSTM** for sequential modeling of vitals.
    - **Transformer** to capture long-range dependencies in physiological signals.
  - Performance: +6–8% AUROC improvement over NLP-only model.

- **Multimodal Fusion**
  - Combines textual embeddings with time-series features.
  - Uses concatenation followed by fully connected layers and attention mechanisms to prioritize critical features.
  - Achieves the highest predictive performance while maintaining interpretability.

---

### Data Preprocessing

- **Clinical Notes**
  - Tokenization, stopword removal, and normalization.
  - Extraction of **medical entities** (ICD codes, medications, symptoms) using spaCy and scispaCy pipelines.
  - Conversion into embeddings with BioClinicalBERT.

- **Time-Series Vitals**
  - Resampling to uniform time intervals.
  - Missing value imputation using forward-fill and interpolation.
  - Normalization per patient to account for baseline differences.
  - Feature engineering: rolling averages, slopes, and variability metrics.

---

### Model Training & Evaluation

- **Data Split:** 70% training, 15% validation, 15% test, stratified by outcome.
- **Loss Functions:** Binary cross-entropy with class weighting for imbalanced outcomes.
- **Metrics:** AUROC, precision-recall curve, F1-score, calibration plots.
- **Cross-Validation:** 5-fold stratified CV to ensure model robustness.
- **Explainability:** SHAP values for feature importance and patient-specific risk factors.

---

### Automation & Deployment

- **Automated Retraining**
  - GitHub Actions triggers retraining when new patient data is added.
  - Reduces model update time by **30%**.
  - Automatic versioning and model checkpointing using MLflow.

- **Streamlit Dashboard**
  - Interactive visualization of patient risk trajectories.
  - Allows clinicians to filter by patient, outcome probability, and time horizon.
  - Provides interpretability charts (e.g., feature contribution, vital trends).

- **CI/CD Integration**
  - Tests model integrity and dashboard functionality on every commit.
  - Ensures production-ready deployment with zero downtime.

---

### Repository Structure

- `src/` → Core ML pipeline including:
  - Data preprocessing
  - Feature extraction (time-series & NLP)
  - Model training and evaluation
- `notebooks/` → Experimentation notebooks:
  - Exploratory data analysis
  - Model benchmarking
  - Feature importance visualization
- `app/` → Streamlit dashboard:
  - Interactive patient risk exploration
  - Visualizations for vital trends and textual insights
- `.github/workflows/ci.yml` → CI/CD automation:
  - Retraining triggers
  - Unit tests
  - Deployment to staging/production

---

### Future Work

- Incorporate **real-time streaming data** for continuous risk prediction.
- Explore **graph-based models** to integrate patient histories and relational data.
- Enhance **clinical interpretability** using attention maps and causal inference.

---

### References

1. Alsentzer et al., "Publicly Available Clinical BERT Embeddings," 2019.  
2. Hochreiter & Schmidhuber, "Long Short-Term Memory," 1997.  
3. Vaswani et al., "Attention is All You Need," 2017.  

---

