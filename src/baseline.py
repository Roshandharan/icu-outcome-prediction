from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def train_baseline(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_baseline(model, X_test, y_test):
    preds = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)