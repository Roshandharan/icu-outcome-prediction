from sklearn.metrics import roc_auc_score

def evaluate_predictions(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)