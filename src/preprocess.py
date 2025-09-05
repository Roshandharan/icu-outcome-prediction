import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(path: str):
    df = pd.read_csv(path)
    return df

def preprocess_text(df, text_column="notes", label_column="outcome"):
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_column], df[label_column], test_size=0.2, random_state=42
    )
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer