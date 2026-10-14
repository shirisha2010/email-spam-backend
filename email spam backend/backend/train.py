"""Train and compare classifiers, save the best pipeline to model/spam_model.joblib.

Usage:  python train.py [path/to/dataset.csv]
Dataset formats supported: columns (label,text)  OR  Kaggle 'spam.csv' (v1,v2, latin-1).
"""
import json, sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from preprocess import clean_text


def load(path):
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    if {"v1", "v2"}.issubset(df.columns):
        df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df[["label", "text"]].dropna().drop_duplicates()
    df["label"] = df["label"].astype(str).str.lower().map(lambda x: 1 if x in ("spam", "1") else 0)
    return df


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/emails.csv"
    df = load(path)
    print(f"Loaded {len(df)} emails  (spam={df.label.sum()}, ham={(df.label == 0).sum()})")
    X_train, X_test, y_train, y_test = train_test_split(
        df.text, df.label, test_size=0.2, stratify=df.label, random_state=42)

    models = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=5),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(C=1.0), cv=3),
    }
    results, best, best_f1 = {}, None, -1
    for name, clf in models.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), stop_words="english", sublinear_tf=True, min_df=1)),
            ("clf", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        m = dict(accuracy=accuracy_score(y_test, pred), precision=precision_score(y_test, pred),
                 recall=recall_score(y_test, pred), f1=f1_score(y_test, pred),
                 confusion_matrix=confusion_matrix(y_test, pred).tolist())
        results[name] = m
        print(f"\n== {name} ==\n{classification_report(y_test, pred, target_names=['ham', 'spam'])}")
        if m["f1"] > best_f1:
            best, best_f1, best_name = pipe, m["f1"], name

    print(f"Best model: {best_name} (F1={best_f1:.4f})")
    joblib.dump(best, "model/spam_model.joblib")
    with open("model/metrics.json", "w") as f:
        json.dump({"best_model": best_name, "train_size": len(X_train), "test_size": len(X_test), "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
