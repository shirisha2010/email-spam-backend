"""Flask REST API for email spam detection.

Endpoints:
  GET  /api/health      -> service status
  GET  /api/metrics     -> training metrics of the deployed model
  POST /api/predict     -> {"text": "..."}  -> {label, is_spam, confidence, spam_probability, top_spam_words}
  POST /api/batch       -> {"emails": ["...", "..."]} -> list of predictions
"""
import json
import os

import joblib
from flask import Flask, jsonify, request

from preprocess import clean_text  # noqa: F401  (needed so joblib can unpickle the pipeline)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "spam_model.joblib")
METRICS_PATH = os.path.join(BASE, "model", "metrics.json")

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


@app.after_request
def add_cors(resp):  # allow the separately hosted frontend to call the API
    resp.headers["Access-Control-Allow-Origin"] = os.environ.get("CORS_ORIGIN", "*")
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


def explain(text, limit=5):
    """Words in the email that push the score towards spam (works for linear models / NB)."""
    try:
        tfidf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
        vec = tfidf.transform([text])
        names = tfidf.get_feature_names_out()
        if hasattr(clf, "feature_log_prob_"):
            weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        elif hasattr(clf, "coef_"):
            weights = clf.coef_[0]
        else:  # calibrated classifier -> average its base estimators' coefficients
            weights = sum(c.estimator.coef_[0] for c in clf.calibrated_classifiers_) / len(clf.calibrated_classifiers_)
        idx = vec.nonzero()[1]
        scored = sorted(((float(weights[i] * vec[0, i]), names[i]) for i in idx), reverse=True)
        return [w for s, w in scored if s > 0][:limit]
    except Exception:
        return []


def predict_one(text):
    p_spam = float(model.predict_proba([text])[0][1])
    is_spam = p_spam >= 0.5
    return {
        "label": "spam" if is_spam else "ham",
        "is_spam": is_spam,
        "spam_probability": round(p_spam, 4),
        "confidence": round(p_spam if is_spam else 1 - p_spam, 4),
        "top_spam_words": explain(text) if is_spam else [],
    }


@app.route("/api/health")
def health():
    return jsonify(status="ok")


@app.route("/api/metrics")
def metrics():
    with open(METRICS_PATH) as f:
        return jsonify(json.load(f))


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(error="Field 'text' is required"), 400
    if len(text) > 20000:
        return jsonify(error="Email text too long (max 20000 chars)"), 413
    return jsonify(predict_one(text))


@app.route("/api/batch", methods=["POST", "OPTIONS"])
def batch():
    if request.method == "OPTIONS":
        return ("", 204)
    emails = (request.get_json(silent=True) or {}).get("emails")
    if not isinstance(emails, list) or not emails:
        return jsonify(error="Field 'emails' must be a non-empty list"), 400
    return jsonify([predict_one(str(e)) for e in emails[:200]])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
