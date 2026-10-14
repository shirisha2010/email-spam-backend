# Email Spam Detection – Backend (Flask + scikit-learn)

## Setup
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train.py            # trains on data/emails.csv, writes model/spam_model.joblib
python app.py              # API on http://localhost:5000
```

## Pipeline
Text -> `clean_text` (lowercase, URL/email/number tokens, punctuation removal) -> TF-IDF (1-2 grams, stop-words removed)
-> classifier. Naive Bayes, Logistic Regression and Linear SVM are trained; the best F1 on a stratified 80/20 split is saved.

## Use a real dataset
`data/emails.csv` is a synthetic starter set. For real accuracy download e.g. the Kaggle *SMS Spam Collection* (`spam.csv`)
or the Enron/SpamAssassin corpus, then run `python train.py path/to/spam.csv` (columns `label,text` or `v1,v2`).

## API
| Method | Route | Body | Response |
|---|---|---|---|
| GET | /api/health | – | `{status:"ok"}` |
| GET | /api/metrics | – | model comparison metrics |
| POST | /api/predict | `{"text":"..."}` | `{label,is_spam,spam_probability,confidence,top_spam_words}` |
| POST | /api/batch | `{"emails":["..",".."]}` | list of predictions |

Production: `gunicorn app:app`. Set `CORS_ORIGIN` to your frontend URL to restrict CORS.

> **Note:** the bundled dataset is generated from templates (`generate_dataset.py`), so the near-perfect scores are optimistic.
> Retrain on a real corpus before relying on the results.
