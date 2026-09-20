"""
sdg_ml.py  -  Phase 2: ML cross-check (TF-IDF + Logistic Regression)
====================================================================

Model choice comes from the earlier comparison on the labelled master sheet
(1,934 / 645 / 641 train / val / test docs, multi-label, SDG 17 absent):

    TF-IDF + Logistic Regression   macro-F1 0.685 (val)   0.614 (test)   <- chosen
    TF-IDF + Linear SVM                     0.636
    TF-IDF + XGBoost (OvR)                  0.625
    CatBoost + TF-IDF                       0.584
    CatBoost (native text)                  0.499
    TF-IDF + Random Forest                  0.338

Logistic Regression won on macro-F1, i.e. it is the best balanced across SDGs,
not just the well-populated ones.  It is also fast, gives calibrated-ish
probabilities and needs no GPU.  (A SciBERT fine-tune was prepared as an optional
experiment; no result showing it beats this baseline has been reported, so it
is NOT used here.)

The model is trained by train_model.py and stored in model/sdg_model.pkl.
"""

import pickle

import numpy as np


def load_model(path):
    """Return (artifacts, warning_or_None)."""
    with open(path, "rb") as f:
        art = pickle.load(f)
    warn = None
    trained_with = art.get("sklearn_version")
    if trained_with:
        import sklearn
        if sklearn.__version__ != trained_with:
            warn = (f"Model was trained with scikit-learn {trained_with} but "
                    f"{sklearn.__version__} is installed - pin the same version "
                    "in requirements.txt or retrain.")
    return art, warn


def build_text(title, abstract, keywords):
    return f"{title or ''}. {abstract or ''}. {keywords or ''}".strip()


def predict_proba(art, title, abstract, keywords):
    """Return {sdg_number: probability} for the SDGs the model was trained on."""
    X = art["tfidf"].transform([build_text(title, abstract, keywords)])
    model = art["model"]
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X))[0]
    else:                                           # e.g. LinearSVC
        z = np.asarray(model.decision_function(X))[0]
        p = 1.0 / (1.0 + np.exp(-z))
    return {int(n): float(v) for n, v in zip(art["present_sdgs"], p)}


def reliability(art, sdg, min_examples=100):
    """'ok' | 'limited' (few training examples) | 'none' (never seen)."""
    counts = art.get("n_examples", {})
    if sdg not in art["present_sdgs"]:
        return "none"
    n = counts.get(sdg)
    if n is not None and n < min_examples:
        return "limited"
    return "ok"
