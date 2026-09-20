#!/usr/bin/env python3
"""
train_model.py  -  train the Phase-2 model from your master sheet
=================================================================

    python train_model.py --xlsx Master_Publications_SDG_Mapping.xlsx

Expected sheet ("Master Sheet"): columns  Title, Abstract, Keywords  and one or
more SDG slot columns (SDG 1, SDG 2, ...) whose cells look like "SDG 3" or
"SDG 3 - Good Health...".  Rows with no SDG are ignored (unlabelled).

Optional "no SDG" examples
  --negatives Unmatched_No_SDG_Publications.xlsx   (Title / Abstract / Keywords)
  Papers Elsevier did NOT tag with any SDG are added as all-zero rows, so the model
  also learns what "no SDG applies" looks like.  Rows duplicating a labelled paper
  (or each other) are dropped.  Rows in the master sheet that have no SDG are used
  the same way (--no-master-untagged to disable).  Caveat: "untagged by Elsevier" is
  not the same as "verified to have no SDG".

What it does
  1. builds a multi-label target (a paper may carry 1-7 SDGs)
  2. stratified 60/20/20 split -> reports validation and TEST metrics
     (and, with negatives, the false-flag rate on held-out untagged papers)
  3. refits the model on ALL rows (use --no-refit to keep the train-only model)
     and saves model/sdg_model.pkl
"""

import argparse
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, f1_score, hamming_loss,
                             precision_score, recall_score)
from sklearn.multiclass import OneVsRestClassifier


def resolve_sheet(path, wanted):
    """Use the requested sheet if it exists, otherwise the first sheet of the workbook."""
    names = pd.ExcelFile(path).sheet_names
    if wanted in names:
        return wanted
    print(f"Sheet {wanted!r} not found in {os.path.basename(path)}; using {names[0]!r}")
    return names[0]


def load_labelled(path, sheet):
    sheet = resolve_sheet(path, sheet)
    df = pd.read_excel(path, sheet_name=sheet)
    slot_cols = [c for c in df.columns if re.match(r"^\s*SDG\b", str(c), re.I)]
    if not slot_cols:
        sys.exit("No 'SDG ...' columns found in the sheet.")

    def nums(row):
        found = set()
        for c in slot_cols:
            v = row[c]
            if pd.notna(v):
                m = re.search(r"(?:SDG\s*)?(\d{1,2})", str(v), re.I)
                if m and 1 <= int(m.group(1)) <= 17:
                    found.add(int(m.group(1)))
        return sorted(found)

    df["labels"] = df.apply(nums, axis=1)
    lab = df[df["labels"].map(len) > 0].copy().reset_index(drop=True)
    for col in ("Title", "Abstract", "Keywords"):
        lab[col] = lab[col].fillna("") if col in lab else ""
    lab["text"] = (lab["Title"] + ". " + lab["Abstract"] + ". " + lab["Keywords"]).str.strip()
    present = sorted({n for l in lab["labels"] for n in l})
    Y = np.array([[int(n in l) for n in present] for l in lab["labels"]])
    return lab, Y, present


def _norm_title(t):
    return re.sub(r"\W+", " ", str(t).lower()).strip()


def load_negatives(path, labelled, master_path=None, sheet="Master Sheet", use_master=True):
    """Return a list of texts of papers with no SDG (deduplicated)."""
    frames = []
    if path:
        frames.append(pd.read_excel(path))
    if use_master and master_path:
        m = pd.read_excel(master_path, sheet_name=resolve_sheet(master_path, sheet))
        slot = [c for c in m.columns if re.match(r"^\s*SDG\b", str(c), re.I)]
        untagged = m[m[slot].isna().all(axis=1)] if slot else m.iloc[0:0]
        if len(untagged):
            frames.append(untagged)
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    for col in ("Title", "Abstract", "Keywords"):
        df[col] = df[col].fillna("") if col in df else ""
    seen = {_norm_title(t) for t in labelled["Title"]}
    texts = []
    for _, r in df.iterrows():
        key = _norm_title(r["Title"])
        if not key or key in seen:
            continue
        seen.add(key)
        text = (r["Title"] + ". " + r["Abstract"] + ". " + r["Keywords"]).strip()
        if len(text) > 40:
            texts.append(text)
    return texts


def split_indices(Y, seed=42):
    idx = np.arange(len(Y))
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit as MS
        a = MS(n_splits=1, test_size=0.40, random_state=seed)
        tr, tmp = next(a.split(idx, Y))
        b = MS(n_splits=1, test_size=0.50, random_state=seed)
        v, t = next(b.split(tmp, Y[tmp]))
        return tr, tmp[v], tmp[t]
    except ImportError:
        print("iterative-stratification not installed -> plain random split "
              "(pip install iterative-stratification for a stratified one)")
        rng = np.random.RandomState(seed)
        p = rng.permutation(len(Y))
        n1, n2 = int(.6 * len(Y)), int(.8 * len(Y))
        return p[:n1], p[n1:n2], p[n2:]


def make_model():
    return OneVsRestClassifier(
        LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0), n_jobs=-1)


def make_tfidf():
    return TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2,
                           sublinear_tf=True, stop_words="english")


def report(name, Yt, Yp):
    print(f"{name:26s} micro-F1={f1_score(Yt, Yp, average='micro', zero_division=0):.3f}  "
          f"macro-F1={f1_score(Yt, Yp, average='macro', zero_division=0):.3f}  "
          f"P={precision_score(Yt, Yp, average='micro', zero_division=0):.3f}  "
          f"R={recall_score(Yt, Yp, average='micro', zero_division=0):.3f}  "
          f"hamming={hamming_loss(Yt, Yp):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--sheet", default="Master Sheet")
    ap.add_argument("--out", default=os.path.join("model", "sdg_model.pkl"))
    ap.add_argument("--negatives", default=None,
                    help="xlsx of papers with no SDG (Title/Abstract/Keywords)")
    ap.add_argument("--no-master-untagged", action="store_true",
                    help="do not use the untagged rows of the master sheet as negatives")
    ap.add_argument("--no-refit", action="store_true",
                    help="save the train-split model instead of refitting on all rows")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.sheet = resolve_sheet(args.xlsx, args.sheet)      # resolve once, reuse everywhere
    lab, Y, present = load_labelled(args.xlsx, args.sheet)
    counts = {n: int(Y[:, i].sum()) for i, n in enumerate(present)}
    print(f"{len(lab)} labelled documents; SDGs present: {present}")
    print("examples per SDG:", counts)

    negs = load_negatives(args.negatives, lab, args.xlsx, args.sheet,
                          use_master=not args.no_master_untagged)
    print(f"{len(negs)} 'no SDG' papers available as negatives")

    tr, va, te = split_indices(Y, args.seed)
    texts = lab["text"].values
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(negs))
    n1, n2 = int(.6 * len(negs)), int(.8 * len(negs))
    neg = np.array(negs, dtype=object)
    neg_tr, neg_va, neg_te = neg[perm[:n1]], neg[perm[n1:n2]], neg[perm[n2:]]
    zeros = lambda k: np.zeros((k, Y.shape[1]), dtype=int)

    fit_texts = list(texts[tr]) + list(neg_tr)
    fit_Y = np.vstack([Y[tr], zeros(len(neg_tr))])
    tfidf = make_tfidf()
    model = make_model().fit(tfidf.fit_transform(fit_texts), fit_Y)

    print()
    report("TF-IDF+LogReg [VAL]", Y[va], model.predict(tfidf.transform(texts[va])))
    pt = model.predict(tfidf.transform(texts[te]))
    report("TF-IDF+LogReg [TEST]", Y[te], pt)
    fpr = None
    if len(neg_te):
        fpr = float(model.predict(tfidf.transform(list(neg_te))).any(axis=1).mean())
        print(f"False-flag rate on held-out untagged papers (>=1 SDG at 0.5): {fpr*100:.1f}%")
    print("\nPer-SDG report on TEST (tagged papers):")
    print(classification_report(Y[te], pt, target_names=[f"SDG {n}" for n in present],
                                zero_division=0))
    per_f1 = {n: float(f) for n, f in zip(present, f1_score(Y[te], pt, average=None,
                                                             zero_division=0))}

    if not args.no_refit:
        print("Refitting on all rows for deployment ...")
        all_texts = list(texts) + list(negs)
        all_Y = np.vstack([Y, zeros(len(negs))])
        tfidf = make_tfidf()
        model = make_model().fit(tfidf.fit_transform(all_texts), all_Y)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({
            "tfidf": tfidf, "model": model, "present_sdgs": present,
            "model_name": "TF-IDF + Logistic Regression",
            "n_examples": counts, "per_sdg_test_f1": per_f1,
            "n_negatives": len(negs), "false_flag_rate_test": fpr,
            "refit_on_all": not args.no_refit,
            "sklearn_version": sklearn.__version__,
        }, f)
    print(f"Saved {args.out}  (scikit-learn {sklearn.__version__})")
    weak = [n for n in present if counts[n] < 100]
    if weak:
        print(f"Note: SDGs with <100 training examples (ML less reliable): {weak}")
    missing = [n for n in range(1, 18) if n not in present]
    if missing:
        print(f"Note: SDGs never seen in training (ML cannot score them): {missing}")


if __name__ == "__main__":
    main()
