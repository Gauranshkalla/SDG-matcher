# SDG Word Finder

Enter a manuscript's **title, abstract and keywords**. The tool

1. **Boolean matcher (Phase 1)** - checks them against Elsevier's 17 SDG search queries using a real
   AND / OR / NOT / proximity expression tree. `EXACT MATCH` = the query is genuinely satisfied.
2. **ML cross-check (Phase 2)** - TF-IDF + Logistic Regression gives a probability per SDG.
3. **Wording suggestions** - shown **only when no SDG matches exactly** (if any SDG has an exact match, none are
   shown). Then, for the SDGs the ML model finds closest, it suggests small wording changes. Each suggestion is tagged *"one edit -> exact match"* or *"helps, not sufficient alone"*,
   and excluded words in the text that block a match are flagged. **The author decides; nothing is applied.**

| Boolean | ML (>= threshold) | Verdict |
|---|---|---|
| exact | yes | **Confirmed** |
| exact | no / not trained | Boolean match (ML inconclusive - trust Boolean, esp. SDG 1, 14, 15, 17) |
| not satisfied | yes | ML supports it (wording suggestions appear only if *no* SDG matched exactly) |
| not satisfied | no | No match (a valid answer) |

## Why these components
* **Boolean matcher** = the corrected tree-based `sdg_match.py` (the earlier flat-term version wrongly ranked
  SDG 17 first on generic words like "model"). Rebuilt here as `sdg_boolean.py` with Scopus precedence
  (OR > W/n, PRE/n > AND > AND NOT), field-code stripping, wildcards and true word-proximity.
* **ML model** = TF-IDF + Logistic Regression: best macro-F1 in the comparison (val 0.685, test 0.614;
  micro-F1 0.727) versus SVM, XGBoost, CatBoost and Random Forest. Weak on SDG 1, 14, 15 (few examples),
  and SDG 17 was never in the training data - the app says so instead of guessing.
* **Suggestions** = the guardrailed logic from `sdg_keywords.py`, tightened: a phrase is only proposed if ALL of
  its key words are already in the manuscript AND already sit close together (a re-wording, not a new claim).
  A checkbox in the sidebar (`--allow-new-words` on the CLI) also shows riskier phrases needing one new word.

## Setup
```bash
pip install -r requirements.txt
# 1. put Elsevier's 17 .txt files in ./sdg_queries/
# 2. train the ML model from your master sheet (sheet "Master Sheet", columns Title/Abstract/Keywords/SDG ...)
#    --negatives adds papers Elsevier did NOT tag as explicit "no SDG" examples
python3 train_model.py --xlsx Master_Publications_SDG_Mapping.xlsx --negatives Unmatched_No_SDG_Publications.xlsx
# 3. run
python3 -m streamlit run app.py
# streamlit run app.py
# or command line:
python sdg_pipeline.py --title "..." --abstract "..." --keywords "a, b, c"
```

## Deploying publicly (Streamlit Community Cloud)
Push the folder to GitHub with `sdg_queries/*.txt` and `model/sdg_model.pkl` included, then deploy `app.py`.
* Pin `scikit-learn==<version printed by train_model.py>` in `requirements.txt`.
* Check the licence of the Elsevier query files before publishing them in a public repo.

## How well does the Boolean matcher reproduce Elsevier's own tags?
Measured on random samples of your master sheet (600 tagged + 600 untagged papers):

| | recall vs Elsevier's tags | precision | untagged papers with a spurious exact match |
|---|---|---|---|
| first rebuild (unquoted words dropped, fields ignored) | 0.97 | 0.81 | 9.3% |
| **current** (unquoted words = terms, field-aware) | 0.96 | **0.89** | **4.3%** |

Remaining gap: no lemmatisation, the sheet's "Keywords" merge author + index keywords, and SDG 17 has no
Elsevier tags in the sheet to compare against.

## Training with "no SDG" papers
Adding ~4,500 Elsevier-untagged papers as negatives cut false flags on held-out untagged papers from 55% to 18%
(test macro-F1 on tagged papers 0.614 -> 0.579). It does NOT remove label artifacts: 61% of wireless-sensor-network
papers in the sheet are tagged SDG 7 by Elsevier, so the model learned that association too.
ML agreement is therefore not independent evidence - see the caution on generic clauses below.

## Known limits
* No lemmatisation: Scopus also matches plurals/inflections of loose phrases; this matches the wording as written
  (plus explicit `*` wildcards). Proximity is counted in words, not sentences.
* The ML labels come from Elsevier's own mapping, so ML agreement is a consistency check, not independent proof.
* Suggestions test *one added phrase at a time*; some SDG queries need several phrases together.
* Decision support only - adopt a suggested phrase only if it is true of your work.
