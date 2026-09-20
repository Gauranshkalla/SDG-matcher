"""
SDG Word Finder - Streamlit app
  streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st

import sdg_boolean as sb
import sdg_ml as ml
import sdg_pipeline as pl

st.set_page_config(page_title="SDG Word Finder", page_icon="🎯", layout="wide")


@st.cache_resource(show_spinner="Parsing Elsevier SDG query files (first run only)...")
def get_engine(query_dir):
    return sb.load_sdg_queries(query_dir)


@st.cache_resource(show_spinner=False)
def get_model(path):
    return ml.load_model(path)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    query_dir = st.text_input("Folder with Elsevier SDG .txt query files", pl.DEFAULT_QUERIES)
    model_path = st.text_input("ML model file", pl.DEFAULT_MODEL)
    ml_threshold = st.slider("ML agreement threshold", 0.1, 0.9, 0.5, 0.05,
                             help="Probability at which the ML model counts as 'agreeing' with an SDG.")
    suggest_floor = st.slider("Suggest wording for SDGs with ML probability above", 0.1, 0.9, 0.3, 0.05,
                              help="Only used for SDGs the Boolean matcher did not match.")
    max_terms = st.slider("Max suggestions per SDG", 3, 15, 8)
    allow_new = st.checkbox("Also suggest phrases that need one new word (riskier)", value=False,
                            help="Default: only re-wordings using words already close together in your text.")
    show_all = st.checkbox("Show all 17 SDGs in the table", value=False)

# ------------------------------------------------------------------ header
st.title("🎯 SDG Word Finder")
st.write(
    "Enter your manuscript's **title, abstract and keywords**. The tool checks them against "
    "Elsevier's SDG search queries (exact Boolean match), cross-checks with a machine-learning model, "
    "and - if the exact match fails - suggests small wording changes **you** can accept or ignore."
)

# ------------------------------------------------------------------ loading
try:
    engine = get_engine(query_dir)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

art, warn = None, None
if os.path.exists(model_path):
    art, warn = get_model(model_path)
    if warn:
        st.warning(warn)
else:
    st.info("No ML model found - running the Boolean matcher only. "
            "Train one with `python train_model.py --xlsx <master sheet>`.")

missing = [n for n in range(1, 18) if n not in engine]
if missing:
    st.warning(f"No query file detected for SDG {missing}. Check the file names contain the SDG number.")

# ------------------------------------------------------------------ inputs
title = st.text_input("Title")
abstract = st.text_area("Abstract", height=220)
keywords = st.text_input("Keywords (comma-separated)")

if st.button("Find matching SDGs", type="primary"):
    if not (title.strip() or abstract.strip() or keywords.strip()):
        st.warning("Please enter at least a title, abstract or keywords.")
        st.stop()
    with st.spinner("Analysing..."):
        result = pl.analyze(engine, art, title, abstract, keywords,
                            ml_threshold=ml_threshold, suggest_floor=suggest_floor,
                            max_terms=max_terms, allow_new_words=allow_new)
    st.session_state["result"] = result

result = st.session_state.get("result")
if not result:
    st.stop()

rows = result["rows"]

# ------------------------------------------------------------------ summary
st.subheader("Result")
confirmed = [r for r in rows if r["status"] == "confirmed"]
bool_only = [r for r in rows if r["status"] == "boolean_only"]
ml_only = [r for r in rows if r["status"] == "ml_only"]

if confirmed:
    st.success("Confirmed (Boolean exact match + ML agrees): " +
               ", ".join(f"SDG {r['sdg']}" for r in confirmed))
if bool_only:
    st.info("Boolean exact match (ML not conclusive): " +
            ", ".join(f"SDG {r['sdg']}" for r in bool_only))
if ml_only:
    st.warning("ML supports, but Elsevier's exact query is not satisfied: " +
               ", ".join(f"SDG {r['sdg']}" for r in ml_only) +
               (" - see wording suggestions below." if result["suggestions"] else "."))
if not (confirmed or bool_only or ml_only):
    st.info("No SDG matched. That is a legitimate outcome - not every paper maps to an SDG.")

table = []
for r in sorted(rows, key=lambda r: (r["status"] == "none", -(r["prob"] or 0), -r["score"])):
    if not show_all and r["status"] == "none" and (r["prob"] or 0) < 0.2:
        continue
    table.append({
        "SDG": r["sdg"], "Goal": r["name"],
        "Boolean (Elsevier)": "✅ exact" if r["exact"] else ("—" if r["has_query"] else "no query file"),
        "ML probability": None if r["prob"] is None else round(r["prob"], 2),
        "Verdict": pl.STATUS_TEXT[r["status"]],
        "Comment": r["note"],
    })
if table:
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")

for r in rows:
    if r["exact"]:
        with st.expander(f"Why SDG {r['sdg']} matched (the Elsevier clause your text satisfied)"):
            st.write(" **AND** ".join(f"“{t}”" for t in r["why"]) or "-")
            st.caption("Elsevier's query is broad. If this SDG does not describe what your paper is really about, "
                       "you do not have to claim it - especially when the ML probability is low.")

# ------------------------------------------------------------------ suggestions
if result["suggestions"]:
    st.subheader("Wording suggestions (your decision)")
    st.caption("Shown only because no SDG matched Elsevier's queries exactly.")
    st.caption(
        "These are proposals, not instructions. Adopt a phrase **only if it accurately describes work "
        "your manuscript really does**. Suggesting terms for concepts the paper does not address would "
        "be SDG-washing, so phrases whose key words are absent from your text are never offered.")
    for n, s in result["suggestions"].items():
        with st.expander(f"SDG {n}: {sb.SDG_NAMES.get(n, '')}", expanded=True):
            if s["exact"]:
                st.success("Already an exact match.")
                continue
            for b in s["blocking_exclusions"]:
                if b["unblocks_match"]:
                    st.warning(f"Your text contains **“{b['term']}”**, which Elsevier's query excludes. "
                               "Rewording or removing it would unblock an exact match "
                               "(only if that is accurate for your paper).")
                else:
                    st.caption(f"Excluded word present in your text: “{b['term']}”")
            if s["candidates"]:
                df = pd.DataFrame([{
                    "Standard phrase": c["term"],
                    "One edit gives exact match?": "✅ yes" if c["single_edit_match"] else "partly",
                    "Closest wording in your text": c["closest_wording"],
                    "Words not in your text": ", ".join(c["missing"]),
                } for c in s["candidates"]])
                st.dataframe(df, hide_index=True, width="stretch")
            elif not s["blocking_exclusions"]:
                st.info("No phrase passed the relevance guardrail - this SDG is probably not a fit "
                        "for your paper, and that is a valid answer.")
            if s["already_present"]:
                st.caption("Standard terms your text already uses: " +
                           ", ".join(s["already_present"][:15]))

st.divider()
st.caption(
    "Decision support, not ground truth. The Boolean check follows Elsevier's published SDG queries; "
    "the ML model was trained on Elsevier-derived labels, so it is a consistency cross-check rather than "
    "an independent authority. It has little training data for some SDGs (e.g. 1, 14, 15) and none for SDG 17.")
