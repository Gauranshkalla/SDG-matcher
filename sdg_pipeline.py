"""
sdg_pipeline.py  -  glue: Boolean matcher + ML cross-check + wording suggestions
==============================================================================

    Input : title, abstract, keywords
    Step 1: exact Boolean match against Elsevier's 17 SDG queries      (sdg_boolean)
    Step 2: ML probability per SDG as a cross-check                    (sdg_ml)
    Step 3: ONLY if NO SDG has an exact Boolean match: suggest small wording
            changes for the author to consider                          (sdg_suggest)
            (if any SDG matches exactly, no suggestions are produced at all)

Verdict logic (per SDG)
    confirmed     Boolean exact match  AND  ML probability >= threshold
    boolean_only  Boolean exact match, ML not convinced / not applicable
    ml_only       ML supports it, Boolean query NOT satisfied  -> suggestions
    none          neither

CLI:  python sdg_pipeline.py --title "..." --abstract "..." --keywords "a, b"
"""

import argparse
import os
import sys

import sdg_boolean as sb
import sdg_ml as ml
import sdg_suggest as ss

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QUERIES = os.path.join(BASE, "sdg_queries")
DEFAULT_MODEL = os.path.join(BASE, "model", "sdg_model.pkl")

STATUS_TEXT = {
    "confirmed": "Confirmed - Boolean exact match and ML agrees",
    "boolean_only": "Boolean exact match",
    "ml_only": "ML suggests it, Boolean query not satisfied",
    "none": "No match",
}


def combine(boolean, probs, art, ml_threshold=0.5):
    rows = []
    for n in range(1, 18):
        b = boolean.get(n)
        p = probs.get(n)
        exact = bool(b and b["exact"])
        ml_yes = p is not None and p >= ml_threshold
        rel = ml.reliability(art, n) if art is not None else "unavailable"
        if exact and ml_yes:
            status = "confirmed"
        elif exact:
            status = "boolean_only"
        elif ml_yes:
            status = "ml_only"
        else:
            status = "none"

        note = ""
        if status == "boolean_only":
            if rel == "unavailable":
                note = "ML model not loaded - Boolean result only."
            elif rel == "none":
                note = "ML was never trained on this SDG - rely on the Boolean result."
            elif rel == "limited":
                note = ("ML had few training examples for this SDG, so its low score is weak "
                        "evidence - check the matched clause and decide yourself.")
            else:
                note = "ML probability is below the threshold - worth a manual look."
        elif status == "ml_only":
            note = "Elsevier's exact query is not satisfied - see wording suggestions."
        rows.append({
            "sdg": n, "name": sb.SDG_NAMES[n], "exact": exact, "prob": p,
            "ml_yes": ml_yes, "reliability": rel, "status": status, "note": note,
            "has_query": b is not None,
            "score": b["score"] if b else 0,
            "matched": b["matched_include"] if b else [],
            "why": b["witness"] if b else [],
            "matched_exclude": b["matched_exclude"] if b else [],
        })
    return rows


def pick_targets(rows, suggest_floor=0.30, max_sdgs=3):
    """SDGs to produce wording suggestions for (only called when nothing matched exactly)."""
    have_ml = any(r["prob"] is not None for r in rows)
    if have_ml:
        ranked = sorted((r for r in rows if r["prob"] is not None), key=lambda r: -r["prob"])
        cand = [r for r in ranked if r["prob"] >= suggest_floor]
        if not cand:                       # nothing clears the floor: still try the closest ones
            cand = ranked[:2]
    else:                                  # no model: fall back to topical contact
        cand = sorted((r for r in rows if r["score"] > 0), key=lambda r: -r["score"])[:2]
    return [r["sdg"] for r in cand[:max_sdgs]]


def analyze(engine, art, title, abstract, keywords, ml_threshold=0.5,
            suggest_floor=0.30, max_sdgs=3, max_terms=8, allow_new_words=False):
    boolean = sb.analyze_boolean(engine, title, abstract, keywords)
    probs = ml.predict_proba(art, title, abstract, keywords) if art is not None else {}
    rows = combine(boolean, probs, art, ml_threshold)
    any_exact = any(r["exact"] for r in rows)
    suggestions = {}
    if any_exact:
        # An exact Boolean match exists -> no wording suggestions at all.
        for r in rows:
            if r["status"] == "ml_only":
                r["note"] = "ML supports it, but Elsevier's exact query is not satisfied."
    else:
        for n in pick_targets(rows, suggest_floor, max_sdgs):
            if n in engine:
                suggestions[n] = ss.suggest_for_sdg(engine[n], title, abstract, keywords,
                                                    max_terms=max_terms,
                                                    allow_new_words=allow_new_words)
    return {"rows": rows, "suggestions": suggestions, "any_exact": any_exact}


def format_report(result):
    out = []
    rows = sorted(result["rows"], key=lambda r: (r["status"] == "none",
                  -(r["prob"] or 0), -r["score"]))
    out.append("=" * 74)
    out.append("SDG MATCH  (Boolean matcher + ML cross-check)")
    out.append("=" * 74)
    shown = [r for r in rows if r["status"] != "none" or (r["prob"] or 0) >= 0.2]
    if not shown:
        out.append("No SDG matched. That is a legitimate result - not every paper maps to an SDG.")
    for r in shown:
        p = "n/a" if r["prob"] is None else f"{r['prob']:.2f}"
        out.append(f"SDG {r['sdg']:>2} {r['name']:<42} boolean={'YES' if r['exact'] else 'no ':<3} ML={p}")
        out.append(f"        -> {STATUS_TEXT[r['status']]}" + (f"  [{r['note']}]" if r["note"] else ""))
    if result.get("any_exact"):
        out.append("")
        out.append("(An exact Boolean match exists, so no wording suggestions are shown.)")
    for n, s in result["suggestions"].items():
        out.append("")
        out.append("-" * 74)
        out.append(f"WORDING SUGGESTIONS for SDG {n}: {sb.SDG_NAMES.get(n, '')}")
        out.append("(proposals only - adopt a phrase ONLY if it truthfully describes your work)")
        if s["exact"]:
            out.append("  Already an exact match - nothing to change.")
            continue
        for b in s["blocking_exclusions"]:
            tag = "removing/rewording it would unblock an exact match" if b["unblocks_match"] else "present in your text"
            out.append(f'  excluded word in your text: "{b["term"]}"  ({tag})')
        for c in s["candidates"]:
            flag = "ONE EDIT -> exact match" if c["single_edit_match"] else "helps, but not sufficient alone"
            out.append(f'  "{c["term"]}"  [{flag}]')
            if c["closest_wording"]:
                out.append(f'      closest wording now: "{c["closest_wording"]}"')
            if c["missing"]:
                out.append(f'      words not in your text: {", ".join(c["missing"])}')
        if not s["candidates"] and not s["blocking_exclusions"]:
            out.append("  Nothing passed the relevance guardrail - this SDG is probably not a fit.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", default=DEFAULT_QUERIES)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--title", default="")
    ap.add_argument("--abstract", default="")
    ap.add_argument("--keywords", default="")
    ap.add_argument("--ml-threshold", type=float, default=0.5)
    ap.add_argument("--allow-new-words", action="store_true",
                    help="also suggest phrases that need one new word (riskier)")
    a = ap.parse_args()
    if not (a.title or a.abstract or a.keywords):
        sys.exit("Provide --title / --abstract / --keywords.")
    engine = sb.load_sdg_queries(a.queries)
    art = None
    if os.path.exists(a.model):
        art, warn = ml.load_model(a.model)
        if warn:
            print("WARNING:", warn)
    else:
        print(f"(no ML model at {a.model} - running Boolean matcher only)")
    res = analyze(engine, art, a.title, a.abstract, a.keywords,
                  ml_threshold=a.ml_threshold,
                  allow_new_words=a.allow_new_words)
    print(format_report(res))


if __name__ == "__main__":
    main()
