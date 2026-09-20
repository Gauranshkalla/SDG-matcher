"""
sdg_suggest.py  -  "What could I alter to get an exact match?"
================================================================

Builds on sdg_boolean.  For an SDG whose Elsevier query is NOT satisfied it
proposes small wording changes.  It never forces anything: every suggestion is
for the AUTHOR to accept or reject on accuracy grounds.

Relevance guardrail (carried over from the earlier sdg_keywords.py)
-------------------------------------------------------------------
A standard SDG phrase is only suggested if (a) ALL of its distinctive
content words (stop-words removed) already appear in the manuscript AND (b) those
words already sit close together in the text - i.e. it is a re-wording, not a new claim.
(allow_new_words=True additionally shows phrases that need one or two NEW words, as long
as more than half of their key words are already present - riskier, off by default.)  So "add biomass / medicare / women" to a wireless-networking paper is
never proposed - that would be SDG-washing.

Two extra checks make the advice more useful
--------------------------------------------
* single_edit_match : simulate adopting the phrase and re-run the Boolean tree.
                      True  => that one phrase is enough for an EXACT MATCH.
* blocking exclusions: words in your text that Elsevier's query EXCLUDES.  If
                      removing/rewording one would unblock an exact match, it is
                      flagged.
"""

import re
from functools import lru_cache

import sdg_boolean as sb

STOPWORDS = {
    "a", "an", "the", "and", "or", "not", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "is", "are", "be", "been", "was", "were", "it",
    "its", "this", "that", "these", "those", "all", "any", "each", "per",
    "into", "over", "under", "between", "within", "via", "using", "use", "used",
    "new", "non", "other", "more", "most", "such", "than", "then", "there",
    "based", "level", "levels", "system", "systems", "data", "study", "studies",
    "paper", "approach", "method", "methods", "result", "results", "analysis",
    "model", "models", "framework", "research", "high", "low", "large", "small",
}


@lru_cache(maxsize=None)
def content_words(phrase):
    words = re.findall(r"[a-z][a-z]*", phrase.lower().replace("*", ""))
    return tuple(w for w in words if w not in STOPWORDS and len(w) > 2)


def manuscript_tokens(text):
    return set(sb.norm_text(text).split())


def score_candidate(term, tokens, min_coverage=0.5, allow_new_words=False):
    """(coverage, n_present, n_total, missing) or None if the guardrail rejects.

    Guardrail: MORE than half of the phrase's content words must already be in the
    manuscript (so a 2-word phrase needs both words, a 3-word phrase needs two).
    """
    cw = content_words(term.text)
    if not cw:
        return None
    present = [w for w in cw if w in tokens or
               (term.kind == "suffix" and w == cw[-1] and any(t.startswith(w) for t in tokens))]
    if not present:
        return None
    coverage = len(present) / len(cw)
    if coverage <= min_coverage:
        return None
    missing = [w for w in cw if w not in present]
    if missing and not allow_new_words:      # default: pure re-wording only
        return None
    return coverage, len(present), len(cw), missing


def closest_window(term, words, slack=3):
    """
    Shortest stretch of the manuscript that contains ALL of the phrase's content
    words that are already present, provided they sit within (count + slack) words
    of each other.  Returns the stretch as text, or None if the words are scattered
    (then the suggestion would be a new claim, not a re-wording).
    """
    cw = set(content_words(term.text))
    present = {w for w in cw if w in words or
               (term.kind == "suffix" and any(x.startswith(w) for x in words))}
    if not present:
        return None
    n, best = len(words), None
    for size in range(len(present), len(present) + slack + 1):
        for st in range(0, max(1, n - size + 1)):
            win = words[st:st + size]
            if sb.SENT in win:
                continue
            got = {w for w in present if w in win or
                   (term.kind == "suffix" and any(x.startswith(w) for x in win))}
            if got == present:
                return " ".join(win)
        # try next size
    return None


def _adopt_text(term):
    return term.text.replace("*", "").replace("?", "").strip()


def suggest_for_sdg(entry, title, abstract, keywords, max_terms=8,
                    min_coverage=0.5, scan_limit=60, allow_new_words=False):
    """
    entry: one value of sdg_boolean.load_sdg_queries().
    Returns dict(already_present, candidates, blocking_exclusions).
    """
    return sb.run_in_big_stack(_suggest, entry, title, abstract, keywords,
                               max_terms, min_coverage, scan_limit, allow_new_words)


def _suggest(entry, title, abstract, keywords, max_terms, min_coverage, scan_limit,
             allow_new_words=False):
    tree = entry["tree"]
    segs = sb.manuscript_segments(title, abstract, keywords)
    idx = sb.TextIndex(segs)
    tokens = set(idx.words) - {sb.SENT}

    already = sorted(t.text for t in entry["include"] if idx.has(t))

    # --- terms that could be adopted -------------------------------------
    scored = []
    for t in entry["include"]:
        if idx.has(t):
            continue
        sc = score_candidate(t, tokens, min_coverage, allow_new_words)
        if sc is None:
            continue
        cov, n_p, n_t, missing = sc
        snippet = closest_window(t, idx.words)
        if snippet is None:              # words are scattered -> would be a new claim
            continue
        scored.append((cov, len(missing), -len(t.words), t, n_p, n_t, missing, snippet))
    scored.sort(key=lambda x: (-x[0], x[1], x[2], x[3].text))

    already_exact = sb.eval_node(tree, idx)
    cands = []
    for cov, _, _, t, n_p, n_t, missing, snippet in scored[:scan_limit]:
        flips = False
        if not already_exact and t.kind != "general":
            flips = sb.eval_node(tree, sb.TextIndex(segs + [("a", _adopt_text(t))]))
        cands.append({
            "term": t.text,
            "coverage": cov,
            "n_present": n_p,
            "n_total": n_t,
            "missing": missing,
            "closest_wording": snippet,
            "single_edit_match": flips,
        })
    cands.sort(key=lambda c: (not c["single_edit_match"], -c["coverage"],
                              len(c["missing"]), -c["n_total"], c["term"]))
    cands = [] if already_exact else cands[:max_terms]

    # --- excluded words that are in the text -----------------------------
    blocking = []
    if not already_exact:
        for t in entry["exclude"]:
            if idx.has(t) and sb.eval_node(tree, idx.without(t)):
                # only report excluded words whose removal would really unblock a match
                blocking.append({"term": t.text, "unblocks_match": True})
        blocking.sort(key=lambda b: (not b["unblocks_match"], b["term"]))

    return {"already_present": already, "candidates": cands,
            "blocking_exclusions": blocking[:10], "exact": already_exact}
