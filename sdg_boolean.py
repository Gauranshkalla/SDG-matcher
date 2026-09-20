"""
sdg_boolean.py  -  Phase 1: exact Boolean matcher for Elsevier's SDG queries
============================================================================

Rebuilds the "final" matcher from the SDG project (the tree-based version that
reports [EXACT MATCH]) as an importable module.

How it works
------------
1. Each Elsevier query file (a Scopus Boolean string) is tokenised and parsed
   into a real expression tree (TERM / AND / OR / NOT / PROX nodes) using
   Scopus's own precedence:  OR  >  W/n, PRE/n  >  AND  >  AND NOT.
   Field wrappers such as TITLE-ABS-KEY( ... ) are dropped (all manuscript text
   is searched together); non-text fields such as SRCID(...) are skipped.
2. The manuscript (title + abstract + keywords) is indexed as word n-grams.
3. The tree is evaluated against that index.  True  =>  EXACT MATCH for that SDG.
4. A weaker "diagnostic" overlap score (title x3, keywords x2, abstract x1,
   exclusion hits x -3) is also computed.  It is only a tie-breaker and is NOT
   evidence of a match on its own (that flat-term idea was the original bug).

Field-aware matching
--------------------
Elsevier's queries wrap terms in field codes.  Each term is only searched in its own
field of the manuscript:  TITLE-ABS(...) -> title+abstract,  AUTHKEY(...) -> keywords,
TITLE(...) -> title,  ABS(...) -> abstract,  TITLE-ABS-KEY(...)/unwrapped -> all three.
Quoted "..." / braced {...} phrases AND unquoted words (e.g. TITLE-ABS-KEY(climate),
agricultur*) are all search terms.

Known approximations
--------------------
* No lemmatisation: Scopus also matches plurals/inflections of loose phrases;
  this matcher only matches the wording as written (+ explicit * wildcards).
* Proximity W/n and PRE/n are implemented on word positions, not sentences.
* Index terms are not modelled: the "keywords" field is whatever keywords you supply.
"""

import bisect
import glob
import os
import re
import sys
import threading
from collections import defaultdict, namedtuple

SDG_NAMES = {
    1: "No Poverty", 2: "Zero Hunger", 3: "Good Health and Well-being",
    4: "Quality Education", 5: "Gender Equality", 6: "Clean Water and Sanitation",
    7: "Affordable and Clean Energy", 8: "Decent Work and Economic Growth",
    9: "Industry, Innovation and Infrastructure", 10: "Reduced Inequalities",
    11: "Sustainable Cities and Communities",
    12: "Responsible Consumption and Production", 13: "Climate Action",
    14: "Life Below Water", 15: "Life on Land",
    16: "Peace, Justice and Strong Institutions", 17: "Partnerships for the Goals",
}

SENT = "\u00a7\u00a7"          # sentinel word: phrases never match across it
MAXK = 10                       # longest phrase (in words) served by the n-gram index

Term = namedtuple("Term", ["text", "words", "kind"])   # kind: exact | suffix | general

# manuscript segment kinds: t = title, a = abstract, k = keyword
FIELD_KINDS = {"tak": set("tak"), "ta": set("ta"), "t": {"t"}, "a": {"a"}, "k": {"k"}}
FIELD_CODES = {
    "TITLE-ABS-KEY": "tak", "TITLE-ABS-KEY-AUTH": "tak", "ALL": "tak",
    "TITLE-ABS": "ta", "TITLE": "t", "ABS": "a",
    "AUTHKEY": "k", "KEY": "k", "INDEXTERMS": "k",
}

# --------------------------------------------------------------------------
# Normalisation & terms
# --------------------------------------------------------------------------

def norm_text(s):
    """Lower-case, strip punctuation/hyphens -> single-space separated words."""
    return re.sub(r"[\W_]+", " ", (s or "").lower()).strip()


def norm_term(s):
    """Like norm_text but keeps the Scopus wildcards * and ?"""
    s = (s or "").lower().replace("_", " ")
    return re.sub(r"[^\w*?]+", " ", s).strip()


_TERM_INTERN = {}


def make_term(raw):
    t = norm_term(raw)
    if not t or not re.search(r"[^\s*?]", t):
        return None
    if t in _TERM_INTERN:
        return _TERM_INTERN[t]
    words = tuple(t.split())
    if "*" not in t and "?" not in t and len(words) <= MAXK:
        kind = "exact"
    elif ("?" not in t and t.count("*") == 1 and t.endswith("*")
          and len(words[-1]) > 1 and len(words) <= MAXK):
        kind = "suffix"
    else:
        kind = "general"
    term = Term(t, words, kind)
    _TERM_INTERN[t] = term
    return term


def run_in_big_stack(fn, *args, **kwargs):
    """Run fn in a thread with a large stack + recursion limit (deep queries)."""
    box = {}

    def target():
        old = sys.getrecursionlimit()
        sys.setrecursionlimit(200000)
        try:
            box["v"] = fn(*args, **kwargs)
        except BaseException as e:            # noqa: BLE001
            box["e"] = e
        finally:
            sys.setrecursionlimit(old)

    old_size = threading.stack_size()
    try:
        threading.stack_size(256 * 1024 * 1024)
    except (ValueError, RuntimeError):
        pass
    th = threading.Thread(target=target)
    th.start()
    try:
        threading.stack_size(old_size)
    except (ValueError, RuntimeError):
        pass
    th.join()
    if "e" in box:
        raise box["e"]
    return box["v"]


def big_stack(fn):
    """Decorator: always run fn in the large-stack thread (deeply nested queries)."""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        return run_in_big_stack(fn, *a, **kw)
    return wrapper


# --------------------------------------------------------------------------
# Tokeniser
# --------------------------------------------------------------------------

_PROX_RE = re.compile(r"^(W|PRE)/(\d+)$", re.I)
_BARE = re.compile(r'[^\s(){}"]+')
_NOT_AFTER = re.compile(r"\s+NOT(?![\w])", re.I)
_FIELD_WORD = re.compile(r"[A-Za-z][A-Za-z\-]*$")


def _skip_group(text, j):
    """text[j] == '('.  Return index just after the matching ')' (quote-aware)."""
    depth, n = 0, len(text)
    while j < n:
        c = text[j]
        if c == '"':
            k = text.find('"', j + 1)
            j = n if k == -1 else k
        elif c == "{":
            k = text.find("}", j + 1)
            j = n if k == -1 else k
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return n


_BARE_TERM = re.compile(r"^[A-Za-z0-9][\w\-*?]*$")


def _tokenize(text):
    toks, i, n = [], 0, len(text)
    depth, fstack, pending = 0, [], None      # fstack: [(depth_at_open, field)]

    def cur_field():
        return fstack[-1][1] if fstack else "tak"

    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c in '"{':
            close = '"' if c == '"' else "}"
            j = text.find(close, i + 1)
            if j == -1:
                break
            term = make_term(text[i + 1:j])
            i = j + 1
            if term:
                toks.append(("TERM", term, cur_field()))
        elif c == "(":
            depth += 1
            if pending:
                fstack.append((depth, pending))
                pending = None
            toks.append(("LP",)); i += 1
        elif c == ")":
            if fstack and fstack[-1][0] == depth:
                fstack.pop()
            depth = max(0, depth - 1)
            toks.append(("RP",)); i += 1
        else:
            m = _BARE.match(text, i)
            word, i = m.group(), m.end()
            u = word.upper()
            if u == "AND":
                m2 = _NOT_AFTER.match(text, i)
                if m2:
                    toks.append(("ANDNOT",)); i = m2.end()
                else:
                    toks.append(("AND",))
            elif u == "OR":
                toks.append(("OR",))
            elif u == "NOT":
                toks.append(("NOT",))
            else:
                pm = _PROX_RE.match(word)
                if pm:
                    toks.append(("PROX", int(pm.group(2)), pm.group(1).upper() == "PRE"))
                elif _FIELD_WORD.match(word) and text[i:].lstrip().startswith("("):
                    if u in FIELD_CODES:                   # text field: scope the group
                        pending = FIELD_CODES[u]
                    else:                                  # SRCID(...), PUBYEAR(...), ...
                        i = _skip_group(text, text.index("(", i))
                elif _BARE_TERM.match(word) and not word.isdigit():
                    term = make_term(word)                 # unquoted search word
                    if term:
                        toks.append(("TERM", term, cur_field()))
                # anything else (bare numbers, stray symbols) is ignored
    return toks


# --------------------------------------------------------------------------
# Parser  (precedence: OR > PROX > AND > AND NOT)
# --------------------------------------------------------------------------

def _make(kind, kids):
    flat = []
    for k in kids:
        if k is None:
            continue
        if k[0] == kind:
            flat.extend(k[1])
        else:
            flat.append(k)
    if not flat:
        return None
    if len(flat) == 1:
        return flat[0]
    return (kind, tuple(flat))


class _Parser:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def parse(self):
        nodes = []
        while self.i < len(self.t):
            start = self.i
            nodes.append(self.and_not())
            if self.peek() is not None and self.peek()[0] == "RP":
                self.i += 1                       # stray ')'
            elif self.i == start:
                self.i += 1
        return _make("AND", nodes)

    def and_not(self):
        left = self.and_()
        negs = []
        while self.peek() is not None and self.peek()[0] in ("ANDNOT", "NOT"):
            self.i += 1
            right = self.and_()
            if right is not None:
                negs.append(("NOT", right))
        if left is None:
            return None
        return _make("AND", [left] + negs)

    def and_(self):
        kids = [self.prox()]
        while True:
            p = self.peek()
            if p is None:
                break
            if p[0] == "AND":
                self.i += 1
                kids.append(self.prox())
            elif p[0] in ("TERM", "LP"):          # implicit AND (Scopus joins adjacent words with AND)
                kids.append(self.prox())
            else:
                break
        return _make("AND", kids)

    def prox(self):
        left = self.or_()
        p = self.peek()
        while p is not None and p[0] == "PROX":
            self.i += 1
            right = self.or_()
            if left is not None and right is not None:
                left = ("PROX", left, right, p[1], p[2])
            p = self.peek()
        return left

    def or_(self):
        kids = [self.atom()]
        while self.peek() is not None and self.peek()[0] == "OR":
            self.i += 1
            kids.append(self.atom())
        return _make("OR", kids)

    def atom(self):
        p = self.peek()
        if p is None:
            return None
        if p[0] == "TERM":
            self.i += 1
            return ("TERM", p[1], p[2])
        if p[0] == "LP":
            self.i += 1
            nodes = []
            while True:
                start = self.i
                nodes.append(self.and_not())
                q = self.peek()
                if q is None:
                    break
                if q[0] == "RP":
                    self.i += 1
                    break
                if self.i == start:
                    self.i += 1                   # skip a stray token
            return _make("AND", nodes)
        if p[0] == "NOT":
            self.i += 1
            inner = self.atom()
            return ("NOT", inner) if inner is not None else None
        return None


@big_stack
def build_query_tree(raw_text):
    return _Parser(_tokenize(raw_text)).parse()


def collect_terms(node, include, exclude, negated=False):
    if node is None:
        return
    k = node[0]
    if k == "TERM":
        (exclude if negated else include).add(node[1])
    elif k == "NOT":
        collect_terms(node[1], include, exclude, not negated)
    elif k in ("AND", "OR"):
        for c in node[1]:
            collect_terms(c, include, exclude, negated)
    elif k == "PROX":
        collect_terms(node[1], include, exclude, negated)
        collect_terms(node[2], include, exclude, negated)


def _leaf_terms(node, out):
    if node is None:
        return out
    k = node[0]
    if k == "TERM":
        out.append((node[1], node[2]))
    elif k in ("AND", "OR"):
        for c in node[1]:
            _leaf_terms(c, out)
    elif k == "PROX":
        _leaf_terms(node[1], out); _leaf_terms(node[2], out)
    return out


# --------------------------------------------------------------------------
# Text index + tree evaluation
# --------------------------------------------------------------------------

class TextIndex:
    """Word n-gram index of a manuscript (fields kept apart by a sentinel)."""

    def __init__(self, segments=None, words=None, kinds=None):
        """segments: list of (kind, text) with kind in {'t','a','k'}."""
        if words is None:
            words, kinds = [], []
            for kind, seg in segments or []:
                w = norm_text(seg).split()
                if w:
                    if words:
                        words.append(SENT); kinds.append("-")
                    words.extend(w); kinds.extend([kind] * len(w))
        self.words = words
        self.kinds = kinds
        self.ngrams = [None] + [defaultdict(list) for _ in range(MAXK)]
        self.prefix = [None] + [defaultdict(list) for _ in range(MAXK)]
        n = len(words)
        for k in range(1, MAXK + 1):
            for s in range(n - k + 1):
                win = words[s:s + k]
                if SENT in win:
                    continue
                self.ngrams[k][" ".join(win)].append(s)
                self.prefix[k][" ".join(win[:-1])].append((s, win[-1]))
        self._cache = {}
        self._joined = None

    def _regex_positions(self, term):
        if self._joined is None:
            self._joined = " ".join(self.words)
            self._starts, pos = [], 0
            for w in self.words:
                self._starts.append(pos)
                pos += len(w) + 1
        pat = "".join(r"\w*" if ch == "*" else r"\w" if ch == "?" else re.escape(ch)
                      for ch in term.text)
        rx = re.compile(r"(?<!\w)" + pat + r"(?!\w)")
        out = []
        for m in rx.finditer(self._joined):
            s = bisect.bisect_right(self._starts, m.start()) - 1
            e = bisect.bisect_right(self._starts, max(m.end() - 1, m.start())) - 1
            out.append((s, e))
        return out

    def positions(self, term):
        r = self._cache.get(term)
        if r is not None:
            return r
        k = len(term.words)
        if term.kind == "exact":
            r = [(s, s + k - 1) for s in self.ngrams[k].get(term.text, ())]
        elif term.kind == "suffix":
            stem = term.words[-1][:-1]
            pre = " ".join(term.words[:-1])
            r = [(s, s + k - 1) for s, lw in self.prefix[k].get(pre, ())
                 if lw.startswith(stem)]
        else:
            r = self._regex_positions(term)
        self._cache[term] = r
        return r

    def positions_in(self, term, field="tak"):
        ok = FIELD_KINDS[field]
        return [p for p in self.positions(term) if self.kinds[p[0]] in ok]

    def has(self, term, field="tak"):
        if field == "tak":
            return bool(self.positions(term))
        return bool(self.positions_in(term, field))

    def near(self, a_terms, b_terms, n, ordered):
        """a_terms / b_terms: lists of (term, field)."""
        pa = [p for t, f in a_terms for p in self.positions_in(t, f)]
        pb = [p for t, f in b_terms for p in self.positions_in(t, f)]
        for a0, a1 in pa:
            for b0, b1 in pb:
                if b0 > a1:
                    gap, lo, hi = b0 - a1 - 1, a1 + 1, b0
                elif not ordered and a0 > b1:
                    gap, lo, hi = a0 - b1 - 1, b1 + 1, a0
                else:
                    continue
                if gap <= n and SENT not in self.words[lo:hi]:
                    return True
        return False

    def without(self, term):
        """A copy of this index with every occurrence of `term` blanked out."""
        w = list(self.words)
        for s, e in self.positions(term):
            for i in range(s, e + 1):
                w[i] = SENT
        return TextIndex(words=w, kinds=list(self.kinds))


def eval_node(node, idx):
    if node is None:
        return False
    k = node[0]
    if k == "TERM":
        return idx.has(node[1], node[2])
    if k == "NOT":
        return not eval_node(node[1], idx)
    if k == "AND":
        return all(eval_node(c, idx) for c in node[1])
    if k == "OR":
        return any(eval_node(c, idx) for c in node[1])
    if k == "PROX":
        return idx.near(_leaf_terms(node[1], []), _leaf_terms(node[2], []),
                        node[3], node[4])
    return False


def witness(node, idx):
    """One satisfying set of terms for a TRUE tree (None if the node is false).
    For OR nodes the most specific (longest) satisfied branch is reported."""
    if node is None:
        return None
    k = node[0]
    if k == "TERM":
        return [node[1].text] if idx.has(node[1], node[2]) else None
    if k == "NOT":
        return [] if not eval_node(node[1], idx) else None
    if k == "AND":
        out = []
        for c in node[1]:
            w = witness(c, idx)
            if w is None:
                return None
            out.extend(w)
        return out
    if k == "OR":
        best = None
        for c in node[1]:
            w = witness(c, idx)
            if w is not None and (best is None or
                                  sum(len(x.split()) for x in w) > sum(len(x.split()) for x in best)):
                best = w
        return best
    if k == "PROX":
        if eval_node(node, idx):
            return [t.text for t, f in _leaf_terms(node, []) if idx.has(t, f)]
    return None


# --------------------------------------------------------------------------
# Loading query files
# --------------------------------------------------------------------------

def detect_sdg_number(filename):
    base = os.path.basename(filename)
    m = re.search(r"sdg[\s_\-]*0*(\d{1,2})(?!\d)", base, re.I)
    cands = [m.group(1)] if m else re.findall(r"(?<!\d)(\d{1,2})(?!\d)", base)
    for c in cands:
        if 1 <= int(c) <= 17:
            return int(c)
    return None


@big_stack
def load_sdg_queries(query_dir):
    files = sorted(glob.glob(os.path.join(query_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(
            f"No .txt files found in {query_dir!r}. Download Elsevier's 17 SDG "
            "query files and put them in that folder.")
    data = {}
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            tree = build_query_tree(fh.read())
        inc, exc = set(), set()
        collect_terms(tree, inc, exc)
        key = detect_sdg_number(f)
        if key is None or key in data:
            key = os.path.splitext(os.path.basename(f))[0]
        data[key] = {"file": f, "tree": tree, "include": inc, "exclude": exc}
    return data


# --------------------------------------------------------------------------
# Manuscript -> per-SDG result
# --------------------------------------------------------------------------

def manuscript_segments(title, abstract, keywords):
    kws = [k for k in re.split(r"[;,\n]+", keywords or "") if k.strip()]
    return [("t", title or ""), ("a", abstract or "")] + [("k", k) for k in kws]


@big_stack
def analyze_boolean(sdg_data, title, abstract, keywords,
                    title_w=3, kw_w=2, abs_w=1, exclude_penalty=3):
    """Return {sdg: {exact, score, matched_include, matched_exclude}}."""
    idx = TextIndex(manuscript_segments(title, abstract, keywords))

    def weight(term):
        if idx.has(term, "t"):
            return title_w
        if idx.has(term, "k"):
            return kw_w
        return abs_w if idx.has(term, "a") else 0

    out = {}
    for key, d in sdg_data.items():
        inc = [(weight(t), t) for t in d["include"]]
        inc = [(w, t) for w, t in inc if w]
        exc = [(weight(t), t) for t in d["exclude"]]
        exc = [(w, t) for w, t in exc if w]
        score = sum(w for w, _ in inc) - exclude_penalty * sum(w for w, _ in exc)
        inc.sort(key=lambda x: (-x[0], -len(x[1].words), x[1].text))
        exact = bool(eval_node(d["tree"], idx))
        out[key] = {
            "exact": exact,
            "witness": (witness(d["tree"], idx) or []) if exact else [],
            "score": score,
            "matched_include": [t.text for _, t in inc],
            "matched_exclude": sorted(t.text for _, t in exc),
        }
    return out


def rank(results):
    return sorted(results.items(),
                  key=lambda kv: (not kv[1]["exact"], -kv[1]["score"], str(kv[0])))
