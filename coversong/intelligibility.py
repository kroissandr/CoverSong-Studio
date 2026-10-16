"""How many words of the original survive in your vocal.

Whisper transcribes the original vocal and your converted vocal; a word of the original counts as kept when your
transcript has the same (or nearly the same) word within ±0.6 s. Interjections (oh, la, ooh…) are not counted.
"""
import difflib
import json
import os
import re
import subprocess

from . import ROOT
from .util import log

INTERJ = re.compile(r"^(o+h*|u+h*|a+h*|e+h*|h?m+|h?mm+|la|na|da|ba|doo+|woo+|w?hoa+|y?eah+h+|"
                    r"у+|о+|а+|э+|м+|ла|на|да|ля)$")
NEAR_S = 0.6
PHRASE_GAP = 0.7
PHRASE_MAX = 8.0  # long runs of singing are cut into ~8 s pieces, so one bad line can be repaired alone


def norm(word):
    return re.sub(r"[^\w]", "", word.lower().replace("’", "'").replace("'", ""))


def lexical(words):
    out = []
    for w in words:
        t = norm(w["word"])
        if t and not INTERJ.match(t):
            out.append(dict(w, t=t))
    return out


def asr(cfg, items):
    """items: [(wav, out_json, language or None)] -> list of word lists. One Whisper load for all files."""
    todo = [(i, o, lang) for i, o, lang in items]
    job = os.path.join(os.path.dirname(todo[0][1]), "_asr_job.json")
    with open(job, "w", encoding="utf-8") as f:
        json.dump([{"in": i, "out": o, "language": lang} for i, o, lang in todo], f, ensure_ascii=False)
    from .util import worker_python
    py, env = worker_python()
    mdir = os.path.normpath(os.path.join(cfg["paths"]["sep_models"], "..", "whisper"))
    r = subprocess.run([py, "-u", "-m", "coversong.asr_worker", "--job", job, mdir], cwd=ROOT, env=env)
    os.remove(job)
    if r.returncode != 0:
        raise RuntimeError("Word recognition (Whisper) failed — see the log above")
    res = []
    for _, o, _ in todo:
        with open(o, encoding="utf-8") as f:
            res.append(json.load(f))
    return res


def phrases(src_lex):
    """Group the original's words into phrases (gap > 0.7 s starts a new one)."""
    out = []
    for w in src_lex:
        if out and w["start"] - out[-1][-1]["end"] <= PHRASE_GAP and w["end"] - out[-1][0]["start"] <= PHRASE_MAX:
            out[-1].append(w)
        else:
            out.append([w])
    return out


def kept_flags(src_lex, cov_words, t0=0.0, t1=1e9):
    """For each original word: is it in your transcript nearby? (each of your words is used once)"""
    cov = [w for w in lexical(cov_words) if t0 - 1 <= w["start"] <= t1 + 1]
    used = [False] * len(cov)
    flags = []
    for s in src_lex:
        mid = (s["start"] + s["end"]) / 2
        best, bj = 0.0, -1
        for j, c in enumerate(cov):
            if used[j] or abs((c["start"] + c["end"]) / 2 - mid) > NEAR_S:
                continue
            r = 1.0 if c["t"] == s["t"] else difflib.SequenceMatcher(None, c["t"], s["t"]).ratio()
            if r > best:
                best, bj = r, j
        ok = best >= 0.75
        if ok:
            used[bj] = True
        flags.append(ok)
    return flags


def score(src_words, cov_words, t0=0.0, t1=1e9):
    src = [w for w in lexical(src_words) if t0 <= w["start"] <= t1]
    if not src:
        return None, 0
    fl = kept_flags(src, cov_words, t0, t1)
    return sum(fl) / len(fl), len(fl)


def per_phrase(src_words, cov_words):
    """[(start, end, kept_fraction, n_words, text)] for every phrase of the original."""
    out = []
    for ph in phrases(lexical(src_words)):
        fl = kept_flags(ph, cov_words, ph[0]["start"], ph[-1]["end"])
        out.append((ph[0]["start"], ph[-1]["end"], sum(fl) / len(fl), len(fl), " ".join(w["word"] for w in ph)))
    return out


def summary(rows):
    n = sum(r[3] for r in rows)
    kept = sum(r[2] * r[3] for r in rows)
    return (kept / n if n else None), n


def fmt_t(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def worst(rows, k=3):
    bad = sorted([r for r in rows if r[3] >= 3 and r[2] < 0.75], key=lambda r: r[2])[:k]
    return ", ".join(f"{fmt_t(r[0])} \"{r[4][:40]}\"" for r in sorted(bad)) or "none"


def log_summary(tag, rows):
    frac, n = summary(rows)
    if frac is None:
        log(f"{tag}: no words to check intelligibility")
        return None
    log(f"{tag}: intelligibility {frac:.0%} ({n} words of the original); weakest places: {worst(rows)}")
    return frac
