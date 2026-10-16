"""Several singers in one song: who sings when (for the "Several voices" tab).

1. The lead vocal is cut into sung lines (Whisper's words; parts without words by loudness).
2. Each line gets a voice fingerprint (WeSpeaker ResNet34, ONNX, CC-BY-4.0 — trained on speech, it still hears
   timbre in singing, but similar voices of one group can mix up; the tab lets you fix lines by hand).
3. Lines are grouped into voices (agglomerative clustering, cosine); the number of voices is estimated
   or given by you. Lines that sound like two voices at once (or like nobody — usually several people singing)
   are sung by the two closest voices together.
   With a song text tagged like on Genius ("[Verse 1: Karina]") the tags decide instead of the fingerprints.

Line labels: "0", "1", … one singer; "0+2" several singers together, each with their own model (that is what a
choir is); "mute" — removed. The backing vocal follows the same plan: at every moment it is sung by whoever sings
the lead then (see stages/convert.py, Voices).
Output: <song>/voices/voices.json + a short sample of each voice (voice_N.wav).
"""
import json
import os

import numpy as np

from . import ROOT
from .util import log

REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
ONNX = "voxceleb_resnet34_LM.onnx"
SR = 16000
MAX_VOICES = 16     # singers the tab can show (big groups: TWICE 9, SEVENTEEN 13)
AUTO_MAX = 8        # most voices to look for by timbre alone
MIN_EMB = 1.2      # s: shorter lines get the voice of their neighbour
FIT_SIM = 0.5      # cosine to the closest voice below this (and no clear winner) -> "fits nobody"
FIT_MARGIN = 0.1
TIE = 0.06         # equally close to two voices -> both sing
MIN_VOICE_S = 5.0  # automatically found "voices" shorter than this are not separate singers


# ---------- labels ----------
def members(lab):
    """'0+2' -> [0, 2]; '1' -> [1]; mute -> []."""
    if lab in ("mute", "chorus") or lab is None:  # ("chorus": from older voices.json)
        return []
    return [int(v) for v in str(lab).split("+")]


def show(lab):
    if lab in ("mute", "chorus"):
        return "—"
    return "+".join(str(v + 1) for v in members(lab))


def parse_label(text, k=None):
    """What the user typed in the table -> label (None if unreadable). "all" / "choir" (or the Russian "все" / "хор") = all k voices together."""
    t = str(text if text is not None else "").strip().lower().replace(",", "+").replace(" и ", "+").replace("&", "+")
    if t in ("хор", "х", "choir", "chorus", "все", "all", "вместе"):
        return "+".join(map(str, range(k))) if k else None
    if t in ("—", "-", "–", "0", "нет", "убрать", "mute"):
        return "mute"
    try:
        vs = sorted({int(float(p)) - 1 for p in t.replace(" ", "").split("+") if p})
    except ValueError:
        return None
    if not vs or min(vs) < 0:
        return None
    return "+".join(map(str, vs))


# ---------- fingerprints ----------
def model_path():
    d = os.path.join(ROOT, "models", "wespeaker")
    p = os.path.join(d, ONNX)
    if not os.path.exists(p):
        from huggingface_hub import hf_hub_download
        log("Voiceprint model WeSpeaker (CC-BY-4.0): one-time download, ~27 MB…")
        hf_hub_download(REPO, ONNX, local_dir=d)
    return p


class Embedder:
    def __init__(self):
        import onnxruntime as ort
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(model_path(), providers=prov)

    def __call__(self, x16):
        """x16: mono float at 16 kHz -> unit 256-d embedding (WeSpeaker front end: 80 fbank, hamming, CMN)."""
        import torch
        import torchaudio.compliance.kaldi as kaldi
        wav = torch.from_numpy(np.asarray(x16, np.float32) * 32768.0).unsqueeze(0)
        feat = kaldi.fbank(wav, num_mel_bins=80, frame_length=25, frame_shift=10, sample_frequency=SR,
                           window_type="hamming", dither=0.0)
        feat = (feat - feat.mean(0)).numpy()[None].astype(np.float32)
        e = self.sess.run(["embs"], {"feats": feat})[0][0]
        return e / (np.linalg.norm(e) + 1e-9)


def activity(x, fps=100):
    hop = SR // fps
    n = len(x) // hop
    db = 10 * np.log10((x[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    return db > np.percentile(db, 95) - 35


# ---------- lines ----------
def _split(run, max_len):
    """Split a run of words longer than max_len at its widest pause (not in the middle of a phrase)."""
    if len(run) < 2 or run[-1]["end"] - run[0]["start"] <= max_len:
        return [run]
    gaps = [run[i + 1]["start"] - run[i]["end"] for i in range(len(run) - 1)]
    mid = (run[0]["start"] + run[-1]["end"]) / 2
    # widest pause, ties -> closer to the middle
    i = max(range(len(gaps)), key=lambda i: (round(gaps[i], 2), -abs(run[i]["end"] - mid)))
    return _split(run[:i + 1], max_len) + _split(run[i + 1:], max_len)


def sung_lines(words, x, gap=0.35, max_len=5.0, tags=None):
    """Lines (start, end, text) from Whisper's words, plus sung stretches without words (vocalizations).
    tags: singers of each word (from a tagged song text) — a line never spans two different singers.
    Returns (lines, line_tags)."""
    order = sorted(range(len(words)), key=lambda i: words[i]["start"])
    runs = []
    for i in order:
        w = dict(words[i], tag=tags[i] if tags else None)
        if runs:
            last = runs[-1][-1]
            same = not tags or w["tag"] is None or last["tag"] is None or w["tag"] == last["tag"]
            if w["start"] - last["end"] <= gap and same:
                runs[-1].append(w)
                continue
        runs.append([w])
    lines = []
    for run in runs:
        for part in _split(run, max_len):
            known = [w["tag"] for w in part if w["tag"] is not None]
            tag = max(set(known), key=known.count) if known else None
            lines.append([part[0]["start"], part[-1]["end"], " ".join(w["word"].strip() for w in part), tag])
    act = activity(x)
    covered = np.zeros(len(act), bool)
    for s, e, _, _ in lines:
        covered[max(0, int((s - 0.1) * 100)):int((e + 0.1) * 100)] = True
    free = act & ~covered
    edges = np.flatnonzero(np.diff(np.r_[0, free.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= 60:
            for k in range(a, b, int(max_len * 100)):
                lines.append([k / 100, min(b, k + int(max_len * 100)) / 100, "", None])
    lines = sorted((r for r in lines if r[1] - r[0] >= 0.3), key=lambda r: r[0])
    return [(float(s), float(e), t) for s, e, t, _ in lines], [g for _, _, _, g in lines]


def embed_lines(emb, x, lines):
    out = []
    for s, e, _ in lines:
        seg = x[int(s * SR):int(e * SR)]
        if len(seg) < int(MIN_EMB * SR):
            out.append(None)
            continue
        win, hop = 3 * SR, int(1.5 * SR)
        parts = [seg[i:i + win] for i in range(0, max(1, len(seg) - win + 1), hop)] if len(seg) > win else [seg]
        v = np.mean([emb(p) for p in parts], 0)
        out.append(v / (np.linalg.norm(v) + 1e-9))
    return out


def cached_embeddings(vdir, x, lines):
    """Fingerprints of the lines; lines seen before (same start/end) are not recomputed."""
    path = os.path.join(vdir, "embeddings.npz")
    known = {}
    if os.path.exists(path):
        z = np.load(path)
        if "starts" in z:
            for s, e, v, o in zip(z["starts"], z["ends"], z["E"], z["ok"]):
                known[(round(float(s), 2), round(float(e), 2))] = (v, bool(o))
    todo = [i for i, (s, e, _) in enumerate(lines) if (round(s, 2), round(e, 2)) not in known]
    if todo:
        log(f"Voices: voiceprints of {len(todo)} lines…")
        new = embed_lines(Embedder(), x, [lines[i] for i in todo])
        for i, v in zip(todo, new):
            s, e, _ = lines[i]
            known[(round(s, 2), round(e, 2))] = (v if v is not None else np.zeros(256, np.float32), v is not None)
    E = np.array([known[(round(s, 2), round(e, 2))][0] for s, e, _ in lines], np.float32).reshape(len(lines), -1)
    ok = np.array([known[(round(s, 2), round(e, 2))][1] for s, e, _ in lines], bool)
    np.savez(path, E=E, ok=ok, starts=[l[0] for l in lines], ends=[l[1] for l in lines])
    return E, ok


def cluster(E, k=None):
    """Average-linkage on cosine distance; k by silhouette when not given (1 if nothing separates well)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    n = len(E)
    if n < 3:
        return np.zeros(n, int), 1, None
    Z = linkage(pdist(E, "cosine"), "average")
    if k:
        return fcluster(Z, min(k, n), "maxclust") - 1, min(k, n), None
    from sklearn.metrics import silhouette_score
    best = (1, -1.0)
    for kk in range(2, min(AUTO_MAX, n - 1) + 1):
        lab = fcluster(Z, kk, "maxclust")
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(E, lab, metric="cosine")
        if s > best[1]:
            best = (kk, s)
    if best[1] < 0.12:
        return np.zeros(n, int), 1, best[1]
    return fcluster(Z, best[0], "maxclust") - 1, best[0], best[1]


def centroids(E, ok, labels, k):
    C = np.zeros((k, E.shape[1]), np.float32)
    has = np.zeros(k, bool)
    for c in range(k):
        m = ok & (np.array(labels, dtype=object) == str(c))
        if m.any():
            v = E[m].mean(0)
            C[c], has[c] = v / (np.linalg.norm(v) + 1e-9), True
    return C, has


def _gap(a, b):
    return max(0.0, a[0] - b[1], b[0] - a[1])


def fill_short(lines, labels, todo):
    """Lines without a label: the label of the closest labelled line (by the pause between them)."""
    done = [j for j in range(len(lines)) if labels[j] is not None and j not in todo]
    for i in todo:
        near = [j for j in done if _gap(lines[i], lines[j]) < 8]
        labels[i] = labels[min(near, key=lambda j: _gap(lines[i], lines[j]))] if near else "0"
    return labels


def by_fingerprints(lines, E, ok, k):
    """Labels from timbre alone. Returns (labels, k, silhouette)."""
    idx = np.flatnonzero(ok)
    lab_ok, kk, sil = cluster(E[idx], k)
    labels = [None] * len(lines)
    for i, l in zip(idx, lab_ok):
        labels[i] = str(int(l))
    if k is None and kk > 1:
        # tiny "voices" are ad-libs/effects, not separate singers: give their lines to the closest real one
        secs = {c: sum(lines[i][1] - lines[i][0] for i in idx if labels[i] == str(c)) for c in range(kk)}
        small = [c for c in range(kk) if secs[c] < MIN_VOICE_S]
        if small and len(small) < kk:
            keep = [c for c in range(kk) if c not in small]
            C, _ = centroids(E, ok, labels, kk)
            for i in idx:
                if int(labels[i]) in small:
                    labels[i] = str(keep[int(np.argmax(E[i] @ C[keep].T))])
            ren = {str(c): str(n) for n, c in enumerate(keep)}
            labels = [ren.get(l, l) for l in labels]
            kk = len(keep)
    if kk > 1:
        # lines that sound like two voices at once, or like nobody — usually several people singing together
        S = np.zeros((kk, E.shape[1]), np.float32)
        for i in idx:
            S[int(labels[i])] += E[i]
        for i in idx:
            own = S.copy()
            own[int(labels[i])] -= E[i]  # the line itself does not count towards its own voice
            C = own / (np.linalg.norm(own, axis=1, keepdims=True) + 1e-9)
            sim = E[i] @ C.T
            o = np.argsort(-sim)
            if lines[i][1] - lines[i][0] < 1.5:
                continue
            if sim[o[0]] - sim[o[1]] < TIE or (sim[o[0]] < FIT_SIM and sim[o[0]] - sim[o[1]] < FIT_MARGIN):
                # sounds like two of them at once, or like nobody (several people) -> the two closest sing it
                labels[i] = "+".join(map(str, sorted(o[:2].tolist())))
    todo = [i for i in range(len(lines)) if labels[i] is None]
    return fill_short(lines, labels, todo), kk, sil


def by_tags(lines, line_tags, E, ok, artists):
    """Labels from a tagged song text; fingerprints only settle lines tagged with several singers."""
    keys = list(artists)
    k = len(keys)
    labels = [None] * len(lines)
    for i, t in enumerate(line_tags):
        if t and len(t) == 1 and t[0] in keys:
            labels[i] = str(keys.index(t[0]))
    C, has = centroids(E, ok, labels, k)
    for i, t in enumerate(line_tags):
        if labels[i] is not None:
            continue
        if t == ("*all*",):
            labels[i] = "+".join(map(str, range(k))) if k > 1 else "0"
        elif t:
            vs = sorted(keys.index(n) for n in t if n in keys)
            if len(vs) == 1:
                labels[i] = str(vs[0])
            elif vs:
                # "[Verse: A, B]" usually means they take turns: give the line to the one it sounds like
                if ok[i] and all(has[v] for v in vs):
                    s = sorted(((float(E[i] @ C[v]), v) for v in vs), reverse=True)
                    if s[0][0] >= FIT_SIM and s[0][0] - s[1][0] >= 0.08:
                        labels[i] = str(s[0][1])
                        continue
                labels[i] = "+".join(map(str, vs))
        elif ok[i] and has.any():
            s = E[i] @ C.T
            s[~has] = -1
            if s.max() >= FIT_SIM:
                labels[i] = str(int(np.argmax(s)))
    todo = [i for i in range(len(lines)) if labels[i] is None]
    return fill_short(lines, labels, todo), k


def analyze(cfg, song_dir, lead_dry, k=None, lyrics=None, song_artist=None):
    """Find the voices of the lead vocal; reuses the expensive parts (words, fingerprints) between calls."""
    import soundfile as sf
    from . import intelligibility as it
    from . import lyrics_tags
    from .dsp import to_sr
    vdir = os.path.join(song_dir, "voices")
    os.makedirs(vdir, exist_ok=True)
    y, sr = sf.read(lead_dry, dtype="float32", always_2d=True)
    x = to_sr(y.mean(1), sr, SR)
    wpath = os.path.join(vdir, "words_lead.json")
    if os.path.exists(wpath):
        with open(wpath, encoding="utf-8") as f:
            words = json.load(f)
    else:
        log("Voices: recognising the song's lines (Whisper)…")
        words = it.asr(cfg, [(lead_dry, wpath, None)])[0]
    tags, artists, names = None, None, None
    if lyrics and lyrics.strip():
        lw, artists = lyrics_tags.parse(lyrics, song_artist)
        if len(artists) > MAX_VOICES:
            raise ValueError(f"The lyrics name {len(artists)} singers ({', '.join(artists.values())}) — "
                             f"the tab handles up to {MAX_VOICES}. Join some of them in the lyrics (e.g. \"[Verse: A, B]\").")
        if artists:
            tags, share = lyrics_tags.align(lw, words["words"])
            names = list(artists.values())
            log(f"Voices from the lyrics: {', '.join(names)}; the lyrics matched what was heard {share:.0%}")
            if share < 0.25:
                log("Voices: the lyrics hardly match what is heard — is it the same song? "
                    "Where the lyrics do not help, voiceprints decide")
        else:
            log("Voices: the lyrics have no singer tags like [Verse 1: Name] — telling them apart by voice")
    lines, line_tags = sung_lines(words["words"], x, tags=tags)
    E, ok = cached_embeddings(vdir, x, lines)
    if tags:
        labels, kk = by_tags(lines, line_tags, E, ok, artists)
        sil, auto = None, False
    else:
        labels, kk, sil = by_fingerprints(lines, E, ok, k)
        auto = k is None
        # voice 1 = the one that sings the most
        dur = [sum(e - s for (s, e, _), l in zip(lines, labels) if l == str(c)) for c in range(kk)]
        order = list(np.argsort(-np.array(dur)))
        labels = [renumber(l, {int(c): i for i, c in enumerate(order)}) for l in labels]
    result = save(cfg, song_dir, lead_dry, x, lines, labels, kk, names=names, auto=auto, silhouette=sil)
    n_ch = sum(1 for l in labels if "+" in str(l))
    log(f"Voices: {result['k']} found" + (" (automatically)" if auto else "") +
        (f", confidence {sil:.2f}" if sil is not None else "") + (f"; together: {n_ch} lines" if n_ch else ""))
    return result


def renumber(lab, remap):
    vs = members(lab)
    if not vs:
        return lab
    return "+".join(map(str, sorted(remap[v] for v in vs)))


def _sample(x, spans, path):
    import soundfile as sf
    parts, got = [], 0.0
    for s, e in sorted(sorted(spans, key=lambda r: r[0] - r[1])[:6]):
        parts += [x[int(s * SR):int(e * SR)], np.zeros(int(0.35 * SR), np.float32)]
        got += e - s
        if got > 12:
            break
    sf.write(path, np.concatenate(parts) if parts else np.zeros(SR, np.float32), SR)
    return path


def save(cfg, song_dir, lead_dry, x, lines, labels, k, names=None, auto=False, silhouette=None):
    """voices.json + samples; also used after manual corrections of the line table."""
    vdir = os.path.join(song_dir, "voices")
    midi = line_pitch(cfg, lead_dry, vdir, lines)
    total = sum(e - s for s, e, _ in lines) or 1.0
    voices = []
    for c in range(k):
        mine = [(s, e) for (s, e, _), l in zip(lines, labels) if l == str(c)]
        withothers = [(s, e) for (s, e, _), l in zip(lines, labels) if c in members(l) and l != str(c)]
        secs = sum(e - s for s, e in mine)
        notes = [m for (_, _, _), l, m in zip(lines, labels, midi) if c in members(l) and m is not None]
        voices.append({"id": c, "name": (names[c] if names and c < len(names) else None),
                       "seconds": round(secs, 1), "share": round(secs / total, 3), "lines": len(mine),
                       "together": len(withothers),
                       "median_midi": float(np.median(notes)) if notes else None,
                       "sample": _sample(x, mine or withothers, os.path.join(vdir, f"voice_{c + 1}.wav"))})
    out = {"k": k, "auto": auto, "silhouette": silhouette, "voices": voices, "by_text": bool(names),
           "lines": [{"start": round(s, 2), "end": round(e, 2), "text": t, "voice": l}
                     for (s, e, t), l in zip(lines, labels)]}
    with open(os.path.join(vdir, "voices.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


def line_pitch(cfg, lead_dry, vdir, lines):
    """Median note of each line (RMVPE track cached next to voices.json)."""
    p = os.path.join(vdir, "f0.npy")
    if os.path.exists(p):
        f0 = np.load(p)
    else:
        from .phonetics import f0_track
        f0 = f0_track(cfg, lead_dry)
        np.save(p, f0)
    out = []
    for s, e, _ in lines:
        seg = f0[int(s * 100):int(e * 100)]
        seg = seg[seg > 40]
        out.append(float(np.median(69 + 12 * np.log2(seg / 440))) if len(seg) > 10 else None)
    return out


def relabel(cfg, song_dir, lead_dry, table):
    """Apply the user's corrections: table rows [time, text, voice] with voice like "2", "1+3", "all", "—"."""
    import soundfile as sf
    from .dsp import to_sr
    with open(os.path.join(song_dir, "voices", "voices.json"), encoding="utf-8") as f:
        cur = json.load(f)
    lines = [(r["start"], r["end"], r["text"]) for r in cur["lines"]]
    labels = []
    for r, row in zip(cur["lines"], table):
        lab = parse_label(row[2] if len(row) > 2 else None, cur.get("k"))
        labels.append(lab if lab is not None else ("mute" if r["voice"] == "chorus" else str(r["voice"])))
    used = sorted({v for l in labels for v in members(l)})  # renumber without gaps
    remap = {v: i for i, v in enumerate(used)}
    labels = [renumber(l, remap) for l in labels]
    old = {v["id"]: v.get("name") for v in cur["voices"]}
    names = [old.get(v) for v in used]
    y, sr = sf.read(lead_dry, dtype="float32", always_2d=True)
    return save(cfg, song_dir, lead_dry, to_sr(y.mean(1), sr, SR), lines, labels, max(1, len(used)),
                names=names if any(names) else None, auto=False, silhouette=cur.get("silhouette"))


def plan(analysis, assign, backing="voices", octaves=None):
    """Voice plan for the pipeline: who sings when, which model sings for whom, octave shifts.
    assign: {voice index: model | "orig" | "mute"}; backing: "voices" (each voice sings its own backing — whoever
    sings the lead at that moment), "orig" or one model for all of it; octaves: {voice index | "backing": "auto" | int}
    ("backing" only when one model sings all of it)."""
    return {"segments": [[r["start"], r["end"], str(r["voice"])] for r in analysis["lines"]],
            "assign": {str(k): v for k, v in assign.items()}, "backing": backing,
            "octaves": {str(k): v for k, v in (octaves or {}).items()}}
