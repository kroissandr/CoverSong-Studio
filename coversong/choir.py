"""A choir of your voice where the original's backing is a choir.

RVC sings one melody line at a time. Where the backing is several voices — a chord (KATSEYE Animal 1:22,
ENHYPEN Bite Me 0:56: G#3 C#4 E4 G#4) or a group singing one line together (Animal 0:13, a hum) — one render of
your voice is either a single line or, worse, a line jumping between the chord's voices. So in those places:

- chord: the notes sounding together are heard with Basic Pitch (Spotify's note model, coversong/third_party);
  besides the normal render (the top line), your voice sings the next one or two lines of the chord, each
  as its own render of the same passage (same words and vowels, that line's notes);
- unison (a group on one line — by default the backing's wordless parts: "mmm", "ooh", "aah"): two more takes
  of the same line, a few cents sharp/flat, a little late, with slightly different settings.
Layers sit 5 dB under the main voice, 18–29 ms late (the lateness slowly moving), each with its own slow pitch
wander and spread left/right, like a recorded stack; unison takes are sung from different stereo channels of the
original. Identical copies with a fixed detune and delay beat at a steady rate and sound robotic (0.20.0).

clean.choir: auto (chords + wordless backing + manual places) | chords | manual | off;
clean.choir_ranges: places by hand, e.g. "0:10-0:16, 1:20-1:30".

Crowd (test, clean.crowd: auto | off; clean.crowd_ranges by hand): where the original's backing is a crowd — loud,
notes heard, but hardly periodic, many voices beating against each other (TWICE The Feels: 50 s in 38 places; HOT,
Animal: none) — six more takes of your voice: from both channels, each with its own pitch wander and offset
(±3…16 cents), settings, lateness (22–61 ms, moving), dullness and place left to right.
"""
import os
import re

import numpy as np

from .util import log

FPS = 100
LAYER_DB = -5.0
CHORD_DB = -3.0      # a chord's other voices: harmony singers are about as loud as the top line
CHORD_THR = 0.35     # Basic Pitch note posterior counted as heard (its own default is 0.3; 0.5 missed TWICE's
                     # chords: The Feels' backing has 2+ notes 41 % of the time, 0.5 heard 26 %)
MAX_LINES = 3
MAX_SECONDS = 120.0


# ---------------------------------------------------------------- hearing several notes
def posteriors(x, sr):
    """Basic Pitch note posteriors: (frames, 88 keys from A0) and frames per second."""
    import onnxruntime as ort
    from .dsp import to_sr
    from .third_party import basic_pitch as bp
    path = os.path.join(os.path.dirname(__file__), "third_party", "basic_pitch", "nmp.onnx")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    s = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    out_name = "StatefulPartitionedCall:1"  # note posteriors (":2" onsets, ":0" contour), as basic_pitch reads them
    x = to_sr(np.asarray(x, np.float32), sr, bp.SAMPLE_RATE).astype(np.float32)
    olap = bp.OVERLAP_FRAMES * bp.FFT_HOP
    hop = bp.WINDOW - olap
    xp = np.concatenate([np.zeros(olap // 2, np.float32), x])
    parts = []
    for i in range(0, len(xp), hop):
        w = xp[i:i + bp.WINDOW]
        w = np.pad(w, (0, bp.WINDOW - len(w)))
        note = s.run([out_name], {s.get_inputs()[0].name: w[None, :, None]})[0][0]
        parts.append(note[bp.OVERLAP_FRAMES // 2:note.shape[0] - bp.OVERLAP_FRAMES // 2])
    note = np.concatenate(parts)[:len(x) // bp.FFT_HOP + 1]
    return note, bp.SAMPLE_RATE / bp.FFT_HOP


def chord_notes(note, fps, n, thr=CHORD_THR):
    """Per 10 ms: the notes sounding (MIDI, high to low; octave doubles of a higher note left out) and how many
    different pitch classes there are."""
    from .third_party.basic_pitch import BASE_MIDI
    idx = np.clip((np.arange(n) / FPS * fps).round().astype(int), 0, len(note) - 1)
    nt = note[idx]
    lines = np.zeros((n, 4))
    pcs = np.zeros(n, int)
    for i in range(n):
        p = nt[i]
        keys = [k for k in np.flatnonzero(p >= thr)
                if (k == 0 or p[k] >= p[k - 1]) and (k == len(p) - 1 or p[k] >= p[k + 1])]
        ms = sorted((BASE_MIDI + k for k in keys), reverse=True)
        kept = []
        for m in ms:
            if not any((h - m) % 12 == 0 for h in kept):
                kept.append(m)
        pcs[i] = len(kept)
        for j, m in enumerate(kept[:4]):
            lines[i, j] = m
    return lines, pcs


def tidy(line, hold=6, min_len=15):
    """A chord line as a singer would sing it: short gaps held, fragments under 150 ms dropped."""
    out = line.copy()
    last, gap = 0.0, 0
    for i, v in enumerate(out):
        if v > 0:
            if 0 < gap <= hold and last:
                out[i - gap:i] = last
            last, gap = v, 0
        else:
            gap += 1
    d = np.diff(np.r_[0, (out > 0).astype(int), 0])
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if b - a < min_len:
            out[a:b] = 0
    return out


# ---------------------------------------------------------------- where
def parse_ranges(text):
    """"0:10-0:16, 1:20–1:30" -> [(10, 16), (80, 90)] seconds."""
    def sec(t):
        t = t.strip()
        if ":" in t:
            m, s = t.split(":", 1)
            return int(m) * 60 + float(s)
        return float(t)
    out = []
    for a, b in re.findall(r"(\d+(?::\d+(?:\.\d+)?)?)\s*[-–—]\s*(\d+(?::\d+(?:\.\d+)?)?)", str(text or "")):
        try:
            s0, s1 = sec(a), sec(b)
        except ValueError:
            continue
        if s1 > s0:
            out.append((s0, s1))
    return out


def find(pcs, wordless, manual, mode, total_s):
    """Choir places: [(start s, end s, kind)], kind 'chord' or 'unison'. Chords found by ear (Basic Pitch);
    wordless backing (mode auto) and places given by hand are a choir whatever is in them."""
    n = len(pcs)
    chord = pcs >= 2

    def runs(m, min_len):
        d = np.diff(np.r_[0, m.astype(int), 0])
        return [(a, b) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= min_len]
    cand = np.zeros(n, bool)
    if mode in ("auto", "chords"):
        m = chord.copy()
        idx = np.flatnonzero(m)
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= 30:
                m[a:b] = True
        for a, b in runs(m, 50):
            if chord[a:b].mean() >= 0.3:            # really a chord, not a few stray frames
                cand[max(0, a - 10):min(n, b + 10)] = True
    places = list(wordless) if mode == "auto" else []
    places += list(manual)
    forced = np.zeros(n, bool)
    for a, b in places:
        forced[int(max(0, a) * FPS):int(min(total_s, b) * FPS)] = True
    cand |= forced
    out = []
    for a, b in runs(cand, 30):
        if chord[a:b].mean() >= 0.25:
            out.append((a / FPS, b / FPS, "chord"))
        elif forced[a:b].any():                      # wordless or given by hand: a group on one line
            out.append((a / FPS, b / FPS, "unison"))
    # the longest places first if there is too much
    if sum(b - a for a, b, _ in out) > MAX_SECONDS:
        keep, tot = [], 0.0
        for seg in sorted(out, key=lambda s: s[1] - s[0], reverse=True):
            if tot + seg[1] - seg[0] <= MAX_SECONDS:
                keep.append(seg)
                tot += seg[1] - seg[0]
        out = sorted(keep)
    return out


CROWD = [  # (cents offset, index rate, lateness s, place -1 left … +1 right, dullness 0..0.6)
    (-16, 0.30, 0.022, -0.9, 0.0), (9, 0.75, 0.035, 0.7, 0.4), (-3, 0.45, 0.048, -0.4, 0.2),
    (16, 0.60, 0.061, 0.95, 0.6), (-9, 0.70, 0.027, -0.7, 0.5), (3, 0.35, 0.054, 0.35, 0.1)]


def crowd_places(x, sr, note, fps, min_s=0.5):
    """Where the original backing (stereo) is a crowd: loud (within 20 dB of its loud parts), notes heard (Basic
    Pitch above 0.25) but hardly periodic (< 0.7: many voices beating). [(start s, end s)]."""
    from scipy.ndimage import binary_closing, binary_opening, uniform_filter1d
    from . import quality
    from .dsp import mono as _mono, to_sr
    m = _mono(x)
    p, db = quality.periodicity(to_sr(m, sr, 16000).astype(np.float32), 16000)
    k = len(p)
    cf = np.interp(np.arange(k) / FPS, np.arange(len(note)) / fps, note.max(1))
    sm = lambda z: uniform_filter1d(z.astype(float), 30)  # noqa: E731  300 ms
    c = (sm(db) > np.percentile(db, 95) - 20) & (sm(p) < 0.7) & (sm(cf) > 0.25)
    c = binary_opening(binary_closing(c, np.ones(30)), np.ones(30))
    d = np.diff(np.r_[0, c.astype(int), 0])
    return [(a / FPS, b / FPS) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))
            if b - a >= min_s * FPS]


def _sung_midi(y, sr, a0, b0, semis):
    """Per 10 ms over [a0, b0]: the note your converted backing sings (MIDI, in the original's key; 0 = none)."""
    from . import f0_extra
    from .dsp import to_sr
    seg = y[int(a0 * sr):int(b0 * sr)].mean(1) if y.ndim == 2 else y[int(a0 * sr):int(b0 * sr)]
    f0, _ = f0_extra.swift(to_sr(seg, sr, 16000).astype(np.float32))
    return np.where(f0 > 0, 69 + 12 * np.log2(np.maximum(f0, 1) / 440) - semis, 0.0)


def _dull(r, sr, amount):
    """A singer further away: some of the highs gone (one-pole low-pass blended in)."""
    if amount <= 0:
        return r
    import scipy.signal as ss
    lp = ss.sosfilt(ss.butter(1, 3500, "lowpass", fs=sr, output="sos"), r).astype(np.float32)
    return (r * (1 - amount) + lp * amount).astype(np.float32)


def _crowd(sess, stage, sides, msr, a, b, total, semis, p, sr, out):
    a0, b0 = max(0.0, a - 0.3), min(total, b + 0.3)
    nfr = int((b0 - a0) * FPS) + 2
    g = 10 ** ((LAYER_DB - 4.8) / 20)            # six takes together about as loud as the two of a unison
    start = int(a0 * sr)
    for k, (cents, ir, late, pos, dull) in enumerate(CROWD):
        pk = dict(p, index_rate=ir)
        r = _render(sess, stage, sides[k % 2], msr, a0, b0, semis, pk, sr,
                    scale=drift(nfr, int(a * 100) + 31 * k, cents=10.0, offset=cents))
        r = _dull(r, sr, dull) * _window(len(r), sr, a, b, a0, fade=0.08) * g
        r = vary_delay(r, sr, late, int(a * 100) + 17 * k, depth=0.008)
        end = min(len(out), start + len(r))
        if end <= start:
            continue
        th = (pos + 1) * np.pi / 4                 # constant-power place left … right
        pan = (np.cos(th), np.sin(th)) if out.shape[1] == 2 else (0.7,)
        for ch in range(out.shape[1]):
            out[start:end, ch] += r[:end - start] * pan[ch]


# ---------------------------------------------------------------- singing
def _render(sess, stage, mono, msr, a0, b0, semis, p, sr_out, override=None, scale=1.0):
    import soundfile as sf
    from . import f0fix
    from .dsp import to_sr
    src, dst = stage.path("_choir_src.wav"), stage.path("_choir_out.wav")
    sf.write(src, mono[int(a0 * msr):int(b0 * msr)], msr, subtype="FLOAT")
    f0fix.set_layer(override, scale, quiet=True)
    try:
        sess.convert(src, dst, semis, p)
    finally:
        f0fix.set_layer(None)
    y, ysr = sf.read(dst, dtype="float32", always_2d=True)
    for f in (src, dst):
        os.remove(f)
    return to_sr(y.mean(1), ysr, sr_out)


def drift(n, seed, cents=7.0, offset=0.0):
    """A singer's slow pitch wander, per 10 ms: smoothed random walk of about ±cents around offset (cents),
    as a frequency ratio. Fixed detune between near-identical copies beats at a steady rate — that is the
    "robot"; a wandering one sounds like people."""
    rng = np.random.default_rng(seed)
    w = np.convolve(rng.standard_normal(n + 100), np.ones(50) / 50, mode="same")[50:50 + n]
    w = w / (w.std() + 1e-9) * cents + offset
    return 2 ** (w / 1200)


def vary_delay(r, sr, base, seed, depth=0.004):
    """Late by base seconds, the lateness slowly moving by ±depth (a real double is never a fixed copy)."""
    rng = np.random.default_rng(seed)
    t = np.arange(len(r)) / sr
    tau = base + depth * np.sin(2 * np.pi * rng.uniform(0.15, 0.35) * t + rng.uniform(0, 2 * np.pi))
    return np.interp(t - tau, t, r, left=0.0).astype(np.float32)


def _window(n, sr, a, b, a0, fade=0.05):
    """Gain over a rendered excerpt (starting at a0 s): 1 inside [a, b], faded."""
    t = a0 + np.arange(n) / sr
    return np.clip(np.minimum((t - a) / fade + 1, (b - t) / fade + 1), 0, 1).astype(np.float32)


def add(cfg, sess, src_path, y, sr, semis, p, stage, wordless, soft=None):
    """Adds the choir layers to y (your converted backing, (n, channels) at sr). p: the backing's settings;
    soft: settings for wordless parts. Returns (y, info)."""
    import soundfile as sf
    c = cfg.get("clean", {})
    mode = str(c.get("choir", "auto")).lower()
    manual = parse_ranges(c.get("choir_ranges", ""))
    crowd_on = str(c.get("crowd", "off")).lower() == "auto" or bool(parse_ranges(c.get("crowd_ranges", "")))
    if mode == "off" and not manual and not crowd_on:
        return y, None
    if mode == "off" and manual:
        mode = "manual"
    x, msr = sf.read(src_path, dtype="float32", always_2d=True)
    mono = x.mean(1)
    sides = [x[:, 0], x[:, 1]] if x.shape[1] > 1 else [mono, mono]  # unison takes from different channels
    total = len(mono) / msr
    n100 = int(total * FPS) + 1
    note, fps = posteriors(mono, msr)
    lines, pcs = chord_notes(note, fps, n100)
    places = find(pcs, wordless or [], manual, mode, total) if mode != "off" else []
    crowd = []
    if str(c.get("crowd", "off")).lower() == "auto":
        crowd = crowd_places(x, msr, note, fps)
    crowd = sorted(crowd + parse_ranges(c.get("crowd_ranges", "")))
    budget, kept = MAX_SECONDS / 2, []
    for a, b in sorted(crowd, key=lambda s: s[0] - s[1]):          # the longest first
        if b - a <= budget:
            kept.append((a, b))
            budget -= b - a
    crowd = sorted(kept)
    places = [s for s in places if not any(ca < s[1] and s[0] < cb for ca, cb in crowd)]  # a crowd replaces a choir
    if not places and not crowd:
        return y, {"places": []}
    out = y.copy()
    for a, b in crowd:
        _crowd(sess, stage, sides, msr, a, b, total, semis, p, sr, out)
    if crowd:
        where = ", ".join(f"{int(a // 60)}:{a % 60:04.1f}–{int(b // 60)}:{b % 60:04.1f}" for a, b in crowd[:12])
        log(f"Crowd in the backing (test): {len(crowd)} places, {sum(b - a for a, b in crowd):.0f} s, 7 voices each — "
            f"{where}" + (" …" if len(crowd) > 12 else ""))
    nch = out.shape[1]
    pans = [(0.85, 0.45), (0.45, 0.85)]
    delays = [0.018, 0.029]
    g = 10 ** (LAYER_DB / 20)
    done = []
    for a, b, kind in places:
        a0, b0 = max(0.0, a - 0.3), min(total, b + 0.3)
        in_soft = any(sa < b and a < sb for sa, sb in (wordless or []))
        pl = dict(p, **(soft or {})) if in_soft and soft else p
        layers = []
        if kind == "chord":
            sung = _sung_midi(y, sr, a0, b0, semis)       # what your main render already sings here
            for j in range(4):
                ln = tidy(lines[int(a0 * FPS):int(b0 * FPS), j])
                if (ln > 0).mean() < 0.25:
                    continue
                k_ = min(len(ln), len(sung))
                both = (ln[:k_] > 0) & (sung[:k_] > 0)
                if both.sum() > 20 and np.mean(np.abs((sung[:k_][both] - ln[:k_][both] + 6) % 12 - 6) < 0.6) >= 0.5:
                    continue                               # that voice is your main line: no double of it
                layers.append(("line", 440.0 * 2 ** ((ln - 69) / 12) * (ln > 0)))
            layers = layers[:MAX_LINES]
        if not layers:
            kind = "unison"
            nfr = int((b0 - a0) * FPS) + 2
            seed = int(a * 100)
            layers = [("take", drift(nfr, seed, offset=6.0)), ("take", drift(nfr, seed + 1, offset=-6.0))]
        for k, (what, arg) in enumerate(layers[:MAX_LINES]):
            if what == "line":
                r = _render(sess, stage, mono, msr, a0, b0, semis, pl, sr, override=arg,
                            scale=drift(len(arg), int(a * 100) + 7 + k, cents=4.0))
                act = np.interp(np.arange(len(r)) / sr * FPS, np.arange(len(arg)), (arg > 0).astype(np.float32))
                env = np.convolve(act, np.ones(int(0.03 * sr)) / int(0.03 * sr), mode="same")
            else:
                ir = float(pl["index_rate"])
                pk = dict(pl, index_rate=round(ir * 0.6, 2) if k == 0 else round(min(0.8, ir * 1.3), 2))
                r = _render(sess, stage, sides[k % 2], msr, a0, b0, semis, pk, sr, scale=arg)
                env = np.ones(len(r), np.float32)
            gk = 10 ** (CHORD_DB / 20) if what == "line" else g
            r = vary_delay(r * env * _window(len(r), sr, a, b, a0) * gk, sr, delays[k % 2] + 0.006 * (k // 2),
                           int(a * 100) + 13 + k)
            start = int(a0 * sr)
            end = min(len(out), start + len(r))
            if end <= start:
                continue
            for ch in range(nch):
                pan = pans[k % 2][ch] if nch == 2 else 0.65
                out[start:end, ch] += r[:end - start] * pan
        done.append((a, b, kind, 1 + len(layers[:MAX_LINES])))
    secs = sum(b - a for a, b, _, _ in done)
    where = ", ".join(f"{int(a // 60)}:{a % 60:04.1f}–{int(b // 60)}:{b % 60:04.1f} "
                      f"({'chord' if k == 'chord' else 'unison'}, {v} voices)" for a, b, k, v in done[:10])
    if done:
        log(f"Backing choir: {len(done)} places, {secs:.0f} s — {where}" + (" …" if len(done) > 10 else ""))
    return out, {"places": [(round(a, 2), round(b, 2), k, v) for a, b, k, v in done],
                 "crowd": [(round(a, 2), round(b, 2)) for a, b in crowd]}
