"""Breaths and sighs: RVC hears a breath as a sound to be sung and draws it as a muffled, radio-like hiss
(Animal 1:16.9: the original's breath centred at 4.2 kHz came out at 2.9 kHz and 5 dB louder). So the original's
breaths are found and the converted sound there is replaced:

  - with your own breaths, recorded in the vocal studio (category "Breathing": inhales before a line, quick
    catch-breaths, sighs) — the closest in kind and length, following the original breath's rise and fall;
  - until you have recorded some, the breath stays as the model sang it (clean.breaths: original puts the
    original singer's breath there instead).

clean.breaths: auto (yours if recorded, else as the model sang it) | original | off.

A breath here: not pitched (SwiftF0 hears no note, periodicity < 0.45), quieter than singing but clearly there
(6–45 dB under the loud parts), 0.12–1.6 s long, not mostly inside a word Whisper heard, and not running straight
into a sung sound (that is a word's first consonant). Before singing (within 0.6 s) it is an inhale, otherwise a sigh/exhale.
"""
import glob
import os

import numpy as np

from .util import log

FPS = 100
MIN_S, MAX_S = 0.12, 1.6


def _runs(m, min_len=1):
    d = np.diff(np.r_[0, m.astype(int), 0])
    return [(a, b) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= min_len]


def detect(x, sr, words=None):
    """Breaths in an original vocal stem: [(start s, end s, 'in'|'out')]."""
    from scipy.ndimage import binary_closing, uniform_filter1d
    from . import f0_extra, quality
    from .dsp import mono, to_sr
    x16 = to_sr(mono(x), sr, 16000).astype(np.float32)
    p, db = quality.periodicity(x16, 16000)
    f0, loose = f0_extra.swift(x16)
    hop = 160
    nf = len(x16) // hop
    fr = x16[:nf * hop].reshape(nf, hop)
    spec = np.abs(np.fft.rfft(fr * np.hanning(hop), axis=1)) ** 2
    fq = np.fft.rfftfreq(hop, 1 / 16000)
    hi = 10 * np.log10(spec[:, fq >= 1000].sum(1) / (spec.sum(1) + 1e-12) + 1e-12)
    k = min(len(p), len(loose), nf)
    p, db, voiced, hi = p[:k], db[:k], loose[:k] > 0, hi[:k]
    pk = np.percentile(db, 95)
    sm = lambda z: uniform_filter1d(z.astype(float), 5)  # noqa: E731  50 ms
    cand = (sm(db) > pk - 45) & (sm(db) < pk - 6) & (sm(p) < 0.45) & ~voiced & (sm(hi) > -6)
    cand = binary_closing(cand, np.ones(4))
    sung = voiced & (db > pk - 30)
    ws = [(float(w["start"]), float(w["end"])) for w in (words or []) if "start" in w and "end" in w]
    out = []
    for a, b in _runs(cand, int(MIN_S * FPS)):
        if b - a > MAX_S * FPS:
            continue
        ta, tb = a / FPS, b / FPS
        if any(min(tb, e) - max(ta, s) > 0.5 * (tb - ta) for s, e in ws):  # mostly inside a word: a consonant
            continue
        if sung[b:b + 3].any():          # runs straight into a sung sound: a word's first consonant (s, h, f)
            continue
        nxt = np.flatnonzero(sung[b:b + 60])
        out.append((ta, tb, "in" if len(nxt) else "out"))
    return out


# ---------------------------------------------------------------- your breaths from the studio
def _kind_of(name):
    n = os.path.basename(name).lower()
    return "in" if "breath_in" in n else "out"


def bank_files(cfg, voice):
    """The voice's breath takes from the studio: voices/<voice>/recordings/studio_breath_*.wav."""
    if not voice:
        return []
    from .pitch import recordings_dir
    return sorted(glob.glob(os.path.join(recordings_dir(cfg, voice), "studio_breath_*.wav")))


def bank_signature(cfg, voice):
    """Changes when breaths are recorded or removed (so the clean stage is redone)."""
    return [os.path.basename(f) for f in bank_files(cfg, voice)]


def bank(cfg, sr, voice):
    """Your recorded breaths, cut into single breaths: [(kind, mono at sr)]."""
    import soundfile as sf
    from .dsp import to_sr
    out = []
    for f in bank_files(cfg, voice):
        try:
            y, fsr = sf.read(f, dtype="float32", always_2d=True)
        except Exception:
            continue
        m = to_sr(y.mean(1), fsr, sr).astype(np.float32)
        hop = sr // FPS
        n = len(m) // hop
        if n < 10:
            continue
        lv = 10 * np.log10((m[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
        act = lv > lv.max() - 30
        for a, b in _runs(act, int(MIN_S * FPS)):
            a0, b0 = max(0, (a - 4) * hop), min(len(m), (b + 4) * hop)
            out.append((_kind_of(f), m[a0:b0]))
    return out


def _stretch(x, n, sr, seed):
    """Noise-like sound to n samples: cropped around its loudest part, or lengthened by overlapping grains."""
    if len(x) >= n:
        e = np.convolve(x ** 2, np.ones(n), "valid")
        s = int(np.argmax(e))
        return x[s:s + n].copy()
    rng = np.random.default_rng(seed)
    g = int(0.05 * sr)
    w = np.hanning(g).astype(np.float32)
    out = np.zeros(n + g, np.float32)
    norm = np.zeros(n + g, np.float32)
    hop = g // 2
    for t in range(0, n, hop):
        src = int(np.clip(t / n * (len(x) - g) + rng.integers(-g // 2, g // 2 + 1), 0, max(0, len(x) - g)))
        seg = x[src:src + g]
        if len(seg) < g:
            seg = np.pad(seg, (0, g - len(seg)))
        out[t:t + g] += seg * w
        norm[t:t + g] += w
    return (out / np.maximum(norm, 1e-3))[:n]


def _env(x, sr, win=0.03):
    k = max(1, int(win * sr))
    return np.sqrt(np.convolve(x.astype(np.float64) ** 2, np.ones(k) / k, "same") + 1e-12)


def replace(y, sr, src, src_sr, events, mine, level_db=0.0, fade=0.02):
    """y: your converted vocal (n, ch) at sr; src: the original stem; events from detect(); mine: bank() or [].
    Each breath is the original's rise and fall at the original's level (+level_db, your vocal against the
    original's), crossfaded over fade s. Returns (y, used: 'yours'|'original' counts)."""
    from .dsp import stereo, to_sr
    out = y.copy()
    n, nch = out.shape
    s2 = stereo(src)
    s2 = np.stack([to_sr(s2[:, c], src_sr, sr) for c in range(2)], 1).astype(np.float32) if src_sr != sr else s2
    used = {"yours": 0, "original": 0}
    last = -1
    g_lvl = 10 ** (level_db / 20)
    for k, (a, b, kind) in enumerate(events):
        i0, i1 = int((a - fade) * sr), int((b + fade) * sr)
        if i0 < 0 or i1 > min(n, len(s2)):
            continue
        L = i1 - i0
        orig = s2[i0:i1]
        om = orig.mean(1)
        pool = [j for j, (kd, _) in enumerate(mine) if kd == kind] or list(range(len(mine)))
        if pool:
            pool.sort(key=lambda j: abs(len(mine[j][1]) - L))
            pick = [j for j in pool[:3] if j != last] or pool[:1]
            j = pick[k % len(pick)]
            last = j
            br = _stretch(mine[j][1], L, sr, seed=k)
            ge = _env(om, sr) / _env(br, sr)                              # follow the original breath's contour
            ge = np.clip(np.convolve(ge, np.ones(int(0.04 * sr)) / int(0.04 * sr), "same"), 0.0, 3.0)
            br = (br * ge).astype(np.float32)
            new = np.repeat(br[:, None], nch, 1) * g_lvl
            used["yours"] += 1
        else:
            new = (orig if nch == 2 else om[:, None]) * g_lvl
            used["original"] += 1
        t = np.arange(L) / sr
        w = np.clip(np.minimum(t / fade, (L / sr - t) / fade), 0, 1)[:, None].astype(np.float32)
        out[i0:i1] = out[i0:i1] * (1 - w) + new * w
    return out, used


def level_offset(y, sr, src, src_sr):
    """Your converted vocal against the original stem while singing (dB, median of 100 ms levels)."""
    from .dsp import env_db, mono, to_sr
    a = env_db(mono(y), 0.1, sr)
    b = env_db(to_sr(mono(src), src_sr, sr), 0.1, sr)
    k = min(len(a), len(b))
    act = (b[:k] > np.percentile(b[:k], 95) - 20) & (a[:k] > np.percentile(a[:k], 95) - 20)
    return float(np.median(a[:k][act] - b[:k][act])) if act.sum() > FPS else 0.0


def apply(cfg, y, sr, src_path, words, name, voice=None):
    """The clean-stage step: find the breaths in the original stem and put breaths there. Returns (y, info)."""
    import soundfile as sf
    mode = str((cfg.get("clean") or {}).get("breaths", "auto")).lower()
    if mode in ("off", "false", "0", "none"):
        return y, None
    src, ssr = sf.read(src_path, dtype="float32", always_2d=True)
    ev = detect(src, ssr, words)
    if not ev:
        return y, {"breaths": 0}
    mine = bank(cfg, sr, voice) if mode in ("auto", "yours") else []
    if mode in ("auto", "yours") and not mine:
        log(f"{name}: the original's {len(ev)} breaths left as the model sang them — no breaths of yours yet "
            f"(vocal studio → Breathing)")
        return y, {"breaths": len(ev), "used": {"yours": 0, "original": 0}}
    y2, used = replace(y, sr, src, ssr, ev, mine, level_offset(y, sr, src, ssr))
    where = ", ".join(f"{int(a // 60)}:{a % 60:04.1f}" for a, _, _ in ev[:12]) + (" …" if len(ev) > 12 else "")
    whose = (f"yours from the studio ({len(mine)} recorded)" if used["yours"] else
             "the original's (record yours in the studio, Breathing)")
    n_in = sum(1 for e in ev if e[2] == "in")
    log(f"{name}: breaths where the original breathes — {len(ev)} ({n_in} in, {len(ev) - n_in} out), "
        f"{whose}: {where}")
    return y2, {"breaths": len(ev), "used": used, "at": [(round(a, 2), round(b, 2), k) for a, b, k in ev]}
