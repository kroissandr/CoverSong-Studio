"""Effects that the original puts on its vocal for a while: a radio / telephone sound, a muffled intro, a thin
filtered bridge (TWICE The Feels 0:01–0:07: the vocal's highs 15–24 dB and its lows 10–24 dB under the song's
usual balance, then clear and bright). The stage-4 EQ matches the whole song's balance, so such a passage came
out as bright as the rest. Here such passages are found in the original stems and the same change of tone is
put on your vocal in the same places.

Found by comparing the original's tone every 0.5 s (third octaves, against its own body 400–1600 Hz) with its usual
tone over the song: a passage counts when for at least 2 s the highs (2.5–8 kHz) are 10 dB or more under the
usual — muffled — or highs and lows (100–300 Hz) are both 8 dB or more under — a radio. The tone change applied is
the original's own (third octaves, smoothed, only cuts and at most +3 dB), faded in and out over 0.25 s.

mix.section_fx: true | false.
"""
import numpy as np

from . import dsp

STEP = 0.5
WIN = 8192


def _profile(x, sr):
    m = dsp.mono(x).astype(np.float64)
    hop = int(STEP * sr)
    f = np.fft.rfftfreq(WIN, 1 / sr)
    sel = [(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6)) for c in dsp.THIRDS]
    w = np.hanning(WIN)
    rows, lv = [], []
    for i in range(0, max(1, len(m) - WIN), hop):
        seg = m[i:i + WIN]
        if len(seg) < WIN:
            seg = np.pad(seg, (0, WIN - len(seg)))
        P = np.abs(np.fft.rfft(seg * w)) ** 2
        rows.append([10 * np.log10(P[s].mean() + 1e-20) for s in sel])
        lv.append(10 * np.log10((seg ** 2).mean() + 1e-12))
    return np.array(rows), np.array(lv)


def find(x, sr, min_s=2.0):
    """Passages with an effect on the tone: [(start s, end s, curve dB at dsp.THIRDS, 'muffled'|'radio')]."""
    R, lv = _profile(x, sr)
    if len(R) < 8:
        return []
    C = dsp.THIRDS
    act = lv > np.percentile(lv, 95) - 20
    if act.sum() < 8:
        return []
    body = (C >= 400) & (C <= 1600)
    Rn = R - R[:, body].mean(1, keepdims=True)
    D = Rn - np.median(Rn[act], 0)
    hi = D[:, (C >= 2500) & (C <= 8000)].mean(1)
    lo = D[:, (C >= 100) & (C <= 300)].mean(1)
    sm = lambda z: np.convolve(z, np.ones(3) / 3, "same")  # noqa: E731  1.5 s
    hi_s, lo_s = sm(np.where(act, hi, 0.0)), sm(np.where(act, lo, 0.0))
    muffled = act & (hi_s <= -10)
    radio = act & (hi_s <= -8) & (lo_s <= -8)
    cand = muffled | radio
    idx = np.flatnonzero(cand)                                # gaps up to 1.5 s inside a passage (a bright "s")
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= 4:
            cand[a:b] = True
    out = []
    d = np.diff(np.r_[0, cand.astype(int), 0])
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if (b - a) * STEP < min_s:
            continue
        use = act[a:b]
        curve = np.median(D[a:b][use], 0) if use.any() else np.median(D[a:b], 0)
        curve = np.convolve(np.pad(curve, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
        curve = np.clip(curve - np.median(curve[body]), -30.0, 3.0)
        kind = "radio" if radio[a:b].mean() >= 0.5 else "muffled"
        t0 = a * STEP
        t1 = b * STEP + WIN / sr - STEP                        # the last window's span
        out.append((t0, t1, curve, kind))
    return out


def _curve(D, act, a, b, body):
    use = act[a:b]
    c = np.median(D[a:b][use], 0) if use.any() else np.median(D[a:b], 0)
    c = np.convolve(np.pad(c, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    return c - np.median(c[body])


def against_cover(places, cover, sr):
    """Keep of each passage only the change your vocal does not already have there (your voice may darken by itself
    on low notes): the original's change minus yours, cuts only (at most +3 dB); passages left with less than 4 dB
    on the highs or lows are dropped."""
    if not places:
        return []
    R, lv = _profile(cover, sr)
    C = dsp.THIRDS
    act = lv > np.percentile(lv, 95) - 20
    body = (C >= 400) & (C <= 1600)
    Rn = R - R[:, body].mean(1, keepdims=True)
    D = Rn - np.median(Rn[act], 0)
    hi, lo = (C >= 2500) & (C <= 8000), (C >= 100) & (C <= 300)
    out = []
    for a, b, curve, kind in places:
        ia, ib = int(a / STEP), max(int(a / STEP) + 1, int(round((b - WIN / sr + STEP) / STEP)))
        mine = _curve(D, act, ia, min(ib, len(D)), body) if ia < len(D) else np.zeros_like(curve)
        c = np.clip(curve - np.minimum(mine, 0.0), -30.0, 3.0)
        if c[hi].mean() <= -4 or c[lo].mean() <= -4:
            out.append((a, b, c, kind))
    return out


def apply(y, sr, places, fade=0.25):
    """y (n, ch) with each passage's tone change, crossfaded."""
    if not places:
        return y
    out = y.copy()
    n = len(y)
    t = np.arange(n) / sr
    for a, b, curve, _ in places:
        i0, i1 = max(0, int((a - 1.0) * sr)), min(n, int((b + 1.0) * sr))
        if i1 <= i0:
            continue
        seg = y[i0:i1]
        f = dsp.apply_eq(seg, curve, sr=sr, centers=dsp.THIRDS, taps=4095)
        tt = t[i0:i1]
        w = np.clip(np.minimum((tt - a) / fade + 0.5, (b - tt) / fade + 0.5), 0, 1).astype(np.float32)
        w = w[:, None] if seg.ndim == 2 else w
        out[i0:i1] = seg * (1 - w) + f * w
    return out.astype(np.float32)


def describe(places):
    names = {"radio": "radio", "muffled": "muffled"}
    return ", ".join(f"{int(a // 60)}:{a % 60:04.1f}–{int(b // 60)}:{b % 60:04.1f} ({names[k]}, highs "
                     f"{np.mean(c[(dsp.THIRDS >= 2500) & (dsp.THIRDS <= 8000)]):+.0f} dB, lows "
                     f"{np.mean(c[(dsp.THIRDS >= 100) & (dsp.THIRDS <= 300)]):+.0f} dB)" for a, b, c, k in places)
