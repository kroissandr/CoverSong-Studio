"""How clean the converted voice is, and the clean-ups that work on it.

Measured on the covers of 13.10 (HOT, Bones, RINGO, UNTOUCHABLE, Sid Bader, Stromae, Tananai, bülow):
- creak/rasp: 0.5–4.6 % of the notes the original sings cleanly come out rough, and 72–89 % of that is in the
  first/last 50 ms of a note (where the pitch tracker switches the voice on and off);
- noise between the harmonics (2–4 kHz): 7–14 dB more than in the original on the same notes ("hiss");
- air above 8 kHz: 15–20 dB less than in the original (RVC hears 16 kHz audio and draws the top itself).

periodicity() / creak() / report() measure; harmonic_clean() and restore_air() fix the last two.
"""
import numpy as np

FPS = 100


# ---------------------------------------------------------------- measuring
def periodicity(x, sr, fmin=70, fmax=1000):
    """Per 10 ms (40 ms window): how clearly pitched the sound is (0..1, normalized autocorrelation at the best
    pitch lag) and its level in dB."""
    x = np.asarray(x, np.float32)
    hop, win = sr // FPS, int(0.04 * sr)
    n = 1 + max(0, (len(x) - win) // hop)
    p, db = np.zeros(n, np.float32), np.full(n, -120.0, np.float32)
    w = np.hanning(win).astype(np.float32)
    wac = np.correlate(w, w, "full")[win - 1:]
    nfft = 1 << int(np.ceil(np.log2(2 * win)))
    lo, hi = int(sr / fmax), min(win - 1, int(sr / fmin))
    for c in range(0, n, 2000):
        idx = np.arange(c, min(n, c + 2000)) * hop
        fr = np.stack([x[i:i + win] for i in idx])
        fr = fr - fr.mean(1, keepdims=True)
        ac = np.fft.irfft(np.abs(np.fft.rfft(fr * w, nfft, axis=1)) ** 2, nfft, axis=1)[:, :win]
        r = ac / (ac[:, :1] + 1e-12) * (wac[0] / np.maximum(wac, 1e-9))[None, :]
        p[c:c + len(idx)] = np.clip(r[:, lo:hi].max(1), 0, 1)
        db[c:c + len(idx)] = 10 * np.log10((fr ** 2).mean(1) + 1e-12)
    return p, db


def creak(src, cov, sr):
    """Share (%) of the notes the original sings cleanly (loud, periodicity >= 0.8) that come out rough in the
    cover (periodicity < 0.6), and how much of that sits at note edges (first/last 50 ms). src, cov: mono, same sr,
    time-aligned."""
    ps, ds = periodicity(src, sr)
    pc, _ = periodicity(cov, sr)
    n = min(len(ps), len(pc))
    ps, ds, pc = ps[:n], ds[:n], pc[:n]
    v = (ds > np.percentile(ds, 95) - 30) & (ps >= 0.8)
    bad = v & (pc < 0.6)
    if v.sum() < 50:
        return {"creak": None, "edges": None, "rough": None}
    edge = ~v
    dist = np.full(n, 99)
    last = -99
    for i in range(n):
        if edge[i]:
            last = i
        dist[i] = i - last
    last = 10 ** 9
    for i in range(n - 1, -1, -1):
        if edge[i]:
            last = i
        dist[i] = min(dist[i], last - i)
    return {"creak": float(bad[v].mean() * 100),
            "edges": float((dist[bad] <= 5).mean() * 100) if bad.any() else None,
            "rough": float(np.mean(1 - pc[v]))}


def band_db(x, sr, lo, hi, seconds=90):
    n = min(len(x), int(seconds * sr))
    X = np.abs(np.fft.rfft(np.asarray(x[:n], np.float64) * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    return float(10 * np.log10(X[(f >= lo) & (f < hi)].mean() + 1e-30))


def air_db(x, sr):
    """8–16 kHz against 200–1000 Hz (dB): how much 'air' a voice has."""
    return band_db(x, sr, 8000, 16000) - band_db(x, sr, 200, 1000)


# ---------------------------------------------------------------- noise between the harmonics
def harmonic_clean(y, sr, f0, depth_db=6.0, lo=700.0, hi=7000.0):
    """In sung frames, turn down what lies between the voice's harmonics by up to depth_db (the hiss a model
    adds), leaving the harmonics themselves alone. f0: Hz per 10 ms of y (0 = not sung). Frequencies below lo
    (harmonics far apart, nothing to clean) and above hi (air, consonants) are not touched; the effect fades in
    and out over ~20 % of the band. Mono or stereo y; the same gains for every channel."""
    from scipy.signal import istft, stft
    y = np.asarray(y, np.float32)
    ys = y[:, None] if y.ndim == 1 else y
    hop = sr // FPS
    nper = 4 * hop
    floor = 10 ** (-depth_db / 20)
    out = np.zeros_like(ys)
    gains = None
    for ch in range(ys.shape[1]):
        f, _, Z = stft(ys[:, ch], sr, nperseg=nper, noverlap=nper - hop, boundary="even", padded=True)
        if gains is None:
            gains = _harmonic_gains(f, Z.shape[1], f0, floor, lo, hi)
        Z *= gains[:, :Z.shape[1]]
        _, z = istft(Z, sr, nperseg=nper, noverlap=nper - hop, boundary=True)
        del Z
        z = z[:len(ys)]
        out[:len(z), ch] = z
    return out[:, 0] if y.ndim == 1 else out


def _harmonic_gains(f, m, f0, floor, lo, hi, block=2000):
    """Gain per (bin, frame): 1 on the harmonics of f0, `floor` half-way between them; stft frame k is centred at
    k*hop, like f0[k]. Computed in blocks of frames (a whole song at once would need gigabytes)."""
    from scipy.ndimage import uniform_filter1d
    f0 = np.asarray(f0, np.float64)
    F = f0[np.clip(np.arange(m), 0, len(f0) - 1)] if len(f0) else np.zeros(m)
    G = np.ones((len(f), m), np.float32)
    band = (np.clip((f - lo) / (0.2 * lo), 0, 1) * np.clip((hi - f) / (0.2 * hi), 0, 1)).astype(np.float32)
    sel = band > 0
    fs = f[sel].astype(np.float32)
    for c in range(0, m, block):
        Fc = F[c:c + block]
        voiced = Fc > 50
        if not voiced.any():
            continue
        h = fs[:, None] / np.where(voiced, Fc, 1.0).astype(np.float32)[None, :]
        dist = np.abs(h - np.round(h))                       # 0 at a harmonic, 0.5 half-way between
        g = floor + (1 - floor) * np.clip((0.5 - dist) / 0.3, 0, 1)
        g = 1 - (1 - g) * band[sel][:, None]
        g[h < 0.75] = 1.0                                    # below the fundamental: leave
        g[:, ~voiced] = 1.0
        G[sel, c:c + block] = g
    return uniform_filter1d(G, 3, axis=1, mode="nearest")    # no sudden on/off between frames


# ---------------------------------------------------------------- air above 8 kHz
def restore_air(y, sr, src, cutoff=7500.0, max_db=14.0):
    """Give the voice back the air the model leaves out: the original's band above `cutoff` (breath, air — no
    pitch or timbre of the singer up there) is added under your voice wherever your voice has less of it than the
    original has relative to its own 1–4 kHz. Level follows your voice frame by frame (50 ms), never more than
    max_db of lift. y: mono/stereo cover, src: mono original, same sr, time-aligned. Returns (y, mean lift dB)."""
    from scipy.signal import butter, sosfiltfilt
    y = np.asarray(y, np.float32)
    ys = y[:, None] if y.ndim == 1 else y
    n = len(ys)
    s = np.asarray(src[:n], np.float32)
    if len(s) < n:
        s = np.pad(s, (0, n - len(s)))
    sos_hi = butter(4, cutoff, "highpass", fs=sr, output="sos")
    sos_mid = butter(4, [1000, 4000], "bandpass", fs=sr, output="sos")
    s_hi = sosfiltfilt(sos_hi, s).astype(np.float32)
    c = ys.mean(1)
    hop = sr // FPS
    m = n // hop

    def env(z):
        e = (z[:m * hop].reshape(m, hop) ** 2).mean(1)
        k = np.ones(5) / 5                                          # 50 ms
        return np.convolve(e, k, mode="same") + 1e-14
    e_s_hi, e_s_mid = env(s_hi), env(sosfiltfilt(sos_mid, s))
    e_c_hi, e_c_mid = env(sosfiltfilt(sos_hi, c)), env(sosfiltfilt(sos_mid, c))
    target = e_c_mid * (e_s_hi / e_s_mid)            # the original's air-to-presence balance, at your level
    need = np.maximum(target - e_c_hi, 0)
    g = np.sqrt(need / e_s_hi)
    active = e_c_mid > np.percentile(e_c_mid, 95) * 10 ** (-40 / 10)
    g = np.where(active, np.minimum(g, 10 ** (max_db / 20) * np.sqrt(e_c_hi / e_s_hi)), 0)
    g = np.convolve(g, np.ones(5) / 5, mode="same")
    gs = np.interp(np.arange(n) / hop, np.arange(m), g).astype(np.float32)
    out = ys + (gs * s_hi)[:, None]
    lift = 10 * np.log10((env(sosfiltfilt(sos_hi, out.mean(1)))[active].mean()) / e_c_hi[active].mean()) if active.any() else 0.0
    return (out[:, 0] if y.ndim == 1 else out), float(lift)
