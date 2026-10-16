"""Pure numpy/scipy DSP for stages 4–5 (vocal processing matched to the original, mixing, mastering)."""
import numpy as np
import scipy.signal as ss

SR = 44100
CTRL = 100  # control rate, frames per second


# ---------------------------------------------------------------- basics
def to_sr(x, sr_in, sr_out=SR):
    if sr_in == sr_out:
        return x
    from math import gcd
    g = gcd(sr_in, sr_out)
    return ss.resample_poly(x, sr_out // g, sr_in // g, axis=0).astype(np.float32)


def mono(x):
    return x.mean(1) if x.ndim == 2 else x


def stereo(x):
    return np.stack([x, x], 1) if x.ndim == 1 else (np.repeat(x, 2, 1) if x.shape[1] == 1 else x)


def fit_len(x, n):
    return x[:n] if len(x) >= n else np.pad(x, [(0, n - len(x))] + [(0, 0)] * (x.ndim - 1))


def env_db(x, win_s, sr=SR, hop_s=1 / CTRL):
    """RMS level in dB on a CTRL-rate grid."""
    m = mono(x).astype(np.float64)
    hop = int(hop_s * sr)
    win = max(hop, int(win_s * sr))
    c = np.concatenate([[0.0], np.cumsum(m ** 2)])
    centers = np.arange(0, len(m), hop)
    a = np.clip(centers - win // 2, 0, len(m))
    b = np.clip(centers + win // 2, 0, len(m))
    p = (c[b] - c[a]) / np.maximum(b - a, 1)
    return 10 * np.log10(p + 1e-12)


def active_mask(level_db, below_peak=30):
    return level_db > np.percentile(level_db, 95) - below_peak


def ctrl_to_samples(g_ctrl, n, sr=SR):
    t_ctrl = np.arange(len(g_ctrl)) / CTRL
    return np.interp(np.arange(n) / sr, t_ctrl, g_ctrl).astype(np.float32)


# ---------------------------------------------------------------- alignment
def align_lag(ref, x, sr=SR, max_s=0.3):
    """Samples by which x lags ref (positive = x late), from 1 kHz-rate envelopes."""
    r = 1000
    a = env_db(ref, 0.01, sr, 1 / r)
    b = env_db(x, 0.01, sr, 1 / r)
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    m = int(max_s * r)
    xc = []
    for k in range(-m, m + 1):
        if k >= 0:
            xc.append(np.dot(a[:n - k], b[k:]))
        else:
            xc.append(np.dot(a[-k:], b[:n + k]))
    return int((np.argmax(xc) - m) * sr / r)


def shift(x, lag):
    """Move x earlier by lag samples (lag > 0) or later (lag < 0), keeping length."""
    if lag == 0:
        return x
    if lag > 0:
        return np.concatenate([x[lag:], np.zeros_like(x[:lag])])
    return np.concatenate([np.zeros_like(x[:(-lag)]), x[:lag]])


# ---------------------------------------------------------------- EQ matching
OCT = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000], float)


def long_term_spectrum(x, sr=SR, n=4096):
    m = mono(x)
    f, t, Z = ss.stft(m, sr, nperseg=n, noverlap=n // 2)
    P = np.abs(Z) ** 2
    e = P.sum(0)
    act = e > np.percentile(e, 50)
    return f, P[:, act].mean(1) + 1e-20


def octave_db(f, P):
    out = []
    for c in OCT:
        sel = (f >= c / np.sqrt(2)) & (f < c * np.sqrt(2))
        out.append(10 * np.log10(P[sel].mean()) if sel.any() else np.nan)
    return np.array(out)


def own_low_band(info):
    """(lo, hi) Hz: where your voice's own fundamentals lie when you sing lower than the original (an octave down:
    there the original has almost nothing, and matching it would cut the body out of your voice). None if not
    lower. info: the lead's pitch summary from stage 3 (clean_info["lead_f0"])."""
    if not info or not info.get("src_med") or not info.get("cov_med"):
        return None
    if info["cov_med"] > info["src_med"] * 2 ** (-5 / 12):
        return None
    return float(info["cov_p10"]) / np.sqrt(2), float(info["src_p10"])


def eq_match_curve(ref, x, sr=SR, limit_db=4.0, hf_boost_cap=2.0, cut_db=None, keep=None, low_limit_db=None,
                   smooth=(0.25, 0.5, 0.25)):
    """Octave-band correction that moves x's tonal balance toward ref, limited so the voice keeps its own timbre.
    Cuts may go deeper than boosts (cut_db): cutting boom/boxiness never brings out model artefacts.
    keep: (lo, hi) Hz band of your own fundamentals (own_low_band): cut there by 2 dB at most.
    low_limit_db: boost limit for the low bands (63/125 Hz) — body is safe to add, unlike highs."""
    f, Pr = long_term_spectrum(ref, sr)
    _, Px = long_term_spectrum(x, sr)
    d = octave_db(f, Pr) - octave_db(f, Px)
    d = np.nan_to_num(d - np.nanmedian(d[2:7]))  # balance only, level is handled later
    d = np.convolve(np.pad(d, 1, mode="edge"), list(smooth), mode="valid")  # smooth across octaves
    lim = np.full(len(d), float(limit_db))
    if low_limit_db is not None:
        lim[OCT <= 125] = low_limit_db
    d = np.clip(d, -(cut_db or limit_db), lim)
    d[OCT >= 4000] = np.minimum(d[OCT >= 4000], hf_boost_cap)  # do not amplify a band-limited model's artefacts
    if keep:
        sel = (OCT >= keep[0]) & (OCT < keep[1])
        d[sel] = np.maximum(d[sel], -2.0)
    return d


def band_balance_db(x, sr=SR, band=(2000, 8000), body=(200, 1000), mask=None, n=2048):
    """Energy of `band` against the voice's body (dB) over loud frames — only the frames of `mask` (bool per
    10 ms, e.g. vowels: not consonants) when given."""
    m = mono(x).astype(np.float64)
    f, t, Z = ss.stft(m, sr, nperseg=n, noverlap=n // 2)
    P = np.abs(Z) ** 2
    e = P.sum(0)
    act = e > np.percentile(e, 50)
    if mask is not None:
        idx = np.clip((t * 100).astype(int), 0, len(mask) - 1)
        act &= np.asarray(mask, bool)[idx]
    if act.sum() < 10:
        act = e > np.percentile(e, 50)
    hi = P[(f >= band[0]) & (f < band[1])][:, act].sum()
    lo = P[(f >= body[0]) & (f < body[1])][:, act].sum()
    return float(10 * np.log10((hi + 1e-20) / (lo + 1e-20)))


def brightness_db(x, sr=SR, mask=None):
    """Presence and air (2–8 kHz) against the body of the voice (200–1000 Hz), dB, over the louder half."""
    return band_balance_db(x, sr, (2000, 8000), (200, 1000), mask)


def exciter_harmonics(x, sr=SR, f_lo=1200.0):
    """The exciter's side chain: the band above f_lo, softly clipped (it gains overtones), kept between 2 and
    14 kHz; mixed in parallel it adds sheen. Mono in, mono out."""
    m = mono(x).astype(np.float64)
    h = ss.sosfiltfilt(ss.butter(2, f_lo, "highpass", fs=sr, output="sos"), m)  # zero phase: adds, never cancels
    a = np.abs(h)
    ref = np.percentile(a[a > a.max() * 1e-3], 95) if (a > 0).any() else 1.0
    g = 1.5 / (ref + 1e-12)
    sat = np.tanh(g * h)
    dist = (sat + 0.3 * (np.abs(sat) - np.abs(sat).mean())) / g  # the band softly clipped: odd + some even overtones
    dist = ss.sosfiltfilt(ss.butter(2, [1500, min(7000, sr / 2 - 500)], "bandpass", fs=sr, output="sos"), dist)
    return dist.astype(np.float32)


AIR_MAX_DB = 8.0
PRESENCE = (4000.0, 7000.0)
PRESENCE_MAX_DB = 6.0


def _band_boost(x, lo, hi, gain_db, sr=SR):
    """Zero-phase boost of the band lo..hi (Hz) by gain_db (hi=None: a high shelf from lo)."""
    if hi is None:
        sos = ss.butter(2, lo, "highpass", fs=sr, output="sos")
    else:
        sos = ss.butter(2, [lo, hi], "bandpass", fs=sr, output="sos")
    b = ss.sosfiltfilt(sos, x, axis=0)
    return (x + (10 ** (gain_db / 20) - 1) * b).astype(np.float32)


def match_brightness(ref, x, sr=SR, max_lift_db=6.0, min_deficit_db=1.5, vowels=None, exciter=True, presence=True,
                     scale=1.0):
    """Where the original's sung vowels are brighter than yours (2–7 kHz against the body), saturation-style
    overtones are added in parallel until the balance meets the original (at most max_lift_db); then the air
    above 7 kHz of the vowels is matched with a high shelf (at most AIR_MAX_DB). A bright, dense produced vocal
    (Industry Baby: vowels 6–7 dB brighter above 4 kHz than yours) gets its sheen from overtones, and boosting
    an EQ alone would only lift the model's noise. Measured on vowels only (vowels: bool per 10 ms) — the
    consonants are matched separately afterwards.
    presence=True: between the two, the vowels' 4–7 kHz is matched on its own (at most PRESENCE_MAX_DB). Off by
    default since 0.23.1: your 4–7 kHz is mostly noise, not overtones (HOT: harmonic-ness 1.03 against the
    original's 1.52), so lifting it made the voice hissy and raspy (The Feels: rough frames 6.8 -> 8.2 %) and it
    spilled into 2–4 kHz. exciter=False: shelves only.
    scale < 1: everything found is applied only partly (overtones' amount and shelves' dB times scale) — the
    roughness guard in stage 4 backs off this way. Returns (x, info)."""
    pres = lambda z: band_balance_db(z, sr, (2000, 7000), (200, 1000), vowels)  # noqa: E731
    t_ref, t_x = pres(ref), pres(x)
    deficit = t_ref - t_x
    info = {"ref_db": t_ref, "before_db": t_x, "after_db": t_x, "amount": 0.0, "air_db": 0.0, "presence_db": 0.0}
    if exciter and deficit >= min_deficit_db:
        target = t_x + min(deficit, max_lift_db)
        h = exciter_harmonics(x, sr)
        hs = h[:, None] if x.ndim == 2 else h
        best = (0.0, x, t_x)
        for amt in (0.15, 0.3, 0.5, 0.75, 1.0, 1.25):  # 1.25: the side chain ~9 dB under the voice, no more
            y = x + amt * hs
            t = pres(y)
            best = (amt, y, t)
            if t >= target:
                break
        if scale < 1:
            amt = best[0] * scale
            y = x + amt * hs
            best = (amt, y, pres(y))
        x = best[1].astype(np.float32)
        info.update(amount=best[0], after_db=best[2])
    prs = lambda z: band_balance_db(z, sr, PRESENCE, (200, 1000), vowels)  # noqa: E731
    p_ref = prs(ref)
    for _ in range(2 if presence else 0):  # the band filter's skirts: a second, smaller step lands it
        d_p = p_ref - prs(x)
        room = PRESENCE_MAX_DB - info["presence_db"]
        if d_p < (min_deficit_db if info["presence_db"] == 0 else 0.5) or room <= 0.1:
            break
        g = min(d_p, room) * scale
        if g <= 0.1:
            break
        x = _band_boost(x, PRESENCE[0], PRESENCE[1], g, sr)
        info["presence_db"] += g
    air = lambda z: band_balance_db(z, sr, (7000, 16000), (200, 1000), vowels)  # noqa: E731
    d_air = air(ref) - air(x)
    if d_air >= min_deficit_db:
        g = min(d_air, AIR_MAX_DB) * scale
        if g > 0.1:
            x = _band_boost(x, 7000.0, None, g, sr)   # a zero-phase high shelf
            info["air_db"] = g
    return x, info


def add_breath(x, ref, sr=SR, max_ratio=1.0, seed=11):
    """Air in the backing: the original's backing is often breathy group singing (Bite Me: periodicity 0.68,
    only 12 % of it a clean tone), while RVC sings every line as a clean tone (0.89, 48 %) — a choir of clean
    tones sounds synthetic. Breath noise shaped like your own voice's spectrum (not the original voice) is added
    frame by frame until the noise-to-tone balance meets the original's, mostly above ~1 kHz, independently in
    each channel. x: stereo/mono cover, ref: the original (time-aligned). Returns (x, info)."""
    from scipy.ndimage import uniform_filter1d
    from .quality import periodicity
    xs = x[:, None] if x.ndim == 1 else x
    n = len(xs)
    r = fit_len(mono(ref), n)

    def per100(z):
        z16 = to_sr(z, sr, 16000).astype(np.float32)
        p, db = periodicity(z16, 16000)
        return uniform_filter1d(p, 5), db
    po, dbo = per100(r)
    pc, dbc = per100(mono(xs))
    m = min(len(po), len(pc))
    po, pc, dbc = po[:m], pc[:m], dbc[:m]
    loud = dbc > np.percentile(dbc, 95) - 30
    ratio = lambda p: (1 - p) / np.maximum(p, 0.3)  # noise / tone power  # noqa: E731
    add = np.where(loud, np.clip(ratio(po) - ratio(pc), 0, max_ratio), 0.0)
    add = uniform_filter1d(add, 10)
    info = {"orig_tonal": float(np.median(po[loud])) if loud.any() else 0.0,
            "cover_tonal": float(np.median(pc[loud])) if loud.any() else 0.0, "added_db": -99.0}
    if not (add > 0.02).any():
        return x, info
    nper, hop = 1024, 256
    rng = np.random.default_rng(seed)
    out = xs.copy()
    tot_n, tot_x = 0.0, 0.0
    for ch in range(xs.shape[1]):
        f, t, Z = ss.stft(xs[:, ch], sr, nperseg=nper, noverlap=nper - hop)
        A = np.abs(Z)
        env = uniform_filter1d(A, 15, axis=0)                             # the voice's spectral envelope
        w = np.clip((f - 600.0) / 600.0, 0, 1)[:, None]                   # breath lives above ~1 kHz
        a_t = np.interp(t * 100, np.arange(m), add)                       # per STFT frame
        e_frame = (A ** 2).sum(0) + 1e-20
        N = env * w * np.exp(2j * np.pi * rng.random(A.shape))
        e_noise = (np.abs(N) ** 2).sum(0) + 1e-20
        N *= np.sqrt(a_t * e_frame / e_noise)[None, :]
        _, y = ss.istft(N, sr, nperseg=nper, noverlap=nper - hop)
        y = fit_len(y.astype(np.float32), n)
        out[:, ch] += y
        tot_n += float((y ** 2).sum())
        tot_x += float((xs[:, ch] ** 2).sum())
    info["added_db"] = float(10 * np.log10(tot_n / (tot_x + 1e-20) + 1e-20))
    pa, _ = per100(mono(out))
    info["after_tonal"] = float(np.median(pa[:m][loud])) if loud.any() else 0.0
    return (out if x.ndim == 2 else out[:, 0]).astype(np.float32), info


def low_shelf(x, fc, gain_db, sr=SR):
    """Zero-phase shelf: everything below fc changed by gain_db."""
    lp = ss.sosfiltfilt(ss.butter(2, fc, "lowpass", fs=sr, output="sos"), x, axis=0)
    return (x + (10 ** (gain_db / 20) - 1) * lp).astype(np.float32)


def match_low(ref, x, max_db, fc=200.0, min_db=1.5, sr=SR):
    """When you sing an octave lower, your fundamentals land where the original has almost nothing (HOT: the
    lead's 120–250 Hz +11 dB over the original's balance, the backing's +13.5 dB) and the voice gets thick and
    muddy. The low end (60–250 Hz against the body, 250–1000 Hz) is brought toward the original's with a gentle
    zero-phase shelf below fc, at most max_db — your body above 250 Hz stays. Returns (x, cut_db)."""
    low = lambda z: band_balance_db(z, sr, (60, 250), (250, 1000))  # noqa: E731
    target = low(ref)
    ex = low(x) - target
    if ex < min_db:
        return x, 0.0
    g = min(max_db, ex)
    for _ in range(3):  # the shelf's slope: correct by what is left
        y = low_shelf(x, fc, -g, sr)
        left = low(y) - target
        if left <= 0.5 or g >= max_db:
            break
        g = min(max_db, g + left)
    return low_shelf(x, fc, -g, sr), float(g)


THIRDS = 1000.0 * 2 ** (np.arange(-10, 12) / 3)   # 1/3-octave centers, 100 Hz … 12.7 kHz


def eq_match_fine(ref, x, sr=SR, boost_db=4.0, cut_db=6.0, hf_boost_cap=2.0):
    """Third-octave correction (finer than eq_match_curve): for sounds whose balance turns within an octave — a
    hum's nasal resonance sits at 2–4 kHz while 1–2 kHz is missing (Animal's "mmm": +5 / -4 dB against the
    original), which an octave EQ averages away. Returns gains (dB) at THIRDS."""
    f, Pr = long_term_spectrum(ref, sr)
    _, Px = long_term_spectrum(x, sr)

    def bands(P):
        return np.array([10 * np.log10(P[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))].mean() + 1e-20)
                         for c in THIRDS])
    d = bands(Pr) - bands(Px)
    mid = (THIRDS >= 250) & (THIRDS <= 4000)
    d = d - np.median(d[mid])
    d = np.convolve(np.pad(d, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    d = np.clip(d, -cut_db, boost_db)
    d[THIRDS >= 4000] = np.minimum(d[THIRDS >= 4000], hf_boost_cap)
    return d


def apply_eq(x, curve_db, sr=SR, taps=2047, centers=None):
    centers = OCT if centers is None else centers
    freqs = np.concatenate([[0], centers, [sr / 2]])
    gains = np.concatenate([[curve_db[0]], curve_db, [curve_db[-1]]])
    lf = np.log10(np.maximum(freqs, 20))
    grid = np.linspace(0, sr / 2, 512 if centers is OCT else 2048)
    g = 10 ** (np.interp(np.log10(np.maximum(grid, 20)), lf, gains) / 20)
    h = ss.firwin2(taps, grid, g, fs=sr)
    if x.ndim == 1:
        return ss.fftconvolve(x, h, mode="same").astype(np.float32)
    return np.stack([ss.fftconvolve(x[:, c], h, mode="same") for c in range(x.shape[1])], 1).astype(np.float32)


# ---------------------------------------------------------------- dynamics
def dyn_range(x, sr=SR):
    lv = env_db(x, 0.1, sr)
    act = active_mask(lv, 30)
    return np.percentile(lv[act], 90) - np.percentile(lv[act], 20), np.percentile(lv[act], 50)


def compressor(x, thr_db, ratio, attack=0.01, release=0.12, knee=6.0, sr=SR):
    lv = env_db(x, 0.01, sr)  # 10 ms RMS on the control grid
    over = lv - thr_db
    gr = np.where(over <= -knee / 2, 0.0,
                  np.where(over >= knee / 2, over * (1 - 1 / ratio),
                           (1 - 1 / ratio) * (over + knee / 2) ** 2 / (2 * knee)))
    a_att, a_rel = np.exp(-1 / (attack * CTRL)), np.exp(-1 / (release * CTRL))
    sm = np.zeros_like(gr)
    prev = 0.0
    for i, v in enumerate(gr):
        a = a_att if v > prev else a_rel
        prev = a * prev + (1 - a) * v
        sm[i] = prev
    g = ctrl_to_samples(10 ** (-sm / 20), len(x), sr)
    return (x * (g[:, None] if x.ndim == 2 else g)).astype(np.float32), float(sm.max())


def match_dynamics(ref, x, sr=SR):
    r_ref, _ = dyn_range(ref, sr)
    r_x, med_x = dyn_range(x, sr)
    if r_x <= r_ref + 1.0:
        return x, {"ratio": 1.0, "range_ref": r_ref, "range_in": r_x}
    for ratio in (1.5, 2.0, 2.5, 3.0, 4.0, 6.0):
        y, max_gr = compressor(x, med_x - 3.0, ratio, sr=sr)
        r_y, med_y = dyn_range(y, sr)
        if r_y <= r_ref + 1.0:
            break
    y = y * 10 ** ((med_x - med_y) / 20)  # make-up
    return y.astype(np.float32), {"ratio": ratio, "threshold_db": med_x - 3.0, "range_ref": r_ref, "range_in": r_x,
                                  "range_out": r_y,
                                  "max_reduction_db": max_gr}


# ---------------------------------------------------------------- reverb: measure the original, synthesise a match
BANDS = [(0, 180), (180, 355), (355, 710), (710, 1400), (1400, 2800), (2800, 5600), (5600, 11200), (11200, 22050)]
RT_TILT = [1.1, 1.1, 1.0, 1.0, 1.0, 0.9, 0.75, 0.6]  # natural high-frequency damping


def measure_reverb(wet, dry, sr=SR):
    """Level per band (reverb vs voice), decay time, pre-delay and width of the original's vocal reverb."""
    rev = wet - dry
    dm = mono(dry)
    n, hop = 2048, 512
    fps = sr / hop
    f, _, Zd = ss.stft(dm, sr, nperseg=n, noverlap=n - hop)
    Pd = np.abs(Zd) ** 2
    Pr = np.mean([np.abs(ss.stft(rev[:, c], sr, nperseg=n, noverlap=n - hop)[2]) ** 2 for c in range(rev.shape[1])], 0)
    Bd = np.array([Pd[(f >= a) & (f < b)].sum(0) for a, b in BANDS])
    Br = np.array([Pr[(f >= a) & (f < b)].sum(0) for a, b in BANDS])
    level = 10 * np.log10(Br.sum(1) / (Bd.sum(1) + 1e-20) + 1e-20)
    # decay after phrase ends
    ed = 10 * np.log10(Bd.sum(0) + 1e-12)
    erm = 10 * np.log10(Br[2:6].sum(0) + 1e-12)
    pk = np.percentile(ed, 95)
    slopes, i = [], 0
    w, gap = int(0.35 * fps), int(0.08 * fps)
    while i < len(ed) - w - gap:
        j = i + gap
        if ed[i] > pk - 25 and (ed[j:j + w] < pk - 45).all():
            seg = erm[j:j + w]
            k = np.polyfit(np.arange(len(seg)) / fps, seg, 1)[0]
            if -200 < k < -5:
                slopes.append(k)
            i += w
        i += 1
    rt = float(np.clip(60 / -np.median(slopes), 0.3, 3.0)) if len(slopes) >= 3 else 1.0
    # pre-delay: onsets of the voice vs build-up of the reverb
    on = np.maximum(np.diff(np.log(Bd.sum(0) + 1e-9), prepend=0), 0)
    rv = np.diff(np.log(Br.sum(0) + 1e-9), prepend=0)
    xc = [np.corrcoef(on[:len(on) - k], rv[k:])[0, 1] for k in range(int(0.12 * fps))]
    predelay = float(np.clip(np.argmax(xc) / fps, 0.01, 0.06))
    corr = float(np.corrcoef(rev[:, 0], rev[:, 1])[0, 1]) if rev.ndim == 2 and rev.shape[1] == 2 else 1.0
    return {"level_db": [float(v) for v in level], "rt60": rt, "rt_events": len(slopes),
            "predelay": predelay, "width_corr": corr}


def synth_ir(params, calib_src=None, sr=SR, seed=7):
    """Stereo noise reverb with the measured per-band level, decay and width."""
    rt = params["rt60"]
    length = int(min(5.0, 2.5 * rt * max(RT_TILT)) * sr)
    rng = np.random.default_rng(seed)
    n, hop = 1024, 256
    out = []
    for ch in range(2):
        noise = rng.standard_normal(length).astype(np.float64)
        f, t, Z = ss.stft(noise, sr, nperseg=n, noverlap=n - hop)
        amp = np.ones_like(np.abs(Z))
        for (a, b), tilt in zip(BANDS, RT_TILT):
            sel = (f >= a) & (f < b)
            amp[sel] = np.exp(-6.9078 * t[None, :] / (rt * tilt))
        _, y = ss.istft(Z * amp, sr, nperseg=n, noverlap=n - hop)
        out.append(y[:length])
    ir = np.stack(out, 1)
    # stereo width: target correlation, never below 0 (mono-compatible)
    rho = float(np.clip(params.get("width_corr", 0.0), 0.0, 0.6))
    th = 0.5 * np.arcsin(rho)
    ir = np.stack([np.cos(th) * ir[:, 0] + np.sin(th) * ir[:, 1], np.sin(th) * ir[:, 0] + np.cos(th) * ir[:, 1]], 1)
    if calib_src is not None:
        ir = calibrate_ir(ir, calib_src, params["level_db"], sr)
    pre = np.zeros((int(params["predelay"] * sr), 2))
    fade = np.minimum(1, np.arange(len(ir)) / (0.005 * sr))[:, None]  # soft onset
    return np.concatenate([pre, ir * fade]).astype(np.float32)


def band_energy(x, sr=SR):
    """Per-band energy, averaged over channels (not of the mono sum: decorrelated reverb would lose 3 dB)."""
    chans = [x] if x.ndim == 1 else [x[:, c] for c in range(x.shape[1])]
    out = 0
    for ch in chans:
        f, _, Z = ss.stft(ch, sr, nperseg=2048, noverlap=1536)
        P = np.abs(Z) ** 2
        out = out + np.array([P[(f >= a) & (f < b)].sum() for a, b in BANDS])
    return out / len(chans) + 1e-20


def calibrate_ir(ir, src, level_db, sr=SR, iters=2):
    """Scale the IR per band so that reverb(src)/src matches level_db (measured on this very voice)."""
    src = mono(src)  # whole song: tails in the pauses count, exactly as in measure_reverb
    e_src = band_energy(src, sr)
    N = 1 << int(np.ceil(np.log2(len(ir))))
    fr = np.fft.rfftfreq(N, 1 / sr)
    centers = np.log10([np.sqrt(max(a, 20) * b) for a, b in BANDS])
    for _ in range(iters):
        wet = np.stack([ss.fftconvolve(src, ir[:, c])[:len(src)] for c in range(2)], 1)
        got = 10 * np.log10(band_energy(wet, sr) / e_src)
        corr_db = np.array(level_db) - got
        g = 10 ** (np.interp(np.log10(np.maximum(fr, 20)), centers, corr_db) / 20)
        ir = np.fft.irfft(np.fft.rfft(ir, n=N, axis=0) * g[:, None], n=N, axis=0)[:len(ir)]
    return ir


def apply_reverb(dry_mono, ir, send_gain=None):
    src = dry_mono if send_gain is None else dry_mono * send_gain
    wet = np.stack([ss.fftconvolve(src, ir[:, c])[:len(src)] for c in range(2)], 1)
    return wet.astype(np.float32)


# ---------------------------------------------------------------- level automation (stage 5)
def level_match_gain(ref, x, sr=SR, win=0.4, max_dev=8.0, exclude=None):
    """Gain curve (linear, per sample) making x follow ref's phrase loudness."""
    lr, lx = env_db(ref, win, sr), env_db(x, win, sr)
    n = min(len(lr), len(lx))
    lr, lx = lr[:n], lx[:n]
    act = active_mask(lr, 30) & active_mask(lx, 30)
    if exclude is not None:
        act &= ~exclude[:n]
    if act.sum() < CTRL:  # less than a second of overlap: just match the median
        return np.ones(len(x), np.float32), 0.0
    d = lr - lx
    base = float(np.median(d[act]))
    t = np.arange(n)
    g = np.interp(t, t[act], np.clip(d[act], base - max_dev, base + max_dev))
    k = CTRL  # 1 s smoothing
    g = np.convolve(np.pad(g, k, mode="edge"), np.ones(k) / k, mode="same")[k:-k]
    return ctrl_to_samples(10 ** (g / 20), len(x), sr), base


def cap_to_reference(x, ref, over_db=3.0, loud_q=90, max_cut_db=6.0, sr=SR, win=0.1):
    """After the phrase-level matching (1 s smoothing), a loud note of yours may still jump out where the
    original's is quieter (Industry Baby: backing stem peaks +2.9 dBFS against the original's -0.7). Among your
    loudest moments (above your loud_q percentile, 100 ms level), wherever you exceed the original by more than
    over_db (beyond the typical offset), you are turned down to that ceiling — fast down, slow back. Quieter
    moments are left alone: there short-term mismatches are only timing. Returns (x, info)."""
    from scipy.ndimage import minimum_filter1d
    lr, lx = env_db(ref, win, sr), env_db(x, win, sr)
    n = min(len(lr), len(lx))
    lr, lx = lr[:n], lx[:n]
    floor = np.percentile(lr, 95) - 30
    act = (lr > floor) & (lx > np.percentile(lx, 95) - 30)
    if act.sum() < CTRL:
        return x, {"pct": 0.0, "max_db": 0.0}
    off = float(np.median(lx[act] - lr[act]))
    ceil = np.maximum(lr, floor) + off + over_db
    loud = lx > np.percentile(lx[act], loud_q)
    red = np.where(loud, np.clip(ceil - lx, -max_cut_db, 0.0), 0.0)   # dB, <= 0; never a hole in a note
    red = minimum_filter1d(red, 7)                       # +-30 ms ahead and behind
    k = int(0.15 * CTRL)                                 # 150 ms smoothing back up
    sm = np.convolve(np.pad(red, k, mode="edge"), np.ones(k) / k, mode="same")[k:-k]
    red = np.minimum(red, sm)
    pct = float(np.mean(red[act] < -1.0) * 100)
    g = ctrl_to_samples(10 ** (red / 20), len(x), sr)
    y = x * (g[:, None] if x.ndim == 2 else g)
    return y.astype(np.float32), {"pct": pct, "max_db": float(-red.min())}


# ---------------------------------------------------------------- loudness (ITU-R BS.1770) and limiter
def _k_filter(sr):
    # pyloudnorm / BS.1770 K-weighting: high shelf + high pass, designed for any sample rate
    G, Q, fc = 3.99984385397, 0.7071752369554193, 1681.9744509555319
    A = 10 ** (G / 40)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * Q)
    b1 = [A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha), -2 * A * ((A - 1) + (A + 1) * np.cos(w0)),
          A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)]
    a1 = [(A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha, 2 * ((A - 1) - (A + 1) * np.cos(w0)),
          (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha]
    Q2, fc2 = 0.5003270373253953, 38.13547087613982
    w0 = 2 * np.pi * fc2 / sr
    alpha = np.sin(w0) / (2 * Q2)
    b2 = [(1 + np.cos(w0)) / 2, -(1 + np.cos(w0)), (1 + np.cos(w0)) / 2]
    a2 = [1 + alpha, -2 * np.cos(w0), 1 - alpha]
    return (np.array(b1) / a1[0], np.array(a1) / a1[0]), (np.array(b2) / a2[0], np.array(a2) / a2[0])


def loudness_lufs(x, sr=SR):
    x = stereo(x).astype(np.float64)
    (b1, a1), (b2, a2) = _k_filter(sr)
    y = ss.lfilter(b2, a2, ss.lfilter(b1, a1, x, axis=0), axis=0)
    blk, hop = int(0.4 * sr), int(0.1 * sr)
    ms = np.array([(y[i:i + blk] ** 2).mean(0).sum() for i in range(0, len(y) - blk, hop)])
    lk = -0.691 + 10 * np.log10(ms + 1e-20)
    g = ms[lk > -70]
    if not len(g):
        return -70.0
    rel = -0.691 + 10 * np.log10(g.mean()) - 10
    g2 = ms[(lk > -70) & (lk > rel)]
    return float(-0.691 + 10 * np.log10(g2.mean()))


def soft_clip(x, thr_db=0.0):
    """Oversampled soft clipper (as in loud pop masters before the limiter): shaves short peaks above ~-3 dB of
    thr so the limiter does not have to duck whole beats for them."""
    up = ss.resample_poly(x, 4, 1, axis=0)
    t = 10 ** (thr_db / 20)
    a = np.abs(up)
    k = 0.7 * t
    y = np.where(a <= k, up, np.sign(up) * (k + (t - k) * np.tanh((a - k) / (t - k))))
    return ss.resample_poly(y, 1, 4, axis=0)[:len(x)].astype(np.float32)


def limiter(x, ceiling_db=-1.0, lookahead=0.005, release=0.08, sr=SR):
    """Peak limiter with look-ahead; peaks measured on a 4x-oversampled signal (approximate true peak)."""
    c = 10 ** (ceiling_db / 20)
    up = np.abs(ss.resample_poly(x, 4, 1, axis=0)).max(1).reshape(-1, 4).max(1)[:len(x)]
    need = np.minimum(1.0, c / np.maximum(up, 1e-9))
    la = int(lookahead * sr)
    from scipy.ndimage import minimum_filter1d
    need = minimum_filter1d(need, size=2 * la + 1, origin=0)
    # release smoothing (only upward movement is slowed)
    a = np.exp(-1 / (release * sr))
    g = np.empty_like(need)
    prev = 1.0
    blk = 64
    for i in range(0, len(need), blk):
        target = need[i:i + blk].min()
        prev = target if target < prev else a ** blk * prev + (1 - a ** blk) * target
        g[i:i + blk] = prev
    g = np.minimum(g, need)
    return (x * g[:, None]).astype(np.float32), float(20 * np.log10(g.min()))


# ---------------------------------------------------------------- density ("radio" glue)
def crest_db(x, blk=0.4, sr=SR):
    """Median peak-to-RMS of 400 ms blocks in the active part: how 'peaky' vs 'dense' a signal is."""
    x = stereo(x)
    m = np.abs(x).max(1)
    L = int(blk * sr)
    k = len(m) // L
    if k < 3:
        return 0.0
    pk = 20 * np.log10(m[:k * L].reshape(k, L).max(1) + 1e-9)
    rms = 10 * np.log10((x[:k * L] ** 2).mean(1).reshape(k, L).mean(1) + 1e-12)
    act = rms > np.percentile(rms, 95) - 20
    return float(np.median((pk - rms)[act]))


def fast_env(x, attack=0.001, release=0.05, sr=SR):
    """Peak envelope at sample rate: instant-ish attack via a short max filter, smooth release."""
    from scipy.ndimage import maximum_filter1d
    m = np.abs(stereo(x)).max(1)
    m = maximum_filter1d(m, size=max(1, int(attack * sr)) * 2 + 1)
    a = np.exp(-1 / (release * sr))
    # one-pole release on a decimated grid (fast), then hold the max of raw and smoothed
    hop = 32
    d = m[: len(m) // hop * hop].reshape(-1, hop).max(1)
    sm = ss.lfilter([1 - a ** hop], [1, -a ** hop], d)
    sm = np.maximum(sm, d)
    return np.repeat(sm, hop)[: len(m)] if len(sm) * hop >= len(m) else np.pad(np.repeat(sm, hop), (0, len(m) - len(sm) * hop), mode="edge")


def peak_comp(x, thr_db, ratio=4.0, knee=4.0, sr=SR, attack=0.001, release=0.05):
    env = 20 * np.log10(fast_env(x, attack=attack, release=release, sr=sr) + 1e-9)
    over = env - thr_db
    gr = np.where(over <= -knee / 2, 0.0, np.where(over >= knee / 2, over * (1 - 1 / ratio),
                                                    (1 - 1 / ratio) * (over + knee / 2) ** 2 / (2 * knee)))
    g = 10 ** (-gr / 20)
    return (x * (g[:, None] if x.ndim == 2 else g)).astype(np.float32)


def saturate(x, drive_db=3.0):
    """Gentle tanh saturation (4x oversampled), level-compensated: adds harmonic density, rounds peaks."""
    up = ss.resample_poly(x, 4, 1, axis=0)
    g = 10 ** (drive_db / 20)
    pk = np.abs(up).max() + 1e-9
    y = np.tanh(up / pk * g) / np.tanh(g) * pk
    y = ss.resample_poly(y, 1, 4, axis=0)[: len(x)]
    return (y * np.sqrt((x ** 2).mean() / ((y ** 2).mean() + 1e-20))).astype(np.float32)


def saturate_low(x, drive_db=2.0, fc=4000.0, sr=SR):
    """Saturate only below fc: the body of the voice gets denser, sibilants and air get no extra grit."""
    sos = ss.butter(4, fc, "lowpass", fs=sr, output="sos")
    lo = ss.sosfiltfilt(sos, x, axis=0).astype(np.float32)
    return (saturate(lo, drive_db) + (x - lo)).astype(np.float32)


def loud_excerpt(x, seconds=40, sr=SR):
    n = int(seconds * sr)
    if len(x) <= n:
        return x
    lv = env_db(x, 1.0, sr)
    start = int(np.argmax(np.convolve(lv, np.ones(seconds * CTRL), mode="valid")) / CTRL * sr)
    return x[start:start + n]


def match_density(x, target_crest, sr=SR, gentle=False):
    """Fast compression + light saturation until the crest factor matches target_crest.
    The threshold is searched on a loud 40 s excerpt, then applied once to the whole song.
    gentle=True (the backing): slower compressor (5 ms / 150 ms) and no saturation — on long hums (Animal's
    "mmm") the fast one follows the waveform itself and, with the saturation, sounds like a radio."""
    att, rel, drive = (0.005, 0.15, 0.0) if gentle else (0.001, 0.05, 2.0)

    def proc(z, thr):
        y = peak_comp(z, thr, sr=sr, attack=att, release=rel)
        return saturate_low(y, drive, sr=sr) if drive > 0 else y
    c0 = crest_db(x, sr=sr)
    if c0 <= target_crest + 0.3:
        return x, {"crest_in": c0, "crest_out": c0, "threshold_db": None}
    ex = loud_excerpt(x, sr=sr)
    target_ex = target_crest + (crest_db(ex, sr=sr) - c0)  # same improvement on the excerpt
    rms = 10 * np.log10((stereo(ex) ** 2).mean() + 1e-12)
    lo, hi = rms, 20 * np.log10(np.abs(ex).max() + 1e-9)
    thr = None
    for _ in range(9):
        mid = (lo + hi) / 2
        c = crest_db(proc(ex, mid), sr=sr)
        if c > target_ex:
            hi = mid  # not dense enough: lower the threshold
        else:
            lo, thr = mid, mid
        if abs(c - target_ex) < 0.2:
            thr = mid
            break
    if thr is None:
        thr = lo
    y = proc(x, thr)
    y = y * np.sqrt((stereo(x) ** 2).mean() / ((stereo(y) ** 2).mean() + 1e-20))  # keep loudness
    return y.astype(np.float32), {"crest_in": c0, "crest_out": crest_db(y, sr=sr), "threshold_db": thr}


def bus_glue(mix, target_gr_db=1.5, sr=SR):
    """Mix-bus 'glue' compressor, 2:1, attack 30 ms, release 150 ms, stereo-linked.
    The threshold is set so that loud passages get about target_gr_db of gain reduction:
    music then gives way slightly when the voice comes in — like in a mastered record."""
    lv = env_db(mix, 0.01, sr)
    loud = lv > np.percentile(lv, 95) - 10
    ratio = 2.0
    thr = float(np.median(lv[loud])) - target_gr_db / (1 - 1 / ratio)
    y, max_gr = compressor(mix, thr, ratio, attack=0.03, release=0.15, knee=6.0, sr=sr)
    gr = 10 * np.log10((mix[loud.repeat(int(sr / CTRL))[:len(mix)]] ** 2).mean() /
                       (y[loud.repeat(int(sr / CTRL))[:len(mix)]] ** 2).mean())
    return y, {"threshold_db": thr, "gr_loud_db": float(gr), "gr_max_db": float(max_gr)}


# ---------------------------------------------------------------- consonants (s, sh, f, t, k, breaths)
CONS_N, CONS_HOP = 1024, 256
CONS_LO, CONS_HI = 3000.0, 4500.0  # soft crossover: above it consonants come from the original


def _stft(x, sr):
    from scipy.signal import stft
    return stft(x, sr, nperseg=CONS_N, noverlap=CONS_N - CONS_HOP)


def _istft(Z, sr, n):
    from scipy.signal import istft
    _, y = istft(Z, sr, nperseg=CONS_N, noverlap=CONS_N - CONS_HOP)
    return fit_len(y.astype(np.float32), n)


def _frames_mask(mask_fps, n_frames, sr, fps=100):
    """Bool mask at fps -> soft 0..1 mask on the STFT grid with ~10 ms ramps (no clicks at the edges)."""
    t = np.arange(n_frames) * CONS_HOP / sr
    idx = np.clip((t * fps).astype(int), 0, len(mask_fps) - 1)
    m = mask_fps[idx].astype(np.float32)
    k = max(1, int(0.01 * sr / CONS_HOP))
    m = np.convolve(m, np.ones(2 * k + 1, np.float32) / (2 * k + 1), mode="same")
    return np.clip(m * 1.5, 0, 1)  # full replacement inside, ramps only at the edges


def _local_ratio(Pa, Pb, skip, sr, smooth_s=0.5):
    """Level of b relative to a (power ratio) per frame, from the frames not in `skip`, smoothed over ~0.5 s."""
    ea, eb = Pa.sum(0), Pb.sum(0)
    ok = (~skip) & (ea > ea.max() * 1e-4) & (eb > eb.max() * 1e-5)
    if ok.sum() < 10:
        return np.ones(len(ea))
    d = 10 * np.log10(eb[ok] / ea[ok])
    t = np.arange(len(ea))
    r = np.interp(t, t[ok], d)
    k = max(1, int(smooth_s * sr / CONS_HOP))
    r = np.convolve(np.pad(r, k, mode="edge"), np.ones(k) / k, mode="same")[k:-k]
    return 10 ** (r / 10)


def transplant_consonants(src, out, fric_fps, sr):
    """RVC hears the voice at 16 kHz and draws everything above 8 kHz itself: 's' comes out duller and ~10 dB
    quieter, close to 'sh'. Unvoiced consonants are almost the same in every voice, so in those moments the band
    above ~4 kHz is taken from the original vocal (level-matched to your vocal around it). Vowels stay yours.
    src, out: mono float, same sr, time-aligned. fric_fps: bool per 10 ms (from phonetics.consonant_mask)."""
    n = len(out)
    f, _, A = _stft(fit_len(src, n), sr)
    _, _, B = _stft(out, sr)
    m = min(A.shape[1], B.shape[1])
    A, B = A[:, :m], B[:, :m]
    Pa, Pb = np.abs(A) ** 2, np.abs(B) ** 2
    fm = _frames_mask(fric_fps, m, sr)
    ratio = _local_ratio(Pa, Pb, fm > 0, sr)
    w = np.clip((f - CONS_LO) / (CONS_HI - CONS_LO), 0, 1)[:, None]
    W = w * fm[None, :]
    C = B * (1 - W) + A * np.sqrt(ratio)[None, :] * W
    hf = f >= 4000
    sel = fm > 0.5
    before = after = 0.0
    if sel.any():
        Pc = np.abs(C) ** 2
        ref = Pa[hf][:, sel].sum() * np.mean(ratio[sel])
        before = 10 * np.log10(Pb[hf][:, sel].sum() / ref + 1e-12)
        after = 10 * np.log10(Pc[hf][:, sel].sum() / ref + 1e-12)
    return _istft(C, sr, n), {"frames_pct": float(sel.mean() * 100), "hf_before_db": float(before),
                              "hf_after_db": float(after)}


def match_consonants(ref, x, fric_fps, sr=SR, max_cut=10.0, max_boost=6.0, split=7000.0):
    """The last word on sibilance (replaces de-essers): in consonant moments, x above 4 kHz follows the original
    vocal's consonant-to-vowel balance — not louder ('hiss') and not quieter ('lisp') than in the original.
    Two bands, 4–7 kHz and above 7 kHz, each with its own gain: a model's "s" has its energy lower than a real
    one, and one gain for everything above 4 kHz left the sparkle of s/t 3–4 dB short (Industry Baby).
    ref, x: time-aligned (mono or stereo). Gains are computed on mono and applied to every channel."""
    xs = stereo(x) if x.ndim == 2 else x[:, None]
    n = len(xs)
    f, _, A = _stft(fit_len(mono(ref), n), sr)
    _, _, B = _stft(mono(xs) if xs.shape[1] > 1 else xs[:, 0], sr)
    m = min(A.shape[1], B.shape[1])
    Pa, Pb = np.abs(A[:, :m]) ** 2, np.abs(B[:, :m]) ** 2
    fm = _frames_mask(fric_fps, m, sr)
    ratio = _local_ratio(Pa, Pb, fm > 0, sr)
    rel = np.exp(-1 / max(1.0, 0.02 * sr / CONS_HOP))  # ~20 ms release towards 0 dB

    def gains(band):
        g_db = 10 * np.log10((Pa[band].sum(0) * ratio + 1e-14) / (Pb[band].sum(0) + 1e-14))
        g_db = np.clip(g_db, -max_cut, max_boost) * fm
        sm = np.zeros_like(g_db)
        for i, v in enumerate(g_db):
            p = sm[i - 1] * rel if i else 0.0
            sm[i] = v if abs(v) >= abs(p) else p
        return sm
    lo_b = (f >= 4000) & (f < split)
    hi_b = f >= split
    sm_lo = gains(lo_b)
    sm_hi = gains(hi_b) if hi_b.any() and Pa[hi_b].sum() > 0 else sm_lo
    w = np.clip((f - CONS_LO) / (CONS_HI - CONS_LO), 0, 1)[:, None]
    mix = np.clip((f - (split - 1000)) / 2000, 0, 1)[:, None]          # 6–8 kHz: from the lower to the upper gain
    G = 10 ** ((w * ((1 - mix) * sm_lo[None, :] + mix * sm_hi[None, :])) / 20)
    out = np.zeros_like(xs)
    for ch in range(xs.shape[1]):
        _, _, Z = _stft(xs[:, ch], sr)
        Z = Z[:, :m] * G
        out[:, ch] = _istft(Z, sr, n)
    sel = fm > 0.5
    sm = (sm_lo + sm_hi) / 2
    info = {"cut_pct": float((sm[sel] < -1).mean() * 100) if sel.any() else 0.0,
            "boost_pct": float((sm[sel] > 1).mean() * 100) if sel.any() else 0.0,
            "median_db": float(np.median(sm[sel])) if sel.any() else 0.0,
            "median_lo_db": float(np.median(sm_lo[sel])) if sel.any() else 0.0,
            "median_hi_db": float(np.median(sm_hi[sel])) if sel.any() else 0.0}
    return (out if x.ndim == 2 else out[:, 0]).astype(np.float32), info


# ---------------------------------------------------------------- vocal bus vs the original's vocals; closeness report
def _mid_side_ratio(x, act):
    x = stereo(x)
    mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    return (side[act] ** 2).mean() / ((mid[act] ** 2).mean() + 1e-20)


def _active_samples(ref, x, sr):
    a = active_mask(env_db(ref, 0.4, sr), 30) & active_mask(env_db(x, 0.4, sr), 30)
    return ctrl_to_samples(a.astype(np.float32), len(x), sr) > 0.5


def match_vocal_bus(vocals_ref, bus, sr=SR, keep=None):
    """Your whole vocal bus (lead + reverb + backing) against the original's vocals, after everything else:
    small residual tone (±2 dB per octave, highs never boosted over +1 dB), level (±2 dB) and stereo width."""
    n = min(len(vocals_ref), len(bus))
    ref, x = stereo(vocals_ref[:n]), stereo(bus[:n])
    curve = eq_match_curve(ref, x, sr, limit_db=2.0, hf_boost_cap=1.0, cut_db=4.0, keep=keep)
    xe = apply_eq(x, curve, sr)
    act = _active_samples(ref, xe, sr)
    if act.sum() < sr:
        return {"curve": [0.0] * len(OCT), "level_db": 0.0, "side_gain": 1.0}
    lvl = 10 * np.log10((ref[act] ** 2).mean() / ((xe[act] ** 2).mean() + 1e-20))
    r_ref, r_x = _mid_side_ratio(ref, act), _mid_side_ratio(xe, act)
    side = float(np.clip(np.sqrt((r_ref + 1e-6) / (r_x + 1e-6)), 0.5, 2.0))
    return {"curve": [float(v) for v in curve], "level_db": float(np.clip(lvl, -2.0, 2.0)), "side_gain": side,
            "width_ref": float(10 * np.log10(r_ref + 1e-9)), "width_in": float(10 * np.log10(r_x + 1e-9))}


def apply_vocal_bus(x, corr, sr=SR):
    y = stereo(apply_eq(stereo(x), np.array(corr["curve"]), sr)) * 10 ** (corr["level_db"] / 20)
    if abs(corr["side_gain"] - 1) > 0.01:
        mid, side = (y[:, 0] + y[:, 1]) / 2, (y[:, 0] - y[:, 1]) / 2 * corr["side_gain"]
        y = np.stack([mid + side, mid - side], 1)
    return y.astype(np.float32)


def closeness(orig, cover, sr=SR):
    """How close the finished cover is to the original: loudness, tone per octave, density, stereo width."""
    n = min(len(orig), len(cover))
    o, c = stereo(orig[:n]), stereo(cover[:n])
    f, Po = long_term_spectrum(o, sr)
    _, Pc = long_term_spectrum(c, sr)
    d = octave_db(f, Pc) - octave_db(f, Po)
    d = d - np.nanmedian(d[2:7])
    act = _active_samples(o, c, sr)
    return {"lufs_diff": float(loudness_lufs(c, sr) - loudness_lufs(o, sr)),
            "tone_db": [float(v) for v in np.nan_to_num(d)],
            "tone_max_db": float(np.nanmax(np.abs(d[1:]))),
            "crest_diff": float(crest_db(c, sr=sr) - crest_db(o, sr=sr)),
            "width_diff_db": float(10 * np.log10((_mid_side_ratio(c, act) + 1e-9) / (_mid_side_ratio(o, act) + 1e-9)))}
