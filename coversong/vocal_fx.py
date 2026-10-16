"""Vocal space measured on the original and rebuilt on your voice — the way chart pop / K-pop lead vocals are made:
an upfront, tightly edited dry voice; a short plate-style reverb whose tail is kept low; doubles for thickness and
width; echoes only where the song has them.

  gate_to_reference  — silences as tight as the original's (RVC leaves a noise floor that compression would lift)
  build_effects      — reverb fitted to how the original's effects actually ring out after phrase ends,
                       echoes found at their real time and level, doubles (ADT) if the original is thicker/wider
"""
import numpy as np
import scipy.ndimage as nd
import scipy.signal as ss

from . import dsp

SR = dsp.SR
FPS = dsp.CTRL  # 100 frames per second
WINDOWS = ((2, 6), (6, 12), (12, 25), (25, 45))  # 20–60, 60–120, 120–250, 250–450 ms after a phrase ends
WEIGHTS = np.array([0.5, 1.0, 1.0, 1.0])


def penv(x, sr=SR, win=0.02):
    """Power envelope (linear) at 100 fps, averaged over channels (decorrelated reverb keeps its energy)."""
    x = x[:, None] if x.ndim == 1 else x
    return np.mean([10 ** (dsp.env_db(x[:, c], win, sr) / 10) for c in range(x.shape[1])], 0)


def gate_to_reference(you, ref, sr=SR, floor_db=-45.0, max_cut=40.0):
    """Where the original singer is silent, you are too (down to the same level). Breaths and quiet notes that the
    original has stay: only moments more than 45 dB below the original's loud level are touched."""
    lr = dsp.env_db(ref, 0.02, sr)
    ly = dsp.env_db(you, 0.02, sr)
    n = min(len(lr), len(ly))
    lr, ly = lr[:n], ly[:n]
    pk = np.percentile(lr, 95)
    loud = lr > pk - 25
    if loud.sum() < FPS:
        return you, 0.0
    off = float(np.median((ly - lr)[loud]))
    g = np.where(lr < pk + floor_db, np.clip(lr + off - ly, -max_cut, 0.0), 0.0)
    g = nd.minimum_filter1d(g, 3)  # 30 ms hold
    out = np.zeros_like(g)
    a_open, a_close = np.exp(-1 / (0.005 * FPS)), np.exp(-1 / (0.06 * FPS))
    prev = 0.0
    for i, v in enumerate(g):
        a = a_open if v > prev else a_close
        prev = a * prev + (1 - a) * v
        out[i] = prev
    gain = dsp.ctrl_to_samples(10 ** (out / 20), len(you), sr)
    y = you * (gain[:, None] if you.ndim == 2 else gain)
    return y.astype(np.float32), float((out < -6).mean() * 100)


def phrase_ends(ref, sr=SR):
    """Frames where a sung phrase stops and the original's dry voice falls silent."""
    lv = dsp.env_db(ref, 0.02, sr)
    pk = np.percentile(lv, 95)
    ev = []
    for drop in (40, 35, 30):
        ev, i = [], 20
        while i < len(lv) - 50:
            if lv[i] > pk - 20 and (lv[i + 5:i + 45] < pk - drop).all():
                ev.append(i)
                i += 45
            i += 1
        if len(ev) >= 6:
            break
    return ev, lv


def tail_profile(p, ref_lv, events):
    """dB of a power envelope in the WINDOWS after each phrase end, relative to that phrase's level (median)."""
    lp = 10 * np.log10(p + 1e-14)
    rows = [[np.mean(lp[i + a:i + b]) - np.mean(ref_lv[i - 20:i]) for a, b in WINDOWS] for i in events
            if i + WINDOWS[-1][1] < len(lp)]
    return np.median(rows, 0) if rows else None


def detect_echoes(rev_p, ref_lv, events, max_ms=800):
    """Echoes: a bump in the effects' level after phrase ends that the decay does not explain, at the same lag in
    most phrases. Returns [(seconds, power gain)]. Conservative: needs 4+ phrase ends and a 6 dB bump."""
    if len(events) < 4:
        return []
    L = int(max_ms / 10)
    lp = 10 * np.log10(rev_p + 1e-14)
    rows = [lp[i:i + L] - np.mean(ref_lv[i - 20:i]) for i in events if i + L < len(lp)]
    if len(rows) < 4:
        return []
    m = np.median(rows, 0)
    trend = nd.median_filter(m, 15)
    out = []
    for k in ss.find_peaks(m - trend, height=6, distance=8)[0]:
        if k >= 8 and m[k] > -35:  # an audible echo, not a ripple of the decay
            out.append((k / FPS, float(10 ** (m[k] / 10))))
    return out[:3]


def adt(you, delays=(0.014, 0.023), cents=6.0, sr=SR, seed=3):
    """Two artificial doubles left/right: slightly late, slowly detuned (time-varying delay), high-passed."""
    rng = np.random.default_rng(seed)
    x = ss.sosfilt(ss.butter(2, 150, "highpass", fs=sr, output="sos"), dsp.mono(you)).astype(np.float64)
    n = len(x)
    t = np.arange(n) / sr
    outs = []
    for d in delays:
        rate = rng.uniform(0.15, 0.35)
        depth = cents / 1200 * np.log(2) / (2 * np.pi * rate)  # seconds of delay swing for ±cents
        tau = d + depth * np.sin(2 * np.pi * rate * t + rng.uniform(0, 2 * np.pi))
        outs.append(np.interp(t - tau, t, x, left=0.0))
    left, right = outs
    return np.stack([0.85 * left + 0.15 * right, 0.15 * left + 0.85 * right], 1).astype(np.float32)


def backing_reverb(bk, dry_ref, wet_ref, rp, rt=None, sr=SR):
    """Reverb for the backing vocal (sung from its de-reverbed version): the lead's fitted reverb (decay rt, band
    levels rp), scaled so that reverb-to-voice energy is the original backing's. Returns (stereo bus, gain dB) or
    (None, 0) when the original backing has no measurable reverb."""
    rev_o = wet_ref - dsp.stereo(dry_ref)
    e_dry = float((dsp.stereo(dry_ref) ** 2).sum())
    if e_dry <= 0:
        return None, 0.0
    ratio = float((rev_o ** 2).sum()) / e_dry
    if ratio < 10 ** (-35 / 10):
        return None, 0.0
    lv = list(rp["level_db"])
    lv[0] = min(lv[0], float(np.mean(lv[2:6])) - 12)
    m = dsp.mono(bk)
    ir = dsp.synth_ir(dict(rp, rt60=rt or rp["rt60"], level_db=lv), calib_src=m, sr=sr)
    w = dsp.apply_reverb(m, ir)
    e_w = float((w ** 2).sum())
    if e_w <= 0:
        return None, 0.0
    g = float(np.clip(np.sqrt(ratio * float((dsp.stereo(bk) ** 2).sum()) / e_w), 10 ** (-30 / 20), 10 ** (12 / 20)))
    return (w * g).astype(np.float32), float(20 * np.log10(g))


def echo_bus(you, echoes, sr=SR):
    x = ss.sosfilt(ss.butter(2, [300, 6000], "bandpass", fs=sr, output="sos"), dsp.mono(you))
    out = np.zeros((len(x), 2), np.float32)
    for k, (t, g) in enumerate(echoes):
        d = int(t * sr)
        e = np.r_[np.zeros(d), x[:len(x) - d]] * np.sqrt(g)
        pan = (0.7, 0.3) if k % 2 == 0 else (0.3, 0.7)
        out[:, 0] += pan[0] * e
        out[:, 1] += pan[1] * e
    return out


def build_effects(you, dry_ref, wet_ref, rp, send=None, sr=SR):
    """Effects bus (stereo) for your dry vocal, fitted to the original. Returns (bus, info)."""
    n = len(you)
    rev_o = dsp.fit_len(wet_ref - dsp.stereo(dry_ref), n)
    events, ref_lv = phrase_ends(dry_ref, sr)
    src = you if send is None else you * send
    # spectral shape as measured, but nothing below ~150 Hz: the original (often an octave higher) had little there,
    # and low reverb on a lower voice is exactly the "empty room" boom
    lv = list(rp["level_db"])
    lv[0] = min(lv[0], float(np.mean(lv[2:6])) - 12)
    p_you = penv(you, sr)
    # a phrase end only counts if the original's effects really decay there: a "tail" as loud as the phrase is
    # another voice coming in (The Feels: +30 and +11 dB after two of its four phrase ends, members trading lines)
    lp_w = 10 * np.log10(penv(wet_ref, sr) + 1e-14)
    a0, b0 = WINDOWS[0]
    events = [i for i in events if i + b0 < len(lp_w) and np.mean(lp_w[i + a0:i + b0]) - np.mean(ref_lv[i - 20:i]) < -2]
    info = {"events": len(events)}
    if len(events) < 3:  # no clean phrase ends: measured level/decay as before
        ir = dsp.synth_ir(dict(rp, level_db=lv), calib_src=you, sr=sr)
        info.update(rt=rp["rt60"], fitted=False)
        bus, info["capped_db"] = _cap_effects(dsp.apply_reverb(src, ir), you, dry_ref, rev_o, sr)
        return bus, info
    target = tail_profile(penv(wet_ref, sr), ref_lv, events)
    my_lv = 10 * np.log10(p_you + 1e-14)
    best = None
    for rt in (0.2, 0.3, 0.4, 0.55, 0.75, 1.0, 1.4, 2.0):
        ir = dsp.synth_ir(dict(rp, rt60=rt, level_db=lv), calib_src=you, sr=sr)
        w = dsp.apply_reverb(src, ir)
        p_w = penv(w, sr)
        for g_db in np.arange(-30, 7, 1.0):
            prof = tail_profile(p_you + 10 ** (g_db / 10) * p_w, my_lv, events)
            err = float(np.sum(WEIGHTS * (prof - target) ** 2))
            if best is None or err < best[0]:
                best = (err, rt, g_db, w, prof)
    err, rt, g_db, w, prof = best
    bus = w * 10 ** (g_db / 20)
    info.update(rt=rt, gain_db=g_db, fitted=True, target=[float(v) for v in target], got=[float(v) for v in prof])
    # echoes the original has after phrase ends
    echoes = detect_echoes(penv(rev_o, sr), ref_lv, events)
    if echoes:
        bus = bus + echo_bus(src, echoes, sr)
    info["echoes"] = [(round(t, 3), round(10 * np.log10(g), 1)) for t, g in echoes]
    # thickness/width while singing: what the original's effects carry beyond the fitted reverb -> doubles
    act = dsp.env_db(dry_ref, 0.02, sr) > np.percentile(dsp.env_db(dry_ref, 0.02, sr), 95) - 25
    act_s = dsp.ctrl_to_samples(act.astype(np.float32), n, sr) > 0.5
    e_o = (rev_o[act_s] ** 2).mean() / ((dsp.stereo(dry_ref)[:n][act_s] ** 2).mean() + 1e-20)
    e_ours = (bus[act_s] ** 2).mean() / ((dsp.stereo(you)[act_s] ** 2).mean() + 1e-20)
    deficit = e_o - e_ours
    info["doubles_db"] = None
    if deficit > 10 ** (-18 / 10):
        dbl = adt(src, sr=sr)
        e_d = (dbl[act_s] ** 2).mean() / ((dsp.stereo(you)[act_s] ** 2).mean() + 1e-20)
        g = min(deficit, 10 ** (-6 / 10)) / (e_d + 1e-20)
        bus = bus + dbl * np.sqrt(g)
        info["doubles_db"] = float(10 * np.log10(min(deficit, 10 ** (-6 / 10))))
    bus, info["capped_db"] = _cap_effects(bus, you, dry_ref, rev_o, sr)
    return bus.astype(np.float32), info


def _cap_effects(bus, you, dry_ref, rev_o, sr=SR, over_db=1.0):
    """Never wetter than the original: the effects' energy against the voice while singing at most the original's
    (+over_db). Industry Baby and HOT land within 0.1 dB anyway; The Feels was 4.7 dB wetter. Returns (bus, dB)."""
    n = len(bus)
    lv = dsp.env_db(dry_ref, 0.02, sr)
    act = dsp.ctrl_to_samples((lv > np.percentile(lv, 95) - 25).astype(np.float32), n, sr) > 0.5
    d = dsp.stereo(dry_ref)[:n]
    e_o = (rev_o[:n][act] ** 2).mean() / ((d[act] ** 2).mean() + 1e-20)
    e_ours = (bus[act] ** 2).mean() / ((dsp.stereo(you)[:n][act] ** 2).mean() + 1e-20)
    over = 10 * np.log10((e_ours + 1e-20) / (e_o + 1e-20)) - over_db
    if over <= 0:
        return bus, 0.0
    return (bus * 10 ** (-over / 20)).astype(np.float32), float(over)
