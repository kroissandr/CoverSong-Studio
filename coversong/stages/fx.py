"""Stage 4: process your converted vocal the way the original vocal was processed.

The original gives us a time-aligned pair: its dry vocal (lead_dry) and the same vocal with all effects (lead_wet).
  1. align your vocal to the original (sub-frame accuracy)
  2. EQ: move your tonal balance toward the original's mix (body up to +6 dB, mids ±4, highs up to +3, cuts to
     -8) so your timbre stays yours; where the original is brighter still, saturation-style overtones in
     parallel (dsp.match_brightness) — a produced vocal's sheen, without lifting the model's noise — then the
     vowels' air (above 7 kHz) with a shelf (fx.vowel_presence / fx.backing_brightness: optional, off)
  3. compression: match the original's loudness range, then its density (peak-to-average), if yours is peakier
  4. consonants: s/sh/t above 4 kHz follow the original's consonant-to-vowel balance (no hiss, no lisp)
  5. silences as tight as the original's (RVC's noise floor would otherwise sound like a room)
  6. effects (vocal_fx): a plate-style reverb fitted to how the original's effects ring out after phrase ends,
     echoes where the original has them, doubles if the original's vocal is thicker than reverb explains
Outputs (44.1 kHz stereo, float): lead_dry.wav + lead_reverb.wav (mixed in stage 5, so the reverb slider is instant),
backing.wav (EQ-matched backing vocals).
"""
import json
import os

import numpy as np

from .. import dsp, vocal_fx
from ..util import log


def read(path, sr_out=dsp.SR):
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    return dsp.to_sr(y, sr, sr_out)


def write(path, y, sr=dsp.SR):
    import soundfile as sf
    sf.write(path, y.astype(np.float32), sr, subtype="FLOAT")


class Roughness:
    """Share (%) of rough frames (periodicity < 0.6) on the notes both you and the original sing, loud parts —
    the frames are fixed once (from the original and your voice before brightening), so versions compare fairly."""

    def __init__(self, ref, you):
        from .. import f0_extra, quality
        self.q = quality
        r16 = self.r16(ref)
        fo, _ = f0_extra.swift(r16)
        fy, _ = f0_extra.swift(self.r16(you))
        po, do = quality.periodicity(r16, 16000)
        k = min(len(fo), len(fy), len(po))
        self.mask = (do[:k] > np.percentile(do[:k], 95) - 25) & (fo[:k] > 0) & (fy[:k] > 0)
        self.orig = self.share(po)

    @staticmethod
    def r16(x):
        return dsp.to_sr(dsp.mono(x), dsp.SR, 16000).astype(np.float32)

    def share(self, p):
        k = min(len(p), len(self.mask))
        m = self.mask[:k]
        return float(np.mean(p[:k][m] < 0.6) * 100) if m.any() else 0.0

    def __call__(self, x):
        return self.share(self.q.periodicity(self.r16(x), 16000)[0])


def eq_by_parts(ref, x, segs, n, sr=dsp.SR, **kw):
    """Backing EQ matched separately on the wordless places (hums, oohs) and on the sung rest: one curve for the
    whole song pushed Animal's "mmm" toward the rest's balance (+1.5 dB at 2 kHz, -4 at 16 kHz) — a radio sound.
    Returns (x, curve of the rest, curve of the wordless places or None)."""
    m = np.zeros(n, bool)
    for a, b in segs or []:
        m[int(a * sr):int(b * sr)] = True
    if m.sum() < 3 * sr or (~m).sum() < 3 * sr:
        c = dsp.eq_match_curve(ref, x, **kw)
        return dsp.apply_eq(x, c), c, None
    c_rest = dsp.eq_match_curve(ref[~m], x[~m], **kw)
    c_w = dsp.eq_match_fine(ref[m], x[m])          # third-octave: a hum's resonance turns within an octave
    w = np.convolve(m.astype(np.float32), np.ones(int(0.05 * sr)) / int(0.05 * sr), "same")[:, None]   # 50 ms fades
    y = dsp.apply_eq(x, c_rest) * (1 - w) + dsp.apply_eq(x, c_w, centers=dsp.THIRDS, taps=4095) * w
    return y.astype(np.float32), c_rest, c_w


def segs_to_ctrl(segs, n):
    m = np.zeros(n, bool)
    for a, b in segs or []:
        m[int(a * dsp.CTRL):int(b * dsp.CTRL)] = True
    return m


def run(cfg, cln, sep, stage, clean_info=None):
    clean_info = clean_info or {}
    opt = cfg.get("fx") or {}
    wet = read(sep["lead_wet"])
    dry_ref = read(sep["lead_dry"])
    n = len(dry_ref)
    you = dsp.fit_len(dsp.mono(read(cln["lead"])), n)

    lag = dsp.align_lag(dsp.mono(dry_ref), you)
    you = dsp.shift(you, lag)

    keep = dsp.own_low_band(clean_info.get("lead_f0"))
    # body may be added up to +6 dB, highs up to +3 (the rest of the sheen comes from overtones below), and a
    # boxy band is cut more precisely (less smoothing across octaves)
    curve = dsp.eq_match_curve(dry_ref, you, cut_db=8.0, keep=keep, low_limit_db=6.0, hf_boost_cap=3.0,
                               smooth=(0.15, 0.7, 0.15))
    you = dsp.apply_eq(you, curve)
    log("EQ toward the original (dB, 63 Hz…16 kHz): " + " ".join(f"{v:+.1f}" for v in curve)
        + (f"; your lows {keep[0]:.0f}–{keep[1]:.0f} Hz kept (you sing lower than the original)" if keep else ""))
    cons_path = clean_info.get("cons_lead")
    cons = np.load(cons_path) if cons_path and os.path.exists(cons_path) else None
    vow = None if cons is None else ~cons
    presence = bool(opt.get("vowel_presence", False))
    eq_you = you
    you, br = dsp.match_brightness(dry_ref, eq_you, vowels=vow, presence=presence)
    if opt.get("rough_guard", True) and (br["amount"] > 0 or br["air_db"] > 0 or br["presence_db"] > 0):
        # brightness lifts what your highs are made of — on some songs mostly the model's noise, and the voice
        # turns raspy (The Feels: rough frames 5.1 -> 6.8 %, the original 2.4 %). It may add at most
        # fx.rough_allow points of roughness over your voice before it, or reach the original's own roughness
        rough = Roughness(dry_ref, eq_you)
        base, now = rough(eq_you), rough(you)
        limit = max(base + float(opt.get("rough_allow", 1.0)), rough.orig)
        scale = 1.0
        if now > limit:
            first = now
            for scale in (0.75, 0.5, 0.25, 0.0):
                you, br = dsp.match_brightness(dry_ref, eq_you, vowels=vow, presence=presence, scale=scale)
                now = rough(you)
                if now <= limit:
                    break
            log(f"Brightness limited so the voice does not turn raspy: rough frames {first:.1f}% → {now:.1f}% (before "
                f"brightening {base:.1f}%, the original {rough.orig:.1f}%, allowed {limit:.1f}%), brightness ×{scale:g}")
    if br["amount"] > 0:
        log(f"Vowel brightness like the original's: overtones added (saturation) — 2–7 kHz against the body "
            f"{br['before_db']:+.1f} → {br['after_db']:+.1f} dB (the original {br['ref_db']:+.1f})")
    if br["presence_db"] > 0:
        log(f"Vowels' 4–7 kHz like the original's: +{br['presence_db']:.1f} dB")
    if br["air_db"] > 0:
        log(f"Vowels' air above 7 kHz like the original's: +{br['air_db']:.1f} dB")

    you, dyn = dsp.match_dynamics(dry_ref, you)
    if dyn["ratio"] > 1:
        log(f"Compression: {dyn['ratio']:.1f}:1, loudness range {dyn['range_in']:.1f} → {dyn['range_out']:.1f} dB "
            f"(the original {dyn['range_ref']:.1f})")
    else:
        log("No compression needed: the dynamics already match the original")

    you, dens = dsp.match_density(you, dsp.crest_db(dry_ref))
    if dens["threshold_db"] is not None:
        log(f"Vocal density like the original's: peak/average {dens['crest_in']:.1f} → {dens['crest_out']:.1f} dB")

    if cons is not None:
        you, cm = dsp.match_consonants(dry_ref, you, cons)
        log(f"Consonants like the original's: turned down {cm['cut_pct']:.0f}%, up {cm['boost_pct']:.0f}% of the time "
            f"(median {cm['median_db']:+.1f} dB)")

    you, gated = vocal_fx.gate_to_reference(you, dry_ref)
    if gated > 0:
        log(f"Silence like the original's: the model's noise in pauses turned down ({gated:.0f}% of the time)")

    rp = dsp.measure_reverb(wet, dry_ref)
    # vocalizations sit a little further back: +3 dB more reverb send there
    send = np.ones(n, np.float32)
    voc = clean_info.get("lead_vocalizations")
    if voc:
        m = segs_to_ctrl(voc, int(n / dsp.SR * dsp.CTRL) + 1)
        send = dsp.ctrl_to_samples(np.where(m, 10 ** (3 / 20), 1.0), n)
    rev, ef = vocal_fx.build_effects(you, dry_ref, wet, rp, send)
    if ef.get("fitted"):
        tg = " / ".join(f"{v:.0f}" for v in ef["target"])
        got = " / ".join(f"{v:.0f}" for v in ef["got"])
        log(f"Effects from the original: reverb {ef['rt']:.2f} s, pre-delay {rp['predelay'] * 1000:.0f} ms; "
            f"tail after phrases {got} dB (the original {tg}; 20–60/60–120/120–250/250–450 ms)")
    else:
        log(f"Effects from the original: reverb {ef['rt']:.2f} s (few clean phrase ends — from the overall measurement)")
    if ef.get("echoes"):
        log("Echoes like the original's: " + ", ".join(f"{t * 1000:.0f} ms ({g:+.0f} dB)" for t, g in ef["echoes"]))
    if ef.get("doubles_db") is not None:
        log(f"Vocal doubles (like the original's layered vocal): {ef['doubles_db']:+.0f} dB under the voice")
    if ef.get("capped_db"):
        log(f"Effects turned down {ef['capped_db']:.1f} dB: no more reverb than the original has")

    out = {"lead_dry": stage.path("lead_dry.wav"), "lead_reverb": stage.path("lead_reverb.wav")}
    write(out["lead_dry"], dsp.stereo(you))
    write(out["lead_reverb"], rev)

    if cln.get("backing"):
        from .separate import backing_src
        src_b = backing_src(sep)
        dry_b = src_b != sep["backing"]          # sung from the de-reverbed backing: reverb goes back on here
        bk_ref = dsp.fit_len(read(src_b), n)
        bk = dsp.fit_len(dsp.stereo(read(cln["backing"])), n)
        bk = dsp.shift(bk, dsp.align_lag(dsp.mono(bk_ref), dsp.mono(bk)))
        bk, bcurve, bcurve_w = eq_by_parts(bk_ref, bk, clean_info.get("backing_vocalizations"), n, keep=keep)
        if bcurve_w is not None:
            log("Backing: wordless parts matched to the original more finely (third octaves, "
                f"{bcurve_w.min():+.1f} to {bcurve_w.max():+.1f} dB); the rest: " + " ".join(f"{v:+.1f}" for v in bcurve))
        # the backing gets the lead's treatment too: loudness range and density like the original's backing
        # (Bite Me: the cover's backing was 6 dB peakier and jumped 7–9 dB over the original in places)
        bk, dyn_b = dsp.match_dynamics(bk_ref, bk)
        bk, den_b = dsp.match_density(bk, dsp.crest_db(bk_ref), gentle=True)
        if dyn_b["ratio"] > 1 or den_b["threshold_db"] is not None:
            log(f"Backing: dynamics like the original's — range {dyn_b['range_in']:.1f} → "
                f"{dyn_b.get('range_out', dyn_b['range_in']):.1f} dB, peak/average {den_b['crest_in']:.1f} → "
                f"{den_b['crest_out']:.1f} dB (the original {dyn_b['range_ref']:.1f} / {dsp.crest_db(bk_ref):.1f})")
        bk, air_b = dsp.add_breath(bk, bk_ref)
        if air_b["added_db"] > -60:
            log(f"Backing: breathiness like the original's — clean-tone share {air_b['cover_tonal']:.2f} → "
                f"{air_b.get('after_tonal', air_b['cover_tonal']):.2f} (the original {air_b['orig_tonal']:.2f})")
        # fx.backing_brightness (off by default since 0.23.1: it lifted the model's noise, not the voice):
        # brightness like the original's backing with shelves only — overtones would sharpen the model's creak
        br_b = {"presence_db": 0.0, "air_db": 0.0}
        if opt.get("backing_brightness", False):
            bk, br_b = dsp.match_brightness(bk_ref, bk, exciter=False)
        if br_b["presence_db"] > 0 or br_b["air_db"] > 0:
            log(f"Backing: brightness like the original's — 4–7 kHz +{br_b['presence_db']:.1f} dB, "
                f"air above 7 kHz +{br_b['air_db']:.1f} dB")
        bk, gb = vocal_fx.gate_to_reference(bk, bk_ref, floor_db=-40.0)
        if gb > 0:
            log(f"Backing: the model's noise in pauses turned down ({gb:.0f}% of the time)")
        if dry_b:
            rev_b, g_b = vocal_fx.backing_reverb(bk, bk_ref, dsp.fit_len(read(sep["backing"]), n), rp, ef.get("rt"))
            if rev_b is not None:
                bk = bk + rev_b
                log(f"Backing: reverb like the original's ({ef.get('rt', rp['rt60']):.2f} s, {g_b:+.0f} dB)")
        out["backing"] = stage.path("backing.wav")
        write(out["backing"], bk)

    params = {"lag_samples": lag, "eq_db": [float(v) for v in curve], "dynamics": {k: float(v) for k, v in dyn.items()},
              "reverb": rp, "effects": {k: v for k, v in ef.items() if k != "events"}}
    with open(stage.path("fx_params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    return out, params
