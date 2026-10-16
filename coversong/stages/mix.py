"""Stage 5: mix and master.

  - your vocal follows the original vocal's loudness phrase by phrase (so it sits in the mix exactly like the original)
  - backing vocals likewise, against the original backing; loud notes that still jump out over the original's
    are capped to it
  - passages where the original changes its vocal's tone for a while — radio, muffled (coversong/sections.py,
    mix.section_fx): the same change on yours, in the same places
  - singing an octave lower: the low end (60–250 Hz) of the lead (≤ -6 dB) and of the backing (≤ -10 dB) toward
    the original's balance (mix.low_cut_octave_down)
  - sliders: reverb (×, 1 = like the original) and vocal level (dB)
  - the whole vocal bus (lead + reverb + backing) is checked against the original's vocals: residual tone (±2 dB),
    level (±2 dB) and stereo width are corrected — measured with neutral sliders, so the sliders stay relative
  - one sound: as close to the original as measured — the mix bus is glued only as much as the original is denser
  - at the end the finished cover is compared with the original (loudness, tone per octave, density, width)
  - master: same loudness (LUFS) as the original song, peak limiter at −1 dBTP
  - exports: WAV 24-bit + MP3 320, stems for a DAW
"""
import os
import shutil
import subprocess

import numpy as np

from .. import dsp
from ..util import log
from .fx import read, segs_to_ctrl


def write24(path, y, sr=dsp.SR):
    import soundfile as sf
    sf.write(path, np.clip(y, -1, 1).astype(np.float32), sr, subtype="PCM_24")


def writef(path, y, sr=dsp.SR):
    import soundfile as sf
    sf.write(path, y.astype(np.float32), sr, subtype="FLOAT")


def to_mp3(wav, mp3):
    r = subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", wav, "-c:a", "libmp3lame", "-b:a", "320k", mp3],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg could not make the MP3: {r.stderr}")


def copy_unlocked(src, dst):
    """Copy; if dst is open in a player (locked on Windows), save as 'name (2).ext' etc."""
    base, ext = os.path.splitext(dst)
    for i in range(1, 50):
        target = dst if i == 1 else f"{base} ({i}){ext}"
        try:
            shutil.copyfile(src, target)
            if i > 1:
                log(f"{os.path.basename(dst)} is open in another program — saved as {os.path.basename(target)}")
            return target
        except PermissionError:
            continue
    raise PermissionError(f"Could not save {dst}: the file is in use")


def pitch_shift(y, semitones):
    """Whole-song key change (mix.pitch_change_all), per channel (librosa: phase vocoder + resampling)."""
    import librosa
    return np.stack([librosa.effects.pitch_shift(y[:, c].astype(np.float32), sr=dsp.SR, n_steps=float(semitones))
                     for c in range(y.shape[1])], 1).astype(np.float32)


def run(cfg, fxo, sep, stage, song_dir, title, voice_model, original, clean_info=None):
    m = cfg["mix"]
    clean_info = clean_info or {}
    orig = read(original)
    n = len(orig)
    inst = dsp.fit_len(read(sep["instrumental"]), n)
    lead_dry = dsp.fit_len(read(fxo["lead_dry"]), n)
    lead_rev = dsp.fit_len(read(fxo["lead_reverb"]), n)
    nctrl = int(n / dsp.SR * dsp.CTRL) + 1
    lead_ref = dsp.fit_len(read(sep["lead_dry"]), n)
    fx_on = bool(m.get("section_fx", True))
    if fx_on:
        # passages where the original changes its vocal's tone for a while (radio, muffled): the same on yours
        from .. import sections
        sec = sections.against_cover(sections.find(dsp.fit_len(read(sep["lead_wet"]), n), dsp.SR),
                                     lead_dry + lead_rev, dsp.SR)
        if sec:
            lead_dry, lead_rev = sections.apply(lead_dry, dsp.SR, sec), sections.apply(lead_rev, dsp.SR, sec)
            log("Lead: tone effect like the original's — " + sections.describe(sec))
    lower = dsp.own_low_band(clean_info.get("lead_f0"))
    low_cut = bool(m.get("low_cut_octave_down", False))
    if lower and low_cut:
        # you sing lower than the original: the low end (60–250 Hz) toward the original's balance, at most -6 dB
        # (HOT: 120–250 Hz +11 dB over the original's) — your body above 250 Hz stays
        lead_dry, cut = dsp.match_low(lead_ref, lead_dry, max_db=6.0)
        if cut > 0:
            lead_rev = dsp.low_shelf(lead_rev, 200.0, -cut)
            log(f"Lead: low end (below 200 Hz) {cut:.1f} dB softer — you sing lower than the original, boom removed")

    # your vocal follows the original's phrase loudness (dry vs dry, so the reverb slider never fights it)
    g, base = dsp.level_match_gain(lead_ref, lead_dry,
                                   exclude=segs_to_ctrl(clean_info.get("lead_vocalizations"), nctrl))
    reverb = float(m.get("reverb", 1.0))
    vgain = 10 ** (float(m.get("vocal_gain_db", 0.0)) / 20)
    lead = (lead_dry + reverb * lead_rev) * g[:, None] * vgain
    log(f"Lead: phrase loudness like the original's (median {base:+.1f} dB), "
        f"reverb ×{reverb:g}, vocal level {m.get('vocal_gain_db', 0):+g} dB")

    backing = np.zeros_like(inst)
    if fxo.get("backing"):
        bk = dsp.fit_len(read(fxo["backing"]), n)
        bk_ref = dsp.fit_len(read(sep["backing"]), n)
        if fx_on:
            from .. import sections
            sec_b = sections.against_cover(sections.find(bk_ref, dsp.SR), bk, dsp.SR)
            if sec_b:
                bk = sections.apply(bk, dsp.SR, sec_b)
                log("Backing: tone effect like the original's — " + sections.describe(sec_b))
        if low_cut:
            # the backing needs no body of its own: its low end may go down to the original's, up to -10 dB
            bk, cut_b = dsp.match_low(bk_ref, bk, max_db=10.0)
            if cut_b > 0:
                log(f"Backing: low end (below 200 Hz) {cut_b:.1f} dB softer — like the original, no boom")
        gb, bb = dsp.level_match_gain(bk_ref, bk,
                                      exclude=segs_to_ctrl(clean_info.get("backing_vocalizations"), nctrl))
        bk = bk * gb[:, None]
        log(f"Backing: loudness from the original (median {bb:+.1f} dB)")
        # the phrase-level matching (1 s smoothing) lets single loud notes jump out over the original's (Animal:
        # up to +5 dB, peaks over 0 dBFS): those are capped to the original's level, the rest stays as matched
        capi = {"pct": 0.0}
        if m.get("backing_cap", True):
            bk, capi = dsp.cap_to_reference(bk, bk_ref)
        if capi["pct"] > 0:
            log(f"Backing: loud notes jumping out over the original turned down ({capi['pct']:.1f}% of the time, "
                f"up to −{capi['max_db']:.1f} dB)")
        backing = bk * vgain
    elif os.path.exists(sep.get("backing", "")):
        backing = dsp.fit_len(read(sep["backing"]), n) * vgain  # not converted: keep the original backing

    # the vocal bus as a whole vs the original's vocals (measured at neutral slider positions)
    if os.path.exists(sep.get("vocals", "")):
        lead0 = (lead_dry + lead_rev) * g[:, None]
        corr = dsp.match_vocal_bus(read(sep["vocals"]), lead0 + backing / vgain,
                                   keep=lower)
        lead, backing = dsp.apply_vocal_bus(lead, corr), dsp.apply_vocal_bus(backing, corr)
        tone = " ".join(f"{v:+.1f}" for v in corr["curve"])
        log(f"Whole vocal against the original's: tone {tone} dB (63 Hz…16 kHz), level {corr['level_db']:+.1f} dB, "
            f"width ×{corr['side_gain']:.2f}")
    if lower and not low_cut:
        # (mix.low_cut_octave_down: false — the earlier, milder way) you sing lower than the original: your
        # fundamentals stay, only the very bottom (< 100 Hz) is eased (Bite Me: +23 dB at 60–120 Hz)
        lead = dsp.low_shelf(lead, 100.0, -3.5)
        log("Lead: the very bottom (below 100 Hz) 3.5 dB softer — you sing lower than the original")

    pall = int(m.get("pitch_change_all", 0))
    if pall:
        log(f"Whole-song key change: {pall:+d}")
        inst = pitch_shift(inst, pall)
        if not fxo.get("backing"):
            backing = pitch_shift(backing, pall)

    mix = inst + lead + backing
    c_orig, c_mix = dsp.crest_db(orig), dsp.crest_db(mix)
    if c_mix > c_orig + 0.5:
        mix, glue = dsp.bus_glue(mix, target_gr_db=min(3.0, c_mix - c_orig))
        log(f"Mix glue like the original's: peak/average {c_mix:.1f} → {dsp.crest_db(mix):.1f} dB "
            f"(the original {c_orig:.1f}), −{glue['gr_loud_db']:.1f} dB on loud parts")
    # loudness target: the original as it would be with a legal ceiling (many releases peak above 0 dBFS),
    # but never squash the mix by more than MAX_LOSS dB of loudness (separated stems are peakier than the master)
    MAX_LOSS = 3.0
    orig_lim, _ = dsp.limiter(orig, -1.0)
    target = dsp.loudness_lufs(orig_lim)
    l_mix = dsp.loudness_lufs(mix)

    def master(gdb):
        y, red = dsp.limiter(dsp.soft_clip(mix * 10 ** (gdb / 20), 0.0), -1.0)
        return y, red, dsp.loudness_lufs(y)

    lo, hi = -24.0, target - l_mix + MAX_LOSS
    best = None
    for _ in range(7):  # bisection on the master gain
        mid = (lo + hi) / 2
        y, red, lu = master(mid)
        if lu <= target + 0.1 and (l_mix + mid) - lu <= MAX_LOSS:
            best, lo = (mid, y, red, lu), mid
        else:
            hi = mid
    gain_db, out, gr, final_lufs = best if best else (lo, *master(lo))
    note = "" if final_lufs > target - 0.5 else f" (the original {target:.1f}; louder only by over-squashing)"
    log(f"Mastering: {final_lufs:.1f} LUFS{note}, peaks at −1 dB, limiter up to {gr:.1f} dB")

    near = dsp.closeness(orig_lim, out)
    bands = ", ".join(f"{int(c) if c < 1000 else f'{c / 1000:g}k'} {v:+.1f}" for c, v in zip(dsp.OCT, near["tone_db"])
                      if abs(v) >= 1.5 and c > 63)
    log(f"Against the original: loudness {near['lufs_diff']:+.1f} LU, tone ±{near['tone_max_db']:.1f} dB"
        + (f" ({bands})" if bands else "") + f", density {near['crest_diff']:+.1f} dB, width {near['width_diff_db']:+.1f} dB")

    # exports
    wav = stage.path("cover.wav")
    write24(wav, out)
    mp3 = stage.path("cover.mp3")
    to_mp3(wav, mp3)
    stems = stage.path("stems")
    os.makedirs(stems, exist_ok=True)
    for name, y in (("vocal", lead), ("backing", backing), ("instrumental", inst)):  # for a DAW: unmastered, float
        writef(os.path.join(stems, f"{name}.wav"), y)

    final_mp3 = copy_unlocked(mp3, os.path.join(song_dir, f"{title} ({voice_model} Ver).mp3"))
    final_wav = copy_unlocked(wav, os.path.join(song_dir, f"{title} ({voice_model} Ver).wav"))
    return {"cover_wav": wav, "cover_mp3": mp3}, {"final": final_mp3, "final_wav": final_wav, "lufs": final_lufs,
                                                  "target_lufs": target, "limiter_db": gr, "master_gain_db": gain_db,
                                                  "closeness": near}
