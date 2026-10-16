"""Stage 3: clean-up of the converted vocals.

  - vocalizations (oo, aa, oh, la-la…): Whisper finds the words; sung parts without real words are
      "soft"   (default) re-converted with your voice using gentle settings, a bit quieter — like ad-libs in a mix
      "remove" muted
  - consonants: s, sh, f, t, k and breaths above ~4 kHz come from the original (RVC draws them dull and quiet)
  - de-clicker: repairs isolated digital clicks (calibrated so that a normal voice is never touched)
  - intelligibility: Whisper compares your lead with the original phrase by phrase; phrases where words got lost
    are re-sung with careful settings (more of the original's consonants, less index) and kept only if clearer
    (only for languages Whisper hears reliably)
  - noise between the harmonics of your lead turned down (clean.harmonic_clean_db, 0 = off)
  - air above ~8 kHz from the original (clean.restore_air): RVC hears 16 kHz audio and leaves the top 15–20 dB short
  - a quality line in the log: creak on notes the original sings cleanly, air against the original
  - breaths (coversong/breaths.py, clean.breaths): where the original breathes, your breath from the vocal studio (or
    the original's) instead of the hiss RVC makes of it
"""
import json
import os
import re
import shutil

import numpy as np

from ..util import log

FPS = 100  # analysis frames per second
INTERJ = re.compile(r"^(o+h*|u+h*|a+h*|e+h*|h?m+|h?mm+|la|na|da|ba|doo+|woo+|w?hoa+|y?eah+h+|"
                    r"у+|о+|а+|э+|м+|ла|на|да|ля)$")


# ---------- helpers ----------
def read(path):
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    return y, sr


def write(path, y, sr):
    import soundfile as sf
    sf.write(path, y if y.shape[1] > 1 else y[:, 0], sr, subtype="FLOAT")


def fmt_t(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def activity(path):
    """10 ms frames where the vocal actually sounds (energy within 35 dB of its loud level)."""
    from my_utils import load_audio
    x = load_audio(path, 16000)
    hop = 16000 // FPS
    n = len(x) // hop
    e = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9
    db = 20 * np.log10(e)
    return db > np.percentile(db, 95) - 35


def segments(mask, min_len=0.5, close_gap=0.3):
    m = mask.copy()
    # close short gaps
    gap = int(close_gap * FPS)
    idx = np.flatnonzero(m)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap:
            m[a:b] = True
    out, start = [], None
    for i, v in enumerate(np.append(m, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if (i - start) / FPS >= min_len:
                out.append((start / FPS, i / FPS))
            start = None
    return out


def vocalization_segments(words, src_vocal, tag):
    """Segments of src_vocal that are sung without real words."""
    act = activity(src_vocal)
    n = len(act)
    lexical = np.zeros(n, bool)
    for w in words["words"]:
        token = re.sub(r"[^\w]", "", w["word"].lower())
        if token and not INTERJ.match(token):
            a, b = int(max(0, w["start"] - 0.1) * FPS), int(min(n / FPS, w["end"] + 0.1) * FPS)
            lexical[a:b] = True
    cover = (lexical & act).sum() / max(1, act.sum())
    if cover < 0.3:
        log(f"{tag}: few words recognised ({cover:.0%} of the singing) — wordless parts left alone, so nothing real is lost")
        return [], cover
    return segments(act & ~lexical), cover


def seg_envelope(n, sr, segs, fade=0.05):
    """1 inside segments, 0 outside, with smooth fades."""
    g = np.zeros(n, np.float32)
    for a, b in segs:
        g[int(a * sr):int(b * sr)] = 1.0
    k = max(1, int(fade * sr))
    g = np.convolve(np.pad(g, (k, k), mode="edge"), np.ones(k, np.float32) / k, mode="same")[k:-k]
    return g[:, None]


def mute(y, sr, segs):
    return y * (1 - seg_envelope(len(y), sr, segs))


def blend(y, soft, sr, segs, gain_db=-3.0):
    n = min(len(y), len(soft))
    y, soft = y[:n], soft[:n]
    e = seg_envelope(n, sr, segs)
    return y * (1 - e) + soft * e * (10 ** (gain_db / 20))


def declick(y, sr, k=25.0, min_amp=0.05):
    """Isolated impulses only: residual against a 5-sample median > k x its local RMS (normal voice stays < 18)."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import medfilt
    out = y.copy()
    total = 0
    for ch in range(y.shape[1]):
        x = y[:, ch]
        res = x - medfilt(x, 5)
        loc = np.sqrt(np.maximum(uniform_filter1d(res ** 2, int(0.02 * sr)), 0)) + 1e-5
        hits = np.flatnonzero((np.abs(res) > k * loc) & (np.abs(res) > min_amp))
        if len(hits) == 0:
            continue
        w = int(0.0005 * sr)
        for g in np.split(hits, np.flatnonzero(np.diff(hits) > int(0.001 * sr)) + 1):
            a, b = max(1, g[0] - w), min(len(x) - 2, g[-1] + w)
            out[a:b, ch] = np.linspace(out[a - 1, ch], out[b + 1, ch], b - a)
            total += 1
    return out, total


def splice(y, alt, sr, regions, fade=0.03):
    """Take `alt` inside regions (seconds), `y` elsewhere, with short crossfades."""
    n = min(len(y), len(alt))
    g = np.zeros(n, np.float32)
    for a, b in regions:
        g[max(0, int(a * sr)):min(n, int(b * sr))] = 1.0
    k = max(1, int(fade * sr))
    g = np.convolve(np.pad(g, (k, k), mode="edge"), np.ones(k, np.float32) / k, mode="same")[k:-k][:, None]
    out = y.copy()
    out[:n] = y[:n] * (1 - g) + alt[:n] * g
    return out


# ---------- stage ----------
SOFT = {"index_rate": 0.25, "protect": 0.2, "rms_mix_rate": 0.0}  # gentle settings for vocalizations


def careful_params(p):
    """More of the original's consonants (protect up), less pull towards your recordings (index down)."""
    return dict(p, protect=max(0.5, float(p["protect"])), index_rate=round(float(p["index_rate"]) / 2, 2))


def repair_phrases(cfg, sess, stage, y, sr, src, words, cons, semitones, p, language):
    """Whisper on your lead and on a careful re-conversion; per phrase keep the version that kept more words."""
    from .. import intelligibility as it
    from .convert import transplant_file
    lex = it.lexical(words["words"])
    if len(lex) < 10:
        log("Intelligibility: too few words in the song to check")
        return y, None
    cur = stage.path("_lead_check.wav")
    write(cur, y, sr)
    alt = stage.path("_lead_careful.wav")
    sess.convert(src, alt, semitones, careful_params(p))
    transplant_file(alt, src, cons)
    ya, _ = read(alt)
    ya, _ = declick(ya, sr)
    write(alt, ya, sr)
    res = it.asr(cfg, [(cur, stage.path("_words_cur.json"), language), (alt, stage.path("_words_alt.json"), language)])
    rows_a = it.per_phrase(words["words"], res[0]["words"])
    rows_b = it.per_phrase(words["words"], res[1]["words"])
    regions, rows, fixed = [], [], []
    for ra, rb in zip(rows_a, rows_b):
        gain_words = (rb[2] - ra[2]) * ra[3]
        if ra[3] >= 2 and gain_words >= 1 and rb[2] >= ra[2] + 0.2:
            regions.append((ra[0] - 0.15, ra[1] + 0.15))
            fixed.append((ra[0], ra[2], rb[2], ra[4]))
            rows.append(rb)
        else:
            rows.append(ra)
    if regions:
        y = splice(y, ya, sr, regions)
    for f in (cur, alt, stage.path("_words_cur.json"), stage.path("_words_alt.json")):
        os.remove(f)
    before, n = it.summary(rows_a)
    after, _ = it.summary(rows)
    if fixed:
        where = ", ".join(f"{it.fmt_t(t)} ({a:.0%}→{b:.0%})" for t, a, b, _ in fixed[:8]) + (" …" if len(fixed) > 8 else "")
        log(f"Intelligibility: phrases re-sung more carefully: {len(fixed)} — {where}")
    log(f"Lead intelligibility: {after:.0%} of the original's words ({n} words)"
        + (f", was {before:.0%}" if fixed else "") + f"; weakest places: {it.worst(rows)}")
    return y, {"before": before, "after": after, "words": n, "fixed": [(t, a, b) for t, a, b, _ in fixed],
               "worst": it.worst(rows)}


def polish(cfg, y, sr, src_vocal, tag, semitones, multi):
    """Harmonic clean-up (lead only) and air from the original; logs the quality of the result."""
    from .. import quality
    from ..dsp import to_sr
    from ..phonetics import f0_track
    c = cfg.get("clean", {})
    name = "lead" if tag == "lead" else "backing"
    s, ssr = read(src_vocal)
    s = to_sr(s.mean(1), ssr, sr).astype(np.float32)
    s = s[:len(y)] if len(s) >= len(y) else np.pad(s, (0, len(y) - len(s)))
    st = {}
    depth = float(c.get("harmonic_clean_db", 6) or 0)
    if tag == "lead":
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            write(tmp, y, sr)
            f0 = f0_track(cfg, tmp)
        finally:
            os.remove(tmp)
        fs = f0_track(cfg, src_vocal)
        if (f0 > 50).sum() > 100 and (fs > 50).sum() > 100:  # where your voice sits against the original (stage 4 EQ)
            st["f0"] = {"cov_p10": float(np.percentile(f0[f0 > 50], 10)), "cov_med": float(np.median(f0[f0 > 50])),
                        "src_p10": float(np.percentile(fs[fs > 50], 10)), "src_med": float(np.median(fs[fs > 50]))}
        if depth > 0:
            y = quality.harmonic_clean(y, sr, f0, depth_db=depth)
            st["harmonic_db"] = depth
    air_before = quality.air_db(y.mean(1), sr)
    if c.get("restore_air", True):
        cutoff = 7500.0 if (not multi and abs(int(semitones)) < 6) else 9000.0  # shifted voice: only the breath band
        y, lift = quality.restore_air(y, sr, s, cutoff=cutoff)
        st["air_lift_db"] = lift
    st["air_db"], st["air_before_db"], st["air_orig_db"] = quality.air_db(y.mean(1), sr), air_before, quality.air_db(s, sr)
    q = quality.creak(s, y.mean(1), sr)
    st.update(q)
    parts = []
    if q["creak"] is not None:
        parts.append(f"creak on the original's clean notes {q['creak']:.1f}%"
                     + (f" ({q['edges']:.0f}% at note edges)" if q["edges"] is not None else ""))
    parts.append(f"air above 8 kHz: original {st['air_orig_db']:.0f} dB, yours {air_before:.0f}"
                 + (f" → {st['air_db']:.0f} dB" if c.get("restore_air", True) else " dB"))
    if "harmonic_db" in st:
        parts.append(f"noise between the harmonics turned down up to {depth:.0f} dB")
    log(f"{name}: " + "; ".join(parts))
    return y, st


def run(cfg, conv, sep, stage, voice_model=None, semitones=0, params=None, language=None, voice_plan=None):
    c = cfg.get("clean", {})
    mode = str(c.get("vocalizations", "soft")).lower()
    p = dict(cfg["convert"], **(params or {}))
    outputs, info = {}, {"vocalizations_mode": mode, "cons_lead": conv.get("cons_lead")}
    tracks = [("lead", sep["lead_dry"])]
    if conv.get("backing"):
        from .separate import backing_src
        tracks.append(("backing", backing_src(sep)))
    words, cons = {}, {}
    for tag, _ in tracks:
        with open(conv[f"words_{tag}"], encoding="utf-8") as f:
            words[tag] = json.load(f)
        cons[tag] = np.load(conv[f"cons_{tag}"])
    from .convert import close_sessions, make_sessions, transplant_file
    sess_lead, sess_bk = make_sessions(cfg, voice_model, sep, voice_plan)
    try:
        for tag, src_vocal in tracks:
            sess = sess_lead if tag == "lead" else sess_bk
            name = "lead" if tag == "lead" else "backing"
            segs = []
            if mode in ("soft", "remove"):
                segs, _ = vocalization_segments(words[tag], src_vocal, tag)
            # consonants of the original first: the vocalization blend below has its own (soft) conversion
            work = stage.path(f"_{tag}.wav")
            shutil.copyfile(conv[tag], work)
            st = transplant_file(work, src_vocal, cons[tag])
            y, sr = read(work)
            os.remove(work)
            if st and st["frames_pct"] > 0:
                log(f"{name}: consonants from the original ({st['frames_pct']:.0f}% of the time; their highs were "
                    f"{st['hf_before_db']:+.0f} dB against the original)")
            if segs:
                where = ", ".join(f"{fmt_t(a)}–{fmt_t(b)}" for a, b in segs[:12]) + (" …" if len(segs) > 12 else "")
                total = sum(b - a for a, b in segs)
                if mode == "remove":
                    y = mute(y, sr, segs)
                    log(f"{name}: wordless parts removed ({len(segs)}, {total:.1f} s): {where}")
                else:
                    from .convert import backing_params
                    sp = dict(backing_params(p) if tag == "backing" else p, **SOFT)
                    soft = stage.path(f"_{tag}_soft.wav")
                    if y.shape[1] > 1:
                        sess.convert_stereo(src_vocal, soft, semitones, sp, stage.path(f"_{tag}_s"))
                    else:
                        sess.convert(src_vocal, soft, semitones, sp)
                    ys, _ = read(soft)
                    if ys.shape[1] != y.shape[1]:
                        ys = np.repeat(ys[:, :1], y.shape[1], axis=1)
                    y = blend(y, ys, sr, segs)
                    os.remove(soft)
                    log(f"{name}: wordless parts sung softer ({len(segs)}, {total:.1f} s): {where}")
                info[f"{tag}_vocalizations"] = segs
            if tag == "backing":
                if voice_plan:
                    if str(c.get("choir", "auto")).lower() != "off":
                        log("Backing choir: not added with several voices")
                else:
                    from .. import choir
                    from .convert import backing_params
                    wl = segs if mode == "soft" else ([] if mode == "remove" else
                                                       vocalization_segments(words[tag], src_vocal, tag)[0])
                    y, info["choir"] = choir.add(cfg, sess, src_vocal, y, sr, semitones, backing_params(p), stage,
                                                 wl, soft=SOFT)
            if c.get("declick", True):
                y, n = declick(y, sr)
                if n:
                    log(f"{name}: clicks repaired: {n}")
            lang = language or words[tag].get("language")
            if tag == "lead":
                from .convert import words_reliable
                if words_reliable(lang):
                    y, info["intelligibility"] = repair_phrases(cfg, sess, stage, y, sr, src_vocal, words[tag],
                                                                cons[tag], semitones, p, lang)
                else:
                    log(f"Intelligibility: Whisper does not hear the song's language ({lang or 'unknown'}) reliably — "
                        f"no phrases re-sung (words cannot be the judge here)")
            y, info[f"{tag}_quality"] = polish(cfg, y, sr, src_vocal, tag, semitones, bool(voice_plan))
            if info[f"{tag}_quality"].get("f0"):
                info["lead_f0"] = info[f"{tag}_quality"].pop("f0")
            from .. import breaths
            y, binfo = breaths.apply(cfg, y, sr, src_vocal, words[tag].get("words"), name, voice_model)
            if binfo:
                info[f"{tag}_breaths"] = binfo
            outputs[tag] = stage.path(f"{tag}.wav")
            write(outputs[tag], y, sr)
    finally:
        close_sessions(sess_lead, sess_bk)
    return outputs, info
