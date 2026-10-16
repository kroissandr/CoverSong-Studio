"""Stage 2: RVC voice conversion of the dry lead vocal and the backing vocals.

  1. the original vocal is analysed once: its words (Whisper) and its consonant moments (RMVPE + spectrum)
  2. auto-tuning on 30 s of the song: how the melody is read (the least creak at the same notes) and, where Whisper
     understands the language, index/protect (the most words of the original kept) — only if your timbre does not
     suffer (otherwise the defaults stay)
  3. the whole lead and backing vocals are converted with the chosen settings
"""
import gc
import glob
import json
import os

import numpy as np

from .. import ROOT
from ..util import log


def find_model(cfg, voice_model):
    model_dir = os.path.join(cfg["paths"]["rvc_models"], voice_model)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"No model folder: {model_dir}")
    pth = idx = None
    for fn in sorted(os.listdir(model_dir)):
        ext = os.path.splitext(fn)[1].lower()
        if ext == ".pth":
            pth = os.path.join(model_dir, fn)
        elif ext == ".index":
            idx = os.path.join(model_dir, fn)
    if not pth:
        raise FileNotFoundError(f"No .pth file in {model_dir}")
    return pth, idx or ""


def model_signature(cfg, voice_model):
    pth, idx = find_model(cfg, voice_model)
    sig = {"pth": os.path.basename(pth), "pth_size": os.path.getsize(pth), "pth_mtime": int(os.path.getmtime(pth))}
    if idx:
        sig["index"] = os.path.basename(idx)
        sig["index_size"] = os.path.getsize(idx)
        sig["index_mtime"] = int(os.path.getmtime(idx))
    return sig


def resolve_octaves(cfg, voice_model, lead_dry):
    """convert.pitch_octaves: a number or "auto" (from the song vocal's middle note and your range)."""
    v = cfg["convert"].get("pitch_octaves", "auto")
    if str(v).lower() != "auto":
        return int(v), None
    from .. import pitch
    if not os.path.exists(os.path.join(cfg["paths"]["rvc_models"], voice_model, "voice_profile.json")) \
            and not pitch.recordings(cfg, voice_model):
        log(f"Key: the voice range of model {voice_model} is unknown — no shift "
            f"(choose a shift by hand if needed)")
        return 0, None
    prof = pitch.voice_profile(cfg, voice_model)
    midi = pitch.f0_midi(cfg, lead_dry)
    src = pitch.stats(midi)
    if not src:
        log("Vocal too short to analyse the key — shift 0")
        return 0, None
    k, cov, cov_mid = pitch.choose_octaves(midi, prof)
    why = ""
    if cov is not None:
        why = f"; {cov:.0%} of the notes fall in your range"
        k_mid = pitch.suggest_octaves(src["median"], prof["median"])
        if k_mid != k:
            why += f" (shifting by the middle note, {k_mid:+d}, would give {cov_mid:.0%})"
    log(f"Key: the song's vocal is around {src['median_note']} ({src['range']}), your voice around {prof['median_note']} "
        f"({prof['range']}) → shift {k:+d} oct.{why}")
    return k, {"song": src, "voice": prof, "coverage": cov}


class RVCSession:
    """Loads hubert + voice model once; converts several files."""

    def __init__(self, cfg, voice_model, pth=None, index=None, hubert=None):
        import torch  # noqa: F401
        from rvc import Config, get_vc, load_hubert

        self.cfg = cfg
        m_pth, m_idx = find_model(cfg, voice_model) if voice_model else (None, "")
        pth = pth or m_pth
        self.index = m_idx if index is None else index
        import torch
        self.config = Config("cuda:0" if torch.cuda.is_available() else "cpu", torch.cuda.is_available())
        device = self.config.device
        from .. import assets
        self.hubert = hubert or load_hubert(device, self.config.is_half, assets.contentvec_dir(cfg))
        self.cpt, self.version, self.net_g, self.tgt_sr, self.vc = get_vc(device, self.config.is_half, self.config, pth)
        import vc_infer_pipeline as vip
        vip.INDEX_NPROBE = int(cfg["convert"].get("index_nprobe", 16))
        vip.LOG = log
        vip.RMVPE_PATH = assets.ensure(cfg, "rmvpe.pt")
        vip._REPORTED[0] = False
        log(f"Model {voice_model or os.path.splitext(os.path.basename(pth))[0]}: {os.path.basename(pth)}, {self.version}, {self.tgt_sr} Hz, index: {os.path.basename(self.index) or 'none'}")
        self.range = None  # comfortable notes of this voice (MIDI): phrases below are sung an octave higher
        prof = os.path.join(cfg["paths"]["rvc_models"], voice_model or "", "voice_profile.json")
        if voice_model and os.path.exists(prof):
            with open(prof, encoding="utf-8") as f:
                pr = json.load(f)
            if pr.get("p10") is not None and pr.get("p90") is not None:
                self.range = (float(pr["p10"]) - 3, float(pr["p90"]) + 3)

    def convert(self, src, dst, pitch, p):
        from rvc import rvc_infer
        from .. import f0fix
        f0fix.set_range(self.range)
        f0fix.set_enabled(self.cfg["convert"].get("fix_low_phrases", True))
        f0fix.set_smooth(p.get("smooth_edges", True))
        f0fix.set_upper(p.get("prefer_upper", False), p.get("backing_fill_voicing", True))
        rvc_infer(self.index, p["index_rate"], src, dst, pitch, p["f0_method"], self.cpt, self.version, self.net_g,
                  p["filter_radius"], self.tgt_sr, p["rms_mix_rate"], p["protect"], p["crepe_hop_length"],
                  self.vc, self.hubert)

    def convert_stereo(self, src, dst, pitch, p, tmp_prefix):
        """Convert each channel separately (keeps the width of doubled backing vocals)."""
        import numpy as np
        import soundfile as sf
        y, sr = sf.read(src, dtype="float32", always_2d=True)
        if y.shape[1] == 1 or np.corrcoef(y[:, 0], y[:, 1])[0, 1] > 0.98:
            self.convert(src, dst, pitch, p)
            return "mono"
        outs = convert_channels(self, self.cfg, y, sr, pitch, p, tmp_prefix)
        n = min(len(o) for o in outs)
        sf.write(dst, np.stack([o[:n] for o in outs], axis=1), self.tgt_sr)
        return "stereo"

    def close(self):
        import torch
        del self.hubert, self.cpt, self.net_g, self.vc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Voices:
    """Several voice models over time ("who sings when" from the multi-voice tab), same interface as RVCSession.
    The source is rendered once per (model, shift) and stitched by time with 30 ms crossfades.
    Line labels: "0" one singer, "0+2" singers together (their renders mixed, slightly offset like real
    double-tracking), "mute" silence. A model can be "orig" (keep the original singer) or "mute".
    The backing vocal, by default ("voices"), follows the lead's plan: each voice sings its own backing, with the
    same model and octave shift as its lead lines (pass `like` = the lead's Voices: models are shared).
    With backing = "orig" or one model name, that sings all of it."""

    multi = True
    OFFSETS = (0.0, 0.017, 0.026, 0.009, 0.031, 0.013, 0.022, 0.004)  # s, for parts sung together

    def __init__(self, cfg, plan, track, src_path, like=None):
        self.cfg = cfg
        self.sessions = like.sessions if like is not None else {}
        self.octaves = {str(k): v for k, v in (plan.get("octaves") or {}).items()}
        self.track = track
        bk = plan.get("backing") or "voices"
        if track == "backing" and bk != "voices":
            self.segments = None
            self.assign = {"backing": bk}
        else:
            self.segments = [(float(s), float(e), str(v)) for s, e, v in plan["segments"]]
            self.assign = {str(k): v for k, v in plan["assign"].items() if v}
        if track == "backing" and bk == "voices" and like is not None:
            self.semis = dict(like.semis)  # the same singer keeps the same shift in the backing
            log("  backing: each voice its own — whoever sings the main line also sings the backing under it")
            return
        self.semis = self._semitones(src_path)
        for who, model in self.assign.items():
            if who not in self.semis:
                continue
            log(f"  {self._name(who)}: " + ({"orig": "original", "mute": "muted"}.get(model) or
                                           f"{model}, shift {self.semis[who]:+d}"))

    @staticmethod
    def _name(who):
        if who == "backing":
            return "backing"
        return f"voice {int(who) + 1}"

    def _parts(self, lab):
        """Who sings a line with this label."""
        if lab in ("mute", "chorus"):
            return []
        if self.segments is None:
            return ["backing"]
        return [v for v in lab.split("+") if self.assign.get(v) not in (None, "mute")]

    def _labels(self):
        return sorted({l for _, _, l in self.segments}) if self.segments is not None else ["all"]

    def _semitones(self, src_path):
        from .. import pitch as P
        from ..phonetics import f0_track
        glob = self.cfg["convert"].get("pitch_octaves", "auto")
        extra = int(self.cfg["mix"].get("pitch_change_all", 0))
        out = {}
        f0 = None
        for who, model in self.assign.items():
            labs = [l for l in self._labels() if who in self._parts(l)]
            if not labs:
                continue
            if model in ("orig", "mute"):
                out[who] = 0
                continue
            v = self.octaves.get(who, glob)
            if str(v).lower() != "auto":
                out[who] = int(v) * 12 + extra
                continue
            prof_path = os.path.join(self.cfg["paths"]["rvc_models"], model, "voice_profile.json")
            if not os.path.exists(prof_path):
                out[who] = extra
                continue
            if f0 is None:
                f0 = f0_track(self.cfg, src_path)
            mask = np.zeros(len(f0), bool)
            for l in labs:
                mask |= self._frames(len(f0), l)
            seg = f0[mask & (f0 > 40)]
            if len(seg) < 50:
                out[who] = extra
                continue
            with open(prof_path, encoding="utf-8") as fh:
                prof = json.load(fh)
            k, cov, _ = P.choose_octaves(69 + 12 * np.log2(seg / 440), prof)
            out[who] = 12 * k + extra
        return out

    def _frames(self, n, lab):
        """Bool per 10 ms: frames belonging to label `lab` (pauses go to the nearest line)."""
        if self.segments is None:
            return np.ones(n, bool)
        owner = np.full(n, -1)
        labs = [l for _, _, l in self.segments]
        for i, (s, e, _) in enumerate(self.segments):
            owner[int(s * 100):int(e * 100)] = i
        known = np.flatnonzero(owner >= 0)
        if len(known):  # pauses: the closer of the previous and the next line
            t = np.arange(n)
            nxt = np.clip(np.searchsorted(known, t), 0, len(known) - 1)
            prv = np.clip(nxt - 1, 0, len(known) - 1)
            pick = np.where(np.abs(known[prv] - t) <= np.abs(known[nxt] - t), known[prv], known[nxt])
            owner = np.where(owner >= 0, owner, owner[pick])
        return np.array([labs[o] == lab if o >= 0 else False for o in owner])

    def _session(self, model):
        if model not in self.sessions:
            shared = next(iter(self.sessions.values()), None)  # one HuBERT for all models
            self.sessions[model] = RVCSession(self.cfg, model, hubert=shared.hubert if shared else None)
        return self.sessions[model]

    def convert(self, src, dst, pitch, p):
        import soundfile as sf
        from ..dsp import to_sr
        sr = 48000
        x, ssr = sf.read(src, dtype="float32", always_2d=True)
        x = to_sr(x.mean(1), ssr, sr)
        n = len(x)
        renders = {}

        def render(who):
            model = self.assign[who]
            key = (model, self.semis[who])
            if key not in renders:
                if model == "orig":
                    renders[key] = x
                else:
                    tmp = dst + f".{len(renders)}.wav"
                    self._session(model).convert(src, tmp, self.semis[who], p)
                    y, ysr = sf.read(tmp, dtype="float32", always_2d=True)
                    os.remove(tmp)
                    renders[key] = fit(to_sr(y.mean(1), ysr, sr), n)
            return renders[key]

        out = np.zeros(n, np.float32)
        weight = np.zeros(n, np.float32)
        for lab in self._labels():
            m = self._frames(int(np.ceil(n / sr * 100)) + 1, lab).astype(np.float32)
            if not m.any():
                continue
            k = 3  # 30 ms crossfade
            m = np.convolve(m, np.ones(k) / k, mode="same")
            w = np.interp(np.arange(n) / sr * 100, np.arange(len(m)), m).astype(np.float32)
            weight += w
            parts = self._parts(lab)
            if not parts:
                continue
            g = 1.0 / np.sqrt(len(parts))
            for j, who in enumerate(parts):
                y = render(who)
                d = int(self.OFFSETS[j % len(self.OFFSETS)] * sr) if len(parts) > 1 else 0
                if d:
                    y = np.concatenate([np.zeros(d, np.float32), y[:-d]])
                out += y * w * g
        sf.write(dst, out / np.maximum(weight, 1e-6), sr, subtype="FLOAT")

    def convert_stereo(self, src, dst, pitch, p, tmp_prefix):
        import soundfile as sf
        y, sr = sf.read(src, dtype="float32", always_2d=True)
        if y.shape[1] == 1 or np.corrcoef(y[:, 0], y[:, 1])[0, 1] > 0.98:
            self.convert(src, dst, pitch, p)
            return "mono"
        outs = convert_channels(self, self.cfg, y, sr, pitch, p, tmp_prefix)
        n = min(len(o) for o in outs)
        sf.write(dst, np.stack([o[:n] for o in outs], axis=1), 48000)
        return "stereo"

    def close(self):
        for s in self.sessions.values():
            s.close()
        self.sessions.clear()  # shared with the backing's Voices: emptied in place, closed only once


def backing_params(p):
    """Settings for the backing vocal: the melody is read with RMVPE (convert.backing_f0_method) — the method the
    auto-tuning picks is chosen on the lead, and a dry solo voice is not a stack of harmonies: in HOT, HPA-RMVPE
    won on the lead but doubled the creak in the backing (6.8 % -> 14.7 %, "metallic"). Octave doubles follow
    the upper voice (convert.backing_upper_voice)."""
    m = str(p.get("backing_f0_method", "rmvpe"))
    out = dict(p, prefer_upper=bool(p.get("backing_upper_voice", True)))
    if m != "same":
        out["f0_method"] = m
    out.update(p.get("backing") or {})  # chosen for the backing by tune_backing
    out.pop("backing", None)
    return out


def tune_backing(cfg, sess, voice_model, bk_src, semis, stage, p):
    """The backing's own auto-tuning (the lead's settings are chosen on a dry solo voice): on the 30 s where the
    backing sounds most (its louder channel), melody readers and index/protect are tried; the least creak wins if
    it is clearly better, stays on the notes and keeps your timbre. Returns (settings, table)."""
    import soundfile as sf
    from .. import quality
    from ..dsp import to_sr
    y, sr = sf.read(bk_src, dtype="float32", always_2d=True)
    if len(y) / sr < 8:
        return {}, None
    s0 = loud_window(y, sr)
    a, b = int(s0 * sr), int((s0 + 30) * sr)
    ch = int(np.argmax((y[a:b] ** 2).mean(0)))
    xs = y[a:b, ch]
    src = stage.path("_tune_bk_src.wav")
    sf.write(src, xs, sr, subtype="FLOAT")
    voice = voice_timbre(cfg, voice_model)
    base = backing_params(dict(p, backing=None))
    keys = ("f0_method", "smooth_edges", "index_rate", "protect")
    ir, pr = float(base["index_rate"]), float(base["protect"])
    tries = [dict(f0_method=m, smooth_edges=sm, index_rate=ir, protect=pr)
             for m, sm in ((base["f0_method"], base.get("smooth_edges", True)), ("rmvpe", True), ("rmvpe", False),
                           ("hpa-rmvpe", True), ("swiftf0", True))]
    seen, rows = set(), []

    def run_one(t):
        key = tuple(t[k] for k in keys)
        if key in seen:
            return None
        seen.add(key)
        dst = stage.path(f"_tune_bk_{len(seen)}.wav")
        try:
            sess.convert(src, dst, semis, dict(base, **t))
        except Exception as e:
            log(f"  backing, {f0_label(t['f0_method'], t['smooth_edges'])}: failed ({e})")
            return None
        c, csr = sf.read(dst, dtype="float32", always_2d=True)
        q = quality.creak(xs, to_sr(c.mean(1), csr, sr)[:len(xs)], sr)
        r = dict(t, creak=q["creak"], off_notes=off_pitch_class(src, dst),
                 timbre=float(np.linalg.norm(mfcc_mean(dst) - voice)) if voice is not None else None)
        os.remove(dst)
        rows.append(r)
        return r

    for t in tries:
        run_one(t)

    def better(r, ref):
        if r["creak"] is None or ref["creak"] is None:
            return False
        clearer = r["creak"] <= ref["creak"] * 0.85 and ref["creak"] - r["creak"] >= 0.5
        on = r["off_notes"] is None or ref["off_notes"] is None or r["off_notes"] <= ref["off_notes"] + 1.5
        tb = r["timbre"] is None or ref["timbre"] is None or r["timbre"] <= ref["timbre"] * 1.08
        return clearer and on and tb
    if not rows:
        os.remove(src)
        return {}, None
    first = rows[0]
    best = min([r for r in rows if r is first or better(r, first)], key=lambda r: r["creak"] if r["creak"] is not None else 1e9)
    # index/protect on the chosen melody reader
    for i2, p2 in ((round(ir / 2, 2), max(pr, 0.5)), (ir, max(pr, 0.5)), (round(ir / 2, 2), pr)):
        run_one(dict(best, index_rate=i2, protect=p2))
    best = min([r for r in rows if r is best or better(r, best)], key=lambda r: r["creak"] if r["creak"] is not None else 1e9)
    os.remove(src)
    log(f"Tuning the backing ({int(s0 // 60)}:{int(s0 % 60):02d}–, {'left' if ch == 0 else 'right'} channel):")
    for r in rows:
        cr = f"creak {r['creak']:.1f}%" if r["creak"] is not None else "creak —"
        on = f", wrong notes {r['off_notes']:.1f}%" if r["off_notes"] is not None else ""
        tb = f", timbre {r['timbre']:.1f}" if r["timbre"] is not None else ""
        log(f"  {f0_label(r['f0_method'], r['smooth_edges'])}, index {r['index_rate']}, protect {r['protect']}: "
            f"{cr}{on}{tb}{' ← chosen' if r is best else ''}")
    return {k: best[k] for k in keys}, rows


def off_pitch_class(src, dst):
    """% of frames where the cover's note is not the original's note in any octave (SwiftF0 on both): the backing's
    octave is moved on purpose (upper voice, range guards), its pitch class is not."""
    from my_utils import load_audio
    from ..f0_extra import swift
    a, _ = swift(load_audio(src, 16000))
    b, _ = swift(load_audio(dst, 16000))
    n = min(len(a), len(b))
    v = (a[:n] > 0) & (b[:n] > 0)
    if v.sum() < 100:
        return None
    d = 12 * np.log2(b[:n][v] / a[:n][v])
    d = d - 12 * np.round(d / 12)
    return float((np.abs(d) > 1).mean() * 100)


def channel_reference(cfg, paths):
    """Backing in stereo: both channels usually carry the same part. Where RMVPE reads one channel an octave
    lower than the other, the upper reading is the reference for both (else the two sides of your backing sing an
    octave apart — Standing Next to You, 1:09)."""
    from ..phonetics import f0_track
    a, b = (f0_track(cfg, q) for q in paths)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ref = np.zeros(n)
    both = (a > 50) & (b > 50)
    d = np.zeros(n)
    d[both] = 12 * np.log2(b[both] / a[both])
    ref[both & (np.abs(d - 12) < 1.5)] = b[both & (np.abs(d - 12) < 1.5)]
    ref[both & (np.abs(d + 12) < 1.5)] = a[both & (np.abs(d + 12) < 1.5)]
    return ref


def convert_channels(conv, cfg, y, sr, pitch, p, tmp_prefix):
    """Each channel of a stereo backing converted separately (keeps the width of doubled backing vocals)."""
    import soundfile as sf
    from .. import f0fix
    paths = []
    for ch in range(2):
        paths.append(f"{tmp_prefix}_ch{ch}.wav")
        sf.write(paths[-1], y[:, ch], sr)
    outs = []
    try:
        if p.get("prefer_upper"):
            f0fix.set_reference(channel_reference(cfg, paths))
        for ci in paths:
            co = ci[:-4] + "_out.wav"
            conv.convert(ci, co, pitch, p)
            outs.append(sf.read(co, dtype="float32")[0])
            os.remove(co)
    finally:
        f0fix.set_reference(None)
        for ci in paths:
            os.remove(ci)
    return [o if o.ndim == 1 else o.mean(1) for o in outs]


def fit(y, n):
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def make_sessions(cfg, voice_model, sep_outputs, voice_plan):
    """(lead, backing) converters: one RVC model, or several singers by the voice plan."""
    if not voice_plan:
        s = RVCSession(cfg, voice_model)
        return s, s
    log("Several voices:")
    lead = Voices(cfg, voice_plan, "lead", sep_outputs["lead_dry"])
    from .separate import backing_src
    bk = Voices(cfg, voice_plan, "backing", backing_src(sep_outputs) or sep_outputs["lead_dry"], like=lead)
    return lead, bk


def close_sessions(a, b):
    a.close()
    if b is not a:
        b.close()


def level_db(path):
    import numpy as np
    import soundfile as sf
    y, _ = sf.read(path, dtype="float32")
    return float(10 * np.log10(np.mean(y ** 2) + 1e-12))


def mfcc_mean(path):
    """Average timbre (MFCC 1–19) of the louder 60% of a file."""
    import librosa
    from my_utils import load_audio
    y = load_audio(path, 16000)
    m = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=20)
    e = librosa.feature.rms(y=y)[0]
    keep = e > np.percentile(e, 40)
    n = min(m.shape[1], len(keep))
    return m[1:, :n][:, keep[:n]].mean(axis=1)


def voice_timbre(cfg, voice_model):
    """Your timbre for comparison: from the phrases held out at training, else from the voice recordings. Cached."""
    path = os.path.join(cfg["paths"]["rvc_models"], voice_model, "timbre_mfcc.npy")
    if os.path.exists(path):
        return np.load(path)
    files = sorted(glob.glob(os.path.join(ROOT, "models", "train", voice_model, "holdout", "*.wav")))
    if not files:  # a model trained elsewhere: no reference recordings of that voice, skip the timbre guard
        return None
    v = np.mean([mfcc_mean(f) for f in files[:40]], axis=0)
    np.save(path, v)
    return v


def transplant_file(path, src_path, cons_mask):
    """Consonants of the original into a converted file (in place); per channel when both are stereo."""
    import soundfile as sf
    from .. import dsp
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    s, ssr = sf.read(src_path, dtype="float32", always_2d=True)
    s = dsp.to_sr(s, ssr, sr)
    out, info = np.zeros_like(y), None
    for ch in range(y.shape[1]):
        src = s[:, ch] if s.shape[1] == y.shape[1] else s.mean(1)
        out[:, ch], st = dsp.transplant_consonants(src, y[:, ch], cons_mask, sr)
        info = info or st
    sf.write(path, out if out.shape[1] > 1 else out[:, 0], sr, subtype="FLOAT")
    return info


def best_window(lex, total_s, win=30.0):
    starts = [w["start"] for w in lex]
    best, bs = 0, 0.0
    for s0 in starts:
        if s0 + win > total_s + 1:
            break
        k = sum(1 for t in starts if s0 <= t < s0 + win)
        if k > best:
            best, bs = k, max(0.0, s0 - 0.5)
    return bs, best


RELIABLE_ASR = {"en", "ru", "uk", "fr", "it", "es", "de", "pt", "pl", "nl"}  # Whisper hears sung words well enough


def words_reliable(language):
    """Is word recognition trustworthy for this language? (Korean covers scored 6–24 % "kept words" while sounding
    fine — there the score says nothing.)"""
    return (language or "").lower() in RELIABLE_ASR


def candidates(p0):
    ir, pr = float(p0["index_rate"]), float(p0["protect"])
    out = []
    for c in ((ir, pr), (ir, max(pr, 0.5)), (round(ir / 2, 2), max(pr, 0.5)), (min(0.8, ir + 0.25), pr)):
        if c not in out:
            out.append(c)
    return out


def f0_candidates(p0):
    """How the melody is read: (f0_method, smooth_edges). The configured one first."""
    out = [(p0.get("f0_method", "rmvpe"), bool(p0.get("smooth_edges", True)))]
    for c in (("rmvpe", True), ("rmvpe", False), ("mangio-crepe", True), ("swiftf0", True), ("hpa-rmvpe", True)):
        if c not in out:
            out.append(c)
    return out


def f0_label(method, smooth):
    return method + (" + smoothed edges" if smooth and method != "mangio-crepe" else "")


def loud_window(y, sr, win=30.0):
    """Start (s) of the 30 s where the vocal sounds most."""
    hop = sr // 10
    m = len(y) // hop
    if m * hop < win * sr:
        return 0.0
    e = (y[:m * hop].mean(1).reshape(m, hop) ** 2).mean(1)
    db = 10 * np.log10(e + 1e-12)
    act = (db > np.percentile(db, 95) - 35).astype(np.float32)
    k = int(win * 10)
    c = np.convolve(act, np.ones(k), mode="valid")
    return float(np.argmax(c) / 10)


def off_notes(cfg, src, cov, semis):
    """% of the sung frames where the cover's melody is more than a semitone away from the original's (shifted):
    catches a pitch tracker that jumps octaves or wobbles."""
    from ..phonetics import f0_track
    a, b = f0_track(cfg, src), f0_track(cfg, cov)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    v = (a > 50) & (b > 50)
    if v.sum() < 100:
        return None
    d = 12 * np.log2(b[v] / a[v]) - semis
    return float((np.abs(d) > 1).mean() * 100)


def autotune(cfg, sess, voice_model, lead_dry, words, cons, semis, stage, p0):
    """On 30 s of the song, try
      1. how the melody is read (RMVPE with smoothed note edges / plain RMVPE / crepe): the one with the least
         creak and rasp on notes the original sings cleanly wins, if it stays on the notes and keeps your timbre;
      2. index/protect (only where Whisper understands the language): the one that keeps the most words of the
         original wins, without losing your timbre.
    The defaults stay unless another setting is clearly better."""
    import soundfile as sf
    from .. import quality
    from ..dsp import to_sr
    y, sr = sf.read(lead_dry, dtype="float32", always_2d=True)
    lex = []
    language = words.get("language")
    reliable = words_reliable(language)
    if reliable:
        from .. import intelligibility as it
        lex = it.lexical(words["words"])
    s0, cnt = best_window(lex, len(y) / sr) if lex else (0.0, 0)
    if cnt < 12:
        s0 = loud_window(y, sr)
    if len(y) / sr < 8:
        log("Tuning: vocal too short — default settings")
        return p0, None
    a, b = int(s0 * sr), int((s0 + 30) * sr)
    src = stage.path("_tune_src.wav")
    sf.write(src, y[a:b], sr, subtype="FLOAT")
    xs = y[a:b].mean(1)
    voice = voice_timbre(cfg, voice_model)
    log(f"Tuning on the excerpt {int(s0 // 60)}:{int(s0 % 60):02d}–{int((s0 + 30) // 60)}:{int((s0 + 30) % 60):02d}:")

    # 1. melody
    rows_f0 = []
    for k, (method, smooth) in enumerate(f0_candidates(p0)):
        dst = stage.path(f"_tune_f0_{k}.wav")
        try:
            sess.convert(src, dst, semis, dict(p0, f0_method=method, smooth_edges=smooth))
        except Exception as e:  # e.g. crepe missing: just skip that variant
            log(f"  melody {f0_label(method, smooth)}: failed ({e})")
            continue
        c, csr = sf.read(dst, dtype="float32", always_2d=True)
        q = quality.creak(xs, to_sr(c.mean(1), csr, sr)[:len(xs)], sr)
        rows_f0.append({"f0_method": method, "smooth_edges": smooth, "creak": q["creak"], "edges": q["edges"],
                        "rough": q["rough"], "off_notes": off_notes(cfg, src, dst, semis),
                        "timbre": float(np.linalg.norm(mfcc_mean(dst) - voice)) if voice is not None else None})
        os.remove(dst)
    best_f0 = rows_f0[0] if rows_f0 else None
    if best_f0 and best_f0["creak"] is not None:
        base = best_f0
        for r in rows_f0[1:]:
            if r["creak"] is None:
                continue
            clearer = r["creak"] <= base["creak"] * 0.8 and base["creak"] - r["creak"] >= 0.3
            on_notes = r["off_notes"] is None or base["off_notes"] is None or r["off_notes"] <= base["off_notes"] + 1.0
            timbre_ok = r["timbre"] is None or base["timbre"] is None or r["timbre"] <= base["timbre"] * 1.08
            if clearer and on_notes and timbre_ok and r["creak"] < best_f0["creak"]:
                best_f0 = r
    for r in rows_f0:
        mark = " ← chosen" if r is best_f0 else ""
        cr = (f"creak {r['creak']:.1f}%" + (f" ({r['edges']:.0f}% at note edges)" if r["edges"] is not None else "")
              if r["creak"] is not None else "creak —")
        on = f", off the notes {r['off_notes']:.1f}%" if r["off_notes"] is not None else ""
        tb = f", timbre {r['timbre']:.1f}" if r["timbre"] is not None else ""
        log(f"  melody {f0_label(r['f0_method'], r['smooth_edges'])}: {cr}{on}{tb}{mark}")
    p = dict(p0)
    if best_f0:
        p.update(f0_method=best_f0["f0_method"], smooth_edges=best_f0["smooth_edges"])

    # 2. index / protect
    rows = None
    if not reliable:
        log(f"  index/protect: Whisper does not hear the song's language ({language or 'unknown'}) reliably — "
            f"default settings (words cannot be the judge here)")
    elif cnt < 12:
        log(f"  index/protect: few words in the song ({cnt} in 30 s) — default settings")
    else:
        p, rows = tune_words(cfg, sess, voice, src, s0, words, cons, semis, stage, p)
    os.remove(src)
    return p, {"pitch": rows_f0, "words": rows}


def tune_words(cfg, sess, voice, src, s0, words, cons, semis, stage, p0):
    """index/protect: the setting that keeps the most words of the original, if your timbre does not suffer."""
    from .. import intelligibility as it
    cmask = cons[int(s0 * 100):int((s0 + 30) * 100)]
    src_words = [dict(w, start=w["start"] - s0, end=w["end"] - s0) for w in words["words"] if s0 <= w["start"] < s0 + 30]
    rows, items = [], []
    for k, (ir, pr) in enumerate(candidates(p0)):
        dst = stage.path(f"_tune_{k}.wav")
        sess.convert(src, dst, semis, dict(p0, index_rate=ir, protect=pr))
        transplant_file(dst, src, cmask)
        rows.append({"index_rate": ir, "protect": pr, "wav": dst,
                     "timbre": float(np.linalg.norm(mfcc_mean(dst) - voice)) if voice is not None else None})
        items.append((dst, stage.path(f"_tune_{k}.json"), words.get("language")))
    for r, res in zip(rows, it.asr(cfg, items)):
        r["kept"], r["n"] = it.score(src_words, res["words"], 0, 30)
    for r, (_, j, _) in zip(rows, items):
        for f in (r.pop("wav"), j):
            os.remove(f)
    base = rows[0]
    if base["kept"] is None:
        return p0, rows
    ok = [r for r in rows if r["kept"] is not None and
          (r["timbre"] is None or base["timbre"] is None or r["timbre"] <= base["timbre"] * 1.08)]
    best = max(ok, key=lambda r: r["kept"])
    if best["kept"] < base["kept"] + 0.05:
        best = base
    for r in rows:
        mark = " ← chosen" if r is best else ""
        tb = f", timbre {r['timbre']:.1f}" if r["timbre"] is not None else ""
        kept = f"{r['kept']:.0%}" if r["kept"] is not None else "—"
        log(f"  index {r['index_rate']:<4} protect {r['protect']:<4}: words kept {kept}{tb}{mark}")
    return dict(p0, index_rate=best["index_rate"], protect=best["protect"]), rows


def run(cfg, voice_model, sep_outputs, stage, voice_plan=None):
    from .. import intelligibility as it
    from .. import phonetics
    p0 = dict(cfg["convert"])
    if voice_plan:
        octaves, analysis = None, None
        pitch = 0
    else:
        octaves, analysis = resolve_octaves(cfg, voice_model, sep_outputs["lead_dry"])
        pitch = octaves * 12 + int(cfg["mix"].get("pitch_change_all", 0))
    from .separate import backing_src
    bk_src = backing_src(sep_outputs)
    has_bk = bool(p0.get("convert_backing", True) and bk_src)
    if has_bk and level_db(bk_src) < -50:
        log("Hardly any backing vocal — skipped")
        has_bk = False
    tracks = [("lead", sep_outputs["lead_dry"])] + ([("backing", bk_src)] if has_bk else [])
    outputs = {}
    log("Analysing the original: words (Whisper) and consonants")
    for tag, src in tracks:
        outputs[f"words_{tag}"] = stage.path(f"words_{tag}.json")
        outputs[f"cons_{tag}"] = stage.path(f"cons_{tag}.npy")
    words = it.asr(cfg, [(src, outputs[f"words_{tag}"], None) for tag, src in tracks])
    cons = {tag: phonetics.consonants_of(cfg, src, outputs[f"cons_{tag}"]) for tag, src in tracks}
    info = {"octaves": octaves, "semitones": pitch, "pitch_analysis": analysis, "language": words[0].get("language")}
    sess, sess_bk = make_sessions(cfg, voice_model, sep_outputs, voice_plan)
    try:
        if voice_plan:
            p, table = p0, None
            info["voice_semitones"] = sess.semis
            log("Tuning: several voices — default settings")
        elif not p0.get("autotune", True):
            p, table = p0, None
            log("Tuning switched off (convert.autotune: false)")
        else:
            p, table = autotune(cfg, sess, voice_model, sep_outputs["lead_dry"], words[0], cons["lead"], pitch, stage,
                                p0)
        p.setdefault("smooth_edges", True)
        info["params"] = {k: p[k] for k in ("index_rate", "protect", "f0_method", "smooth_edges")}
        info["autotune"] = table
        log(f"Lead vocal: " + ("" if voice_plan else f"shift {pitch:+d} semitones, ") +
            f"index {p['index_rate']}, protect {p['protect']}, melody {f0_label(p['f0_method'], p['smooth_edges'])}")
        outputs["lead"] = stage.path("lead.wav")
        sess.convert(sep_outputs["lead_dry"], outputs["lead"], pitch, p)
        if has_bk:
            outputs["backing"] = stage.path("backing.wav")
            if not voice_plan and p0.get("autotune", True) and p0.get("autotune_backing", True):
                p["backing"], info["autotune_backing"] = tune_backing(cfg, sess_bk, voice_model, bk_src, pitch,
                                                                       stage, p)
                info["params"]["backing"] = p["backing"]
            p_bk = backing_params(p)
            mode = sess_bk.convert_stereo(bk_src, outputs["backing"], pitch, p_bk, stage.path("_bk"))
            log(f"Backing converted ({'per channel' if mode == 'stereo' else 'mono'}{', de-reverbed' if bk_src != sep_outputs.get('backing') else ''}): "
                f"index {p_bk['index_rate']}, protect {p_bk['protect']}, "
                f"melody {f0_label(p_bk['f0_method'], p_bk.get('smooth_edges', True))}")
            info["backing"] = mode
    finally:
        close_sessions(sess, sess_bk)
    return outputs, info
