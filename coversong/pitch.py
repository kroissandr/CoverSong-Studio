"""Pitch analysis with RMVPE: median note of a vocal, the singer's own range, auto octave shift."""
import glob
import json
import os

import numpy as np

from .util import log

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_rmvpe = None


def note_name(midi):
    m = int(round(midi))
    return f"{NOTES[m % 12]}{m // 12 - 1}"


def _model(cfg):
    global _rmvpe
    if _rmvpe is None:
        import torch
        from rmvpe import RMVPE
        from . import assets
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _rmvpe = RMVPE(assets.ensure(cfg, "rmvpe.pt"), is_half=dev == "cuda", device=dev)
    return _rmvpe


def f0_midi(cfg, path, chunk_s=30):
    """Voiced frames of a file as MIDI note numbers (float)."""
    from my_utils import load_audio
    audio = load_audio(path, 16000)
    model = _model(cfg)
    parts = []
    step = 16000 * chunk_s
    for i in range(0, len(audio), step):
        f0 = model.infer_from_audio(audio[i:i + step], thred=0.03)
        parts.append(f0[f0 > 50])
    f0 = np.concatenate(parts) if parts else np.array([])
    return 69 + 12 * np.log2(f0 / 440.0) if len(f0) else f0


def stats(midi):
    if len(midi) < 50:
        return None
    p10, p50, p90 = np.percentile(midi, [10, 50, 90])
    return {"median": float(p50), "p10": float(p10), "p90": float(p90),
            "range": f"{note_name(p10)}–{note_name(p90)}", "median_note": note_name(p50)}


def recordings_dir(cfg, voice):
    """voices/<voice>/recordings: the studio's takes of this voice (and anything you put there yourself)."""
    return os.path.join(cfg["paths"].get("voices", "voices"), voice, "recordings")


def recordings(cfg, voice, exclude=()):
    d = recordings_dir(cfg, voice)
    files = sorted(f for ext in ("wav", "flac", "mp3", "m4a", "ogg")
                   for f in glob.glob(os.path.join(d, "**", f"*.{ext}"), recursive=True))
    ex = {e.lower() for e in exclude}
    return [f for f in files if os.path.basename(f).lower() not in ex and "_tmp" not in f.split(os.sep)]


def voice_profile(cfg, voice_model):
    """Median/range of the singer, computed once from the training recordings and cached next to the model."""
    path = os.path.join(cfg["paths"]["rvc_models"], voice_model, "voice_profile.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    files = recordings(cfg, voice_model)
    ds = recordings_dir(cfg, voice_model)
    if not files:
        raise FileNotFoundError(f"No voice recordings for the profile in {ds}")
    log(f"Measuring your voice's range from {len(files)} recordings (once)...")
    midi = np.concatenate([f0_midi(cfg, f) for f in files])
    prof = stats(midi)
    prof["files"] = len(files)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prof, f, ensure_ascii=False, indent=2)
    log(f"Your voice: middle note {prof['median_note']}, usual range {prof['range']}")
    return prof


def suggest_octaves(src_median, voice_median):
    return int(max(-2, min(2, round((voice_median - src_median) / 12.0))))


def coverage(midi, prof, k):
    """Share of the song's notes (MIDI array) that land in the voice's usual range (p10–p90, ±2 semitones) after a
    shift of k octaves."""
    m = np.asarray(midi, np.float64) + 12 * k
    return float(((m >= prof["p10"] - 2) & (m <= prof["p90"] + 2)).mean()) if len(m) else 0.0


def choose_octaves(midi, prof):
    """Octave shift that puts the most notes of the song into the voice's range (not just its middle note: a song
    whose verses and chorus sit an octave apart needs the shift that suits both). Within 3 % of the best,
    the shift whose middle note is closest to the voice's wins. Returns (k, coverage of k, coverage of the
    middle-note shift)."""
    midi = np.asarray(midi, np.float64)
    if len(midi) < 50 or prof.get("p10") is None or prof.get("p90") is None:
        k = suggest_octaves(float(np.median(midi)) if len(midi) else prof["median"], prof["median"])
        return k, None, None
    med = float(np.median(midi))
    cov = {k: coverage(midi, prof, k) for k in range(-2, 3)}
    best = max(cov.values())
    k = min((k for k in cov if cov[k] >= best - 0.03), key=lambda k: abs(med + 12 * k - prof["median"]))
    return k, cov[k], cov[suggest_octaves(med, prof["median"])]
