"""Where the consonants are in a vocal: unvoiced, hissy moments (s, sh, f, t, k, breaths), 100 frames per second.

Voicing comes from RMVPE (the pitch tracker RVC itself uses); "hissy" = the band above 4 kHz is at least
a third of the 200 Hz–3 kHz band. Used by stage 3 (consonant transplant) and stage 4 (consonant balance).
"""
import numpy as np

FPS = 100


def f0_track(cfg, path):
    """RMVPE f0 per 10 ms (0 = unvoiced) for the whole file."""
    from my_utils import load_audio
    from .pitch import _model
    audio = load_audio(path, 16000)
    model = _model(cfg)
    step = 16000 * 30
    parts = [model.infer_from_audio(audio[i:i + step], thred=0.03)[: step // 160] for i in range(0, len(audio), step)]
    return np.concatenate(parts) if parts else np.zeros(0)


def consonant_mask(x, sr, f0):
    """x: mono float; f0: per 10 ms. Returns bool per 10 ms."""
    hop = sr // FPS
    n = min(len(x) // hop, len(f0))
    if n < 10:
        return np.zeros(len(f0), bool)
    win = 2 * hop
    xp = np.pad(x, (0, win))
    fr = np.lib.stride_tricks.sliding_window_view(xp, win)[::hop][:n] * np.hanning(win)
    S = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(win, 1 / sr)
    hf = S[:, (f >= 4000) & (f < 16000)].sum(1)
    lf = S[:, (f >= 200) & (f < 3000)].sum(1)
    tot = 10 * np.log10(S.sum(1) + 1e-12)
    act = tot > np.percentile(tot, 95) - 40
    m = act & (f0[:n] <= 0) & (hf >= lf / 3)
    # one frame of margin on both sides: attacks and tails of consonants
    m = m | np.r_[m[1:], False] | np.r_[False, m[:-1]]
    return np.r_[m, np.zeros(len(f0) - n, bool)]


def consonants_of(cfg, path, out_npy):
    """Mask for an audio file (mono mix of channels), saved as .npy."""
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    m = consonant_mask(y.mean(1), sr, f0_track(cfg, path))
    np.save(out_npy, m)
    return m
