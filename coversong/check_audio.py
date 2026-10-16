"""Checks voice recordings before training: length, clipping, noise in pauses and — most important — the highs.

Are 8–14 kHz (breath, consonants, overtones) clearly above the noise floor? If not, the recording went through a
noise suppressor / "voice enhancement" or a narrow band, and a model trained on it sounds dull or metallic.

    python -m coversong.check_audio <file or folder> [...]
"""
import glob
import os
import subprocess
import sys

import numpy as np

SR = 48000
EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".mp4", ".mkv", ".mov")


def load(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.float32)


def band_db(S, f, a, b, tot):
    m = (f >= a) & (f < b)
    return 10 * np.log10(S[:, m].sum() / tot / ((b - a) / 1000) + 1e-20)


def analyze(path):
    x = load(path)
    if len(x) < SR:
        return None
    n, hop = 4096, 2048
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    rms = np.sqrt((fr ** 2).mean(1)) + 1e-12
    db = 20 * np.log10(rms)
    S = np.abs(np.fft.rfft(fr * np.hanning(n), axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / SR)
    loud = db > np.percentile(db, 50)
    Sl = S[loud]
    tot = Sl.sum()
    b34, b45 = band_db(Sl, f, 3000, 4000, tot), band_db(Sl, f, 4200, 5000, tot)
    b68, b1016 = band_db(Sl, f, 6000, 8000, tot), band_db(Sl, f, 10000, 16000, tot)
    hb = 10 * np.log10(S[:, (f >= 8000) & (f < 14000)].mean(1) + 1e-30)
    hf_alive = float(np.median(hb[loud]) - np.percentile(hb, 5))
    # short takes without pauses (vocal studio) have no quiet part to compare with: also give the level of
    # 8–14 kHz against 200–1000 Hz while singing (songs: about −45 dB, sung vowels −55…−68, cut-off: below −80)
    hf_rel = band_db(Sl, f, 8000, 14000, tot) - band_db(Sl, f, 200, 1000, tot)
    return {
        "dur": len(x) / SR,
        "clip": float(100 * np.mean(np.abs(x) > 0.99)),
        "peak": float(20 * np.log10(np.abs(x).max() + 1e-12)),
        "noise": float(np.percentile(db, 5)),
        "cliff4k": float(b34 - b45),
        "air": float(b1016),
        "b68": float(b68),
        "hf_alive": hf_alive,
        "hf_rel": float(hf_rel),
        "range": float(np.percentile(db, 95) - np.percentile(db, 5)),
    }


def verdict(r):
    probs = []
    if r["hf_alive"] < 8:
        probs.append(f"no highs (8–14 kHz only {r['hf_alive']:.0f} dB above the noise) — a noise suppressor?")
    if r["clip"] > 0.01:
        probs.append(f"clipping {r['clip']:.2f}% — turn the input gain down")
    if r["peak"] < -18:
        probs.append(f"quiet (peak {r['peak']:.0f} dB) — turn the input gain up")
    if r["noise"] > -55:
        probs.append(f"noisy pauses ({r['noise']:.0f} dB)")
    return probs


def main(paths):
    files = []
    for p in paths:
        p = p.strip('"')
        if os.path.isdir(p):
            files += sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(EXTS))
        else:
            files.append(p)
    total = bad = 0
    print(f"{'file':40s} {'min':>5s} {'highs':>6s} {'noise':>6s} {'peak':>5s}  verdict")
    for fp in files:
        r = analyze(fp)
        if not r:
            print(f"{os.path.basename(fp)[:40]:40s}  too short")
            continue
        total += r["dur"]
        probs = verdict(r)
        bad += bool(probs)
        print(f"{os.path.basename(fp)[:40]:40s} {r['dur'] / 60:5.1f} {r['hf_alive']:6.0f} {r['noise']:6.0f} {r['peak']:5.0f}  "
              + ("OK" if not probs else "; ".join(probs)))
    print(f"\nTotal: {len(files)} files, {total / 60:.1f} min. With problems: {bad}.")
    print("Good: highs 8 dB or more, noise below -55 dB, peak between -18 and -1 dB. 15–30 minutes are enough to train.")


if __name__ == "__main__":
    main(sys.argv[1:])
