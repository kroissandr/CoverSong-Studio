"""Runs in its own process (audio-separator). Called by stages/separate.py.

    python -m coversong.sep_worker job.json

job.json = {"model_dir": "...", "tmp_dir": "...", "use_autocast": true,
            "steps": [{"model": "x.ckpt", "input": "in.wav",
                       "roles": {"lead": ["vocals"], ...}, "targets": {"lead": "out.wav", ...}}]}
Each produced stem is matched to a role by its name in the output file: "<input>_(<Stem>)_<model>.wav".
"""
import json
import os
import re
import shutil
import sys
import time


def norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def main(job_path):
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    from audio_separator.separator import Separator

    os.makedirs(job["tmp_dir"], exist_ok=True)
    sep = Separator(output_dir=job["tmp_dir"], model_file_dir=job["model_dir"], output_format="WAV",
                    sample_rate=44100, use_autocast=job.get("use_autocast", True))
    loaded = None
    for i, step in enumerate(job["steps"], 1):
        t = time.time()
        if loaded != step["model"]:
            sep.load_model(model_filename=step["model"])
            loaded = step["model"]
        files = sep.separate(step["input"])
        found = {}
        for fn in files:
            path = fn if os.path.isabs(fn) else os.path.join(job["tmp_dir"], fn)
            m = re.search(r"_\(([^)]+)\)", os.path.basename(path))
            stem = norm(m.group(1)) if m else ""
            role = next((r for r, names in step["roles"].items() if stem in {norm(n) for n in names}), None)
            if role and role in step["targets"]:
                shutil.move(path, step["targets"][role])
                found[role] = stem
            else:
                os.remove(path)
        missing = [r for r in step["targets"] if r not in found]
        if missing:
            raise RuntimeError(f"Model {step['model']} gave no {missing} track; files: {files}")
        print(f"[sep] step {i}/{len(job['steps'])}: {step['model']} → {found} in {time.time() - t:.1f} s", flush=True)
    shutil.rmtree(job["tmp_dir"], ignore_errors=True)
    for item in job.get("restore", []):
        restore_levels(item["target"], item["parts"])


def restore_levels(target, parts):
    """audio-separator normalises every input to a fixed peak, so stems lose their true level.
    Find the scale of each part so that sum(parts) == target (least squares), rewrite parts as float WAV."""
    import numpy as np
    import soundfile as sf
    t, sr = sf.read(target, dtype="float32", always_2d=True)
    xs = [sf.read(p, dtype="float32", always_2d=True)[0] for p in parts]
    n = min(len(t), *(len(x) for x in xs))
    A = np.stack([x[:n].ravel() for x in xs], 1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, t[:n].ravel().astype(np.float64), rcond=None)
    res = t[:n].ravel() - A @ coef
    err = 10 * np.log10((res ** 2).mean() / ((t[:n] ** 2).mean() + 1e-20) + 1e-20)
    for p, x, c in zip(parts, xs, coef):
        sf.write(p, (x * c).astype(np.float32), sr, subtype="FLOAT")
    names = ", ".join(f"{os.path.basename(p)}×{c:.3f}" for p, c in zip(parts, coef))
    print(f"[sep] levels restored: {names} (accuracy {err:.0f} dB)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
