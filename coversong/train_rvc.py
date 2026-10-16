"""Training a voice model (RVC v2, 48 kHz) from the vocal studio's recordings — Studio → Train in the app, or

    python -m coversong.train_rvc --voice NAME                 # everything in order
    python -m coversong.train_rvc --voice NAME --from evaluate # pick the best version again
    python -m coversong.train_rvc --voice NAME --install e180  # install another version

Steps are remembered: running again continues where it stopped (the training itself too).
  dataset    — the voice's recordings (voices/<name>/recordings and train.extra_recordings): highs and clipping
               checked, cut at phrases; ~8% of the phrases of each recording held out for an honest check
  pretrain   — the base model: TITAN-Medium 48 kHz (Apache-2.0, downloaded once), else the RVC v2 48 kHz base
  preprocess — slicing, 48 kHz        \
  extract    — melody (RMVPE) and     |  Applio's RVC engine (MIT): coversong/third_party/applio
               ContentVec features    |
  train      — training; a version of the model is saved every N epochs
  index      — the timbre index       /
  evaluate   — every version re-sings the held-out phrases (and song excerpts) → metrics → the best version,
               a "listen" folder with a blind comparison of the two best
  install    — the best version → rvc_models/<name>
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time

import numpy as np

from . import ROOT
from .config import load_config
from .util import log

STEPS = ["dataset", "pretrain", "preprocess", "extract", "train", "index", "evaluate", "install"]
SR = 48000
DATASET_VERSION = 3
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac")
TITAN = {"repo": "blaise-tk/TITAN",
         "G": "models/medium/48k/pretrained/G-f048k-TITAN-Medium.pth",
         "D": "models/medium/48k/pretrained/D-f048k-TITAN-Medium.pth"}
DEFAULTS = dict(extra_recordings=[], exclude=[], epochs="auto", batch_size=8, holdout=0.08, target_steps=20000,
                save_every="auto", eval_songs="auto")
APPLIO = os.path.join(ROOT, "coversong", "third_party", "applio")
MIN_TRAIN_MIN = 3.0
GOOD_TRAIN_MIN = 15.0
PROGRESS = None   # the app sets a callback(step, fraction 0..1, text)


def progress(step, frac, text=""):
    if PROGRESS:
        try:
            PROGRESS(step, frac, text)
        except Exception:  # noqa: BLE001
            pass


# ================================================================== context / state
class Ctx:
    def __init__(self, cfg, voice):
        t = dict(DEFAULTS)
        t.update(cfg.get("train") or {})
        self.cfg, self.t = cfg, t
        self.name = re.sub(r"[^A-Za-z0-9_-]", "_", str(voice))
        from . import pitch
        self.rec = [pitch.recordings_dir(cfg, self.name)]
        extra = t.get("extra_recordings") or []
        self.rec += [os.path.normpath(os.path.join(ROOT, str(r))) for r in (extra if isinstance(extra, list) else [extra])]
        self.work = os.path.join(ROOT, "models", "train", self.name)
        self.train_set = os.path.join(self.work, "train_set")
        self.holdout = os.path.join(self.work, "holdout")
        self.logs = os.path.join(self.work, "engine")
        self.exp = os.path.join(self.logs, self.name)
        self.state_path = os.path.join(self.work, "state.json")
        os.makedirs(self.work, exist_ok=True)
        self.state = {}
        if os.path.exists(self.state_path):
            with open(self.state_path, encoding="utf-8") as f:
                self.state = json.load(f)

    def save(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.state_path)

    def done(self, step):
        return step in self.state.get("done", [])

    def mark(self, step, info=None):
        d = [s for s in self.state.get("done", []) if s != step] + [step]
        self.state["done"] = d
        if info is not None:
            self.state[step] = info
        self.save()

    def reset_from(self, step):
        k = STEPS.index(step)
        self.state["done"] = [s for s in self.state.get("done", []) if s in STEPS and STEPS.index(s) < k]
        self.save()


def engine_env(ctx):
    env = dict(os.environ, APPLIO_ROOT=APPLIO, APPLIO_CODE_ROOT=APPLIO, APPLIO_LOGS_DIR=ctx.logs,
               APPLIO_CONFIG_FILE=os.path.join(ROOT, "models", "train", "engine_settings.json"),
               PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    env["PYTHONPATH"] = APPLIO + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_engine(ctx, args, label, on_line=None):
    """Run one of Applio's scripts (cwd = its folder, as it expects). Returns (exit code, last lines)."""
    cmd = [sys.executable, "-u", *[str(a) for a in args]]
    log(f"{label}…")
    p = subprocess.Popen(cmd, cwd=APPLIO, env=engine_env(ctx), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tail = []
    con = open(os.path.join(ctx.work, "engine_console.txt"), "a", encoding="utf-8")
    con.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {label}: {' '.join(cmd)}\n")
    for raw in iter(p.stdout.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip()
        if not line.strip():
            continue
        con.write(line + "\n")
        con.flush()
        tail = (tail + [line])[-60:]
        show = on_line(line) if on_line else False
        if show:
            log("    " + (show if isinstance(show, str) else line))
    code = p.wait()
    con.close()
    return code, tail


class TrainError(RuntimeError):
    pass


def fail(msg, tail=()):
    if tail:
        log("\n".join("    | " + t for t in tail[-25:]))
    raise TrainError(msg)


def link_or_copy(src, dst):
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def engine_models(ctx):
    """The engine reads rmvpe.pt and ContentVec from its own folders: the copies the app already downloaded."""
    from . import assets
    link_or_copy(assets.ensure(ctx.cfg, "rmvpe.pt"), os.path.join(APPLIO, "rvc", "models", "predictors", "rmvpe.pt"))
    cv = assets.contentvec_dir(ctx.cfg)
    for f in ("config.json", "pytorch_model.bin"):
        link_or_copy(os.path.join(cv, f), os.path.join(APPLIO, "rvc", "models", "embedders", "contentvec", f))


# ================================================================== 1. dataset
def list_recordings(ctx):
    ex = {str(e).lower() for e in (ctx.t.get("exclude") or [])}
    files = []
    for d in ctx.rec:
        if not os.path.isdir(d):
            continue
        files += sorted(f for f in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                        if f.lower().endswith(AUDIO_EXTS) and os.path.basename(f).lower() not in ex
                        and "_tmp" not in f.split(os.sep))
    return files


def rel(path):
    return os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))


def dataset_key(ctx, files):
    h = hashlib.sha1()
    for f in files:
        h.update(f"{rel(f)}|{os.path.getsize(f)}|{int(os.path.getmtime(f))}\n".encode("utf-8"))
    h.update(json.dumps([ctx.t["holdout"], DATASET_VERSION]).encode())
    return h.hexdigest()[:12]


def load48(path):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


def highpass(x, hz=40.0):
    import scipy.signal as ss
    sos = ss.butter(4, hz, "highpass", fs=SR, output="sos")
    return ss.sosfiltfilt(sos, x - x.mean())


def frame_db(x, hop=SR // 100):
    n = len(x) // hop
    fr = x[:n * hop].reshape(n, hop)
    db = 10 * np.log10((fr ** 2).mean(1) + 1e-12)
    return np.convolve(db, np.ones(3) / 3, mode="same"), hop


def find_phrases(x):
    """Sung phrases as (start, end) samples: active frames, gaps < 0.35 s merged, long ones split at quiet points."""
    db, hop = frame_db(x)
    if len(db) < 10:
        return [], db, hop
    thr = max(np.percentile(db, 5) + 12, np.percentile(db, 99) - 40)
    if np.percentile(db, 99) - np.percentile(db, 5) < 18:  # sung from start to end (a vocal-studio take)
        thr = np.percentile(db, 99) - 30
    act = (db > thr).astype(int)
    edges = np.flatnonzero(np.diff(np.r_[0, act, 0]))
    merged = []
    for a, b in zip(edges[::2], edges[1::2]):
        if merged and a - merged[-1][1] < 35:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        if b - a < 25:
            continue
        while b - a > 1200:  # > 12 s: cut near 6 s at the quietest 10 ms within ±1 s
            lo, hi = a + 500, a + 700
            c = lo + int(np.argmin(db[lo:hi]))
            out.append((a, c))
            a = c
        out.append((a, b))
    return [(a * hop, b * hop) for a, b in out], db, hop


def clipped(seg):
    hot = np.abs(seg) > 0.985
    if hot.sum() < 3:
        return False
    run = np.convolve(hot.astype(int), np.ones(3, int), mode="valid")
    return bool((run >= 3).any())


def safe_stem(path, i):
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(os.path.basename(path))[0]).strip("_")[:40]
    return f"{i:02d}_{s}" if s else f"rec{i:02d}"


def split_file(x, frac):
    """Assign each phrase to train / hold / drop; returns spans and labels."""
    ph, db, hop = find_phrases(x)
    if not ph:
        return [], [], []
    bounds = [0] + [(ph[i - 1][1] + ph[i][0]) // 2 for i in range(1, len(ph))] + [len(x)]
    own = [(bounds[i], bounds[i + 1]) for i in range(len(ph))]
    labels = ["drop" if clipped(x[a:b]) else "train" for a, b in ph]
    act_s = sum(b - a for a, b in ph) / SR
    cands = [i for i, (a, b) in enumerate(ph) if labels[i] == "train" and 1.5 * SR <= b - a <= 12 * SR]
    if act_s >= 30 and cands:
        avg = np.mean([(ph[i][1] - ph[i][0]) / SR for i in cands])
        k = max(1, int(round(act_s * frac / avg)))
        k = min(k, max(1, len(cands) // 3))
        for j in range(k):
            labels[cands[int((j + 0.5) * len(cands) / k)]] = "hold"
    return ph, own, labels


def step_dataset(ctx, force=False):
    import soundfile as sf
    from . import check_audio

    files = list_recordings(ctx)
    if not files:
        fail(f"no recordings (wav/flac/mp3) in {', '.join(ctx.rec)} — record some in the vocal studio first")
    key = dataset_key(ctx, files)
    old = ctx.state.get("dataset", {})
    if ctx.done("dataset") and old.get("key") == key:
        log(f"Recordings unchanged ({old.get('train_min', 0):.1f} min to train on) — no new cutting")
        return False
    if ctx.done("dataset"):
        log("The recordings changed — the model is trained again from the start with all of them")
        shutil.rmtree(ctx.logs, ignore_errors=True)

    for d in (ctx.train_set, ctx.holdout):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    frac = float(ctx.t["holdout"])
    report, all_hold = [], []
    tot_train = tot_hold = 0.0
    log(f"Recordings: {len(files)} files")
    for i, path in enumerate(files):
        progress("dataset", i / max(1, len(files)), os.path.basename(path))
        base = rel(path)
        info = check_audio.analyze(path)
        if info is None:
            log(f"  {base}: shorter than a second — skipped")
            continue
        notes = []
        # short takes (vocal studio) have no real silence to measure the noise floor against: there the level of
        # 8–14 kHz against the voice itself decides
        pauses = info["range"] >= 20 and info["dur"] >= 30
        no_hf = info["hf_alive"] < 8 if pauses else info["hf_rel"] < -80
        if no_hf and "breath" not in os.path.basename(path).lower():
            log(f"  {base}: SKIPPED — no highs (a noise suppressor or a narrow band); a model learns to sound dull from it")
            report.append({"file": base, "skipped": "no_hf"})
            continue
        if pauses and info["noise"] > -55:
            notes.append(f"noise in pauses {info['noise']:.0f} dB")
        x = load48(path)
        ph, own, labels = split_file(x, frac)
        if not ph:
            log(f"  {base}: no singing found — skipped")
            continue
        y = highpass(x)
        db, _ = frame_db(y)
        loud = np.percentile(db[db > np.percentile(db, 50)], 90) if len(db) else -20
        g = min(10 ** ((-16 - loud) / 20), 10 ** (-1 / 20) / (np.abs(y).max() + 1e-9))
        y = (y * g).astype(np.float32)
        stem = safe_stem(path, i)
        # train chunks: runs of consecutive "train" phrases (each phrase owns the span up to its neighbours)
        n_tr = n_ho = 0
        t_tr = t_ho = 0.0
        run = None
        for j, (lab, (a, b)) in enumerate(zip(labels + ["end"], own + [(len(y), len(y))])):
            if lab == "train":
                run = [run[0] if run else a, b]
                continue
            if run and run[1] - run[0] >= SR:
                n_tr += 1
                sf.write(os.path.join(ctx.train_set, f"{stem}_{n_tr:03d}.wav"), y[run[0]:run[1]], SR, subtype="PCM_24")
                t_tr += sum(min(pb, run[1]) - max(pa, run[0]) for pa, pb in ph if pb > run[0] and pa < run[1]) / SR
            run = None
            if lab == "hold":
                pa, pb = ph[j]
                a2, b2 = max(a, pa - int(0.15 * SR)), min(b, pb + int(0.15 * SR))
                n_ho += 1
                hp = os.path.join(ctx.holdout, f"{stem}_h{n_ho:02d}.wav")
                sf.write(hp, y[a2:b2], SR, subtype="PCM_24")
                all_hold.append(os.path.basename(hp))
                t_ho += (pb - pa) / SR
        drops = labels.count("drop")
        if drops:
            notes.append(f"clipping: {drops} phrases left out")
        tot_train += t_tr
        tot_hold += t_ho
        report.append({"file": base, "dur": round(len(x) / SR, 1), "train_s": round(t_tr, 1), "hold_s": round(t_ho, 1),
                       "dropped": drops, "notes": notes})
        log(f"  {base}: singing {t_tr + t_ho:5.0f} s → training {t_tr:5.0f} s, check {t_ho:4.0f} s"
            + (f"  ({'; '.join(notes)})" if notes else ""))

    train_min = tot_train / 60
    log(f"To train on: {train_min:.1f} min of singing; held out for the check: {tot_hold / 60:.1f} min ({len(all_hold)} phrases)")
    if train_min < MIN_TRAIN_MIN and not force:
        fail(f"too little singing to train ({train_min:.1f} min; at least {MIN_TRAIN_MIN:.0f} needed, "
             f"{GOOD_TRAIN_MIN:.0f}+ is better). Record more in the studio")
    if train_min < GOOD_TRAIN_MIN:
        log(f"Note: {GOOD_TRAIN_MIN:.0f}+ minutes of singing give a better model")
    if not all_hold:
        log("Note: the recordings are too short to hold phrases out — the best version is chosen by songs only")

    pitch_info = pitch_coverage(ctx)
    ctx.reset_from("dataset")
    ctx.mark("dataset", {"key": key, "train_min": round(train_min, 2), "hold_min": round(tot_hold / 60, 2),
                         "holdout": all_hold, "files": report, "pitch": pitch_info})
    return True


def pitch_job(cfg, folder, out):
    """Runs in a separate process so the GPU is completely free for training afterwards."""
    from . import pitch
    midi = np.concatenate([pitch.f0_midi(cfg, f) for f in sorted(glob.glob(os.path.join(folder, "*.wav")))])
    st = pitch.stats(midi)
    if st:
        notes = np.round(midi).astype(int)
        lo, hi = int(np.percentile(notes, 1)), int(np.percentile(notes, 99))
        st["lo"], st["hi"] = pitch.note_name(lo), pitch.note_name(hi)
        st["seconds_per_note"] = {pitch.note_name(m): round(float((notes == m).sum()) * 0.01, 1) for m in range(lo, hi + 1)}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)


def pitch_coverage(ctx):
    """How many seconds of singing per note: shows gaps in the range (and the studio fills them first)."""
    out = os.path.join(ctx.work, "pitch.json")
    if os.path.exists(out):
        os.remove(out)
    log("Counting the singing per note…")
    from .util import worker_python
    py, env = worker_python()
    r = subprocess.run([py, "-u", "-m", "coversong.train_rvc", "--pitch-of", ctx.train_set, out], cwd=ROOT, env=env)
    if r.returncode or not os.path.exists(out):
        log("Could not count the range — continuing without it")
        return None
    with open(out, encoding="utf-8") as f:
        st = json.load(f)
    if not st:
        return None
    secs = st["seconds_per_note"]
    log(f"Range in the recordings: {st['lo']}–{st['hi']}, middle note {st['median_note']}")
    log("Seconds of singing per note: " + "  ".join(f"{n} {v:.0f}" for n, v in secs.items()))
    weak = [n for n, v in secs.items() if v < 5]
    if weak:
        log(f"Little material (< 5 s) on: {', '.join(weak)}")
    return st


# ================================================================== 2. pretrain
def step_pretrain(ctx):
    dst = os.path.join(ROOT, "models", "pretrained", "titan")
    res = None
    try:
        g, d = (os.path.join(dst, *TITAN[k].split("/")) for k in ("G", "D"))
        if not (os.path.exists(g) and os.path.exists(d)):
            from huggingface_hub import hf_hub_download
            log("Base model TITAN-Medium 48 kHz (Apache-2.0): one-time download, ~1.3 GB…")
            g = hf_hub_download(TITAN["repo"], TITAN["G"], local_dir=dst)
            d = hf_hub_download(TITAN["repo"], TITAN["D"], local_dir=dst)
        res = {"G": g, "D": d, "kind": "TITAN-Medium-48k"}
    except Exception as e:  # noqa: BLE001
        log(f"Could not get TITAN ({type(e).__name__}: {e}) — taking the standard RVC v2 base")
    if not res:
        got = {}

        def grab(line):
            if line.startswith("PRETRAINED "):
                got["G"], got["D"] = line[11:].split("|")
            return False
        code, tail = run_engine(ctx, ["-c", "from rvc.lib.tools.prerequisites_download import ensure_pretrained; "
                                            "print('PRETRAINED ' + '|'.join(ensure_pretrained('HiFi-GAN', 48000)))"],
                                "Downloading the RVC v2 48 kHz base (once)", grab)
        if code or not got:
            fail("no base model could be downloaded — check the internet connection", tail)
        res = {"G": got["G"].strip(), "D": got["D"].strip(), "kind": "RVC-v2-48k"}
    log(f"Base: {res['kind']} ✓")
    ctx.mark("pretrain", res)


# ================================================================== 3–4. slicing, melody and features
def count(d, pattern="*"):
    return len(glob.glob(os.path.join(d, pattern)))


def step_preprocess(ctx):
    if os.path.isdir(ctx.exp):
        log(f"Removing the previous slices and versions of {ctx.name} (new recordings)")
        shutil.rmtree(ctx.exp)
    os.makedirs(ctx.exp)
    n_p = max(1, min(8, (os.cpu_count() or 4) // 2))
    code, tail = run_engine(ctx, [os.path.join("rvc", "train", "preprocess", "preprocess.py"), ctx.exp, ctx.train_set,
                                  SR, n_p, "Automatic", False, 3.0, 0.3, "none"], "Slicing (48 kHz)")
    n = count(os.path.join(ctx.exp, "sliced_audios"), "*.wav")
    if code != 0 or n == 0:
        fail("slicing failed", tail)
    log(f"Slices: {n}")
    ctx.mark("preprocess", {"slices": n})


def step_extract(ctx):
    engine_models(ctx)
    import torch
    gpu = "0" if torch.cuda.is_available() else "-"
    n_p = max(1, min(8, (os.cpu_count() or 4) // 2))
    code, tail = run_engine(ctx, [os.path.join("rvc", "train", "extract", "extract.py"), ctx.exp, "rmvpe", n_p, gpu, SR,
                                  "contentvec", "None", 2], "Melody (RMVPE) and features (ContentVec)")
    n, m = count(os.path.join(ctx.exp, "sliced_audios"), "*.wav"), count(os.path.join(ctx.exp, "extracted"), "*.npy")
    if code != 0 or m < n:
        fail(f"features made for {m} of {n} slices", tail)
    ctx.mark("extract")


# ================================================================== 5. train
def plan_epochs(ctx, items, bs):
    spe = math.ceil(items / bs)
    ep = ctx.t.get("epochs", "auto")
    if str(ep).lower() == "auto":
        ep = int(round(float(ctx.t["target_steps"]) / spe / 10) * 10)
        ep = min(600, max(100, ep))
    ep = int(ep)
    se = ctx.t.get("save_every", "auto")
    if str(se).lower() == "auto":
        se = max(5, int(round(ep / 25 / 5) * 5))
    se = int(se)
    ep = int(math.ceil(ep / se) * se)
    return ep, se, spe


def step_train(ctx, epochs_override=None):
    pre = ctx.state["pretrain"]
    items = count(os.path.join(ctx.exp, "sliced_audios"), "*.wav") + 2
    st = ctx.state.get("train", {})
    bs = int(st.get("batch_size") or ctx.t["batch_size"])
    if epochs_override:
        ctx.t["epochs"] = epochs_override
        st.pop("epochs", None)
    if st.get("epochs"):
        epochs, se = st["epochs"], st["save_every"]
        spe = math.ceil(items / bs)
    else:
        epochs, se, spe = plan_epochs(ctx, items, bs)
    st.update(epochs=epochs, save_every=se, batch_size=bs, items=items, base=pre["kind"])
    ctx.state["train"] = st
    ctx.save()
    resume = glob.glob(os.path.join(ctx.exp, "G_*.pth"))
    log(f"Training: {epochs} epochs of {spe} steps (batch {bs}), a version saved every {se} epochs; "
        f"base {pre['kind']}" + ("; continuing from the saved point" if resume else ""))
    log("You can use the computer meanwhile (no games or heavy GPU work). If it stops, start the training again — "
        "it continues from the last saved version.")
    import torch
    gpu = "0" if torch.cuda.is_available() else "-"
    keep_awake(True)
    try:
        while True:
            t0, times, last = time.time(), [], [None]
            finished = [False]

            def on_line(line):
                m = re.search(r"\| epoch=(\d+) \| step=\d+ \| (?!batch)", line)
                if m:
                    ep = int(m.group(1))
                    now = time.time()
                    if last[0] is not None:
                        times.append(now - last[0])
                    last[0] = now
                    avg = np.mean(times[-10:]) if times else None
                    eta = f", ~{fmt_dur(avg * (epochs - ep))} left" if avg else ""
                    progress("train", ep / epochs, f"epoch {ep}/{epochs}{eta}")
                    return f"epoch {ep}/{epochs}{eta}"
                if "trained successfully" in line:
                    finished[0] = True
                    return "training finished"
                if re.search(r"Traceback|out of memory|^\s*\w*(Error|Exception)\b", line) and "Warning" not in line:
                    return line
                return False
            code, tail = run_engine(ctx, [os.path.join("rvc", "train", "train.py"), ctx.name, se, epochs, pre["G"],
                                          pre["D"], gpu, bs, SR, False, True, False, False, "HiFi-GAN", False],
                                    "Training", on_line)
            if finished[0] or (ckpt_list(ctx) and ckpt_list(ctx)[-1][0] >= epochs):
                break
            if any("out of memory" in t.lower() for t in tail) and bs > 2:
                bs = max(2, bs - 2)
                st["batch_size"] = bs
                ctx.save()
                log(f"Out of GPU memory — batch down to {bs}, continuing")
                continue
            fail(f"training stopped (code {code}). Start it again — it continues from the last saved version", tail)
    finally:
        keep_awake(False)
    log(f"Training took {fmt_dur(time.time() - t0)}")
    ctx.mark("train", st)


def keep_awake(on):
    """Windows must not go to sleep while the model trains (the screen may still turn off)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:  # noqa: BLE001
        pass


def fmt_dur(s):
    s = int(s)
    return f"{s // 3600} h {s % 3600 // 60:02d} min" if s >= 3600 else f"{s // 60} min"


def ckpt_list(ctx):
    """Saved versions: <name>_<epoch>e_<step>s.pth in the engine's experiment folder."""
    pat = re.compile(re.escape(ctx.name) + r"_(\d+)e_(\d+)s\.pth$")
    out = []
    for f in os.listdir(ctx.exp) if os.path.isdir(ctx.exp) else []:
        m = pat.match(f)
        if m:
            out.append((int(m.group(1)), os.path.join(ctx.exp, f)))
    return sorted(out)


# ================================================================== 6. index
def step_index(ctx):
    path = os.path.join(ctx.exp, f"{ctx.name}.index")
    if os.path.exists(path):
        os.remove(path)
    code, tail = run_engine(ctx, [os.path.join("rvc", "train", "process", "extract_index.py"), ctx.exp, "Auto"],
                            "Timbre index")
    if code != 0 or not os.path.exists(path):
        fail("the index was not built", tail)
    ctx.mark("index", {"path": path})


# ================================================================== 8. evaluate
def resample(y, sr_in, sr_out=SR):
    from .dsp import to_sr
    return to_sr(np.asarray(y, np.float32), sr_in, sr_out)


def read48(path):
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    return resample(y.mean(1), sr)


def stft_pow(y, n_fft=2048, hop=480):
    import scipy.signal as ss
    _, _, Z = ss.stft(y, SR, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    return np.abs(Z) ** 2


def mel_fb(n_fft=2048, n_mels=80, fmin=50, fmax=16000):
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    f = np.linspace(0, SR / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(f)))
    for i in range(n_mels):
        a, c, b = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((f - a) / (c - a), (b - f) / (b - c)), 0, None)
    return fb


_FB = None


def pair_metrics(ref, out):
    """Distance between your real phrase and its re-sung version: mel-cepstral distortion + 4–16 kHz difference."""
    global _FB
    from scipy.fft import dct
    n = min(len(ref), len(out))
    ref, out = ref[:n], out[:n]
    # RVC output keeps the input's timing exactly, so no alignment is needed
    Pa, Pb = stft_pow(ref), stft_pow(out)
    m = min(Pa.shape[1], Pb.shape[1])
    Pa, Pb = Pa[:, :m], Pb[:, :m]
    ea, eb = Pa.sum(0), Pb.sum(0)
    act = ea > ea.max() * 10 ** (-35 / 10)
    if act.sum() < 10:
        return None
    Pb = Pb / max(float(np.median(eb[act] / ea[act])), 1e-12)  # ignore overall level difference
    floor = Pa.max() * 1e-6  # -60 dB below the loudest bin: the recording's noise floor does not count
    A, B = 10 * np.log10(Pa + floor), 10 * np.log10(Pb + floor)
    if _FB is None:
        _FB = mel_fb()
    Ma, Mb = _FB @ Pa, _FB @ Pb
    mfloor = Ma.max() * 1e-5
    la = 0.5 * np.log(Ma + mfloor)
    lb = 0.5 * np.log(Mb + mfloor)
    ca = dct(la, type=2, norm="ortho", axis=0)[1:25]
    cb = dct(lb, type=2, norm="ortho", axis=0)[1:25]
    mcd = (10 / np.log(10)) * np.sqrt(2 * ((ca - cb) ** 2).sum(0))
    f = np.linspace(0, SR / 2, A.shape[0])
    band = (f >= 4000) & (f <= 16000)
    hf = np.sqrt(((A[band] - B[band]) ** 2).mean(0))
    return float(mcd[act].mean()), float(hf[act].mean())


def pick_eval_songs(ctx):
    v = ctx.t.get("eval_songs", "auto")
    out_root = ctx.cfg["paths"]["output"]
    if isinstance(v, list):
        cands = [os.path.join(out_root, s, "01_separate", "lead_dry.wav") for s in v]
    else:
        cands = [p for p in glob.glob(os.path.join(out_root, "*", "01_separate", "lead_dry.wav"))
                 if not re.search(r"_x[\d-]+$", os.path.basename(os.path.dirname(os.path.dirname(p))))]
        cands.sort(key=os.path.getmtime, reverse=True)
        cands = cands[:2]
    return [p for p in cands if os.path.exists(p)]


def prepare_eval(ctx, ev_dir):
    """Holdout phrases (≤ ~2 min) and 25 s excerpts of separated song vocals, with the octave shift for your voice."""
    import soundfile as sf
    from . import pitch
    from .dsp import loud_excerpt
    hold = [os.path.join(ctx.holdout, h) for h in ctx.state["dataset"].get("holdout", [])]
    hold = [h for h in hold if os.path.exists(h)]
    durs = [sf.info(h).duration for h in hold]
    if sum(durs) > 120:
        k = max(4, int(len(hold) * 120 / sum(durs)))
        idx = np.linspace(0, len(hold) - 1, k).round().astype(int)
        hold = [hold[i] for i in sorted(set(idx))]
    songs = []
    vmed = (ctx.state["dataset"].get("pitch") or {}).get("median")
    for i, p in enumerate(pick_eval_songs(ctx)):
        y, sr = sf.read(p, dtype="float32", always_2d=True)
        ex = loud_excerpt(y.mean(1), 25, sr)
        dst = os.path.join(ev_dir, f"song{i + 1}.wav")
        sf.write(dst, ex, sr)
        st = pitch.stats(pitch.f0_midi(ctx.cfg, dst))
        semis = 12 * pitch.suggest_octaves(st["median"], vmed) if (st and vmed) else 0
        songs.append({"src": dst, "song": os.path.basename(os.path.dirname(os.path.dirname(p))), "semis": semis})
    return hold, songs


def step_evaluate(ctx):
    from .stages.convert import RVCSession, find_model

    ev_dir = os.path.join(ctx.work, "eval")
    shutil.rmtree(ev_dir, ignore_errors=True)
    os.makedirs(ev_dir)
    hold, songs = prepare_eval(ctx, ev_dir)
    ck = ckpt_list(ctx)
    if not ck:
        fail(f"no saved versions {ctx.name}_*e_*s.pth in {ctx.exp}")
    if not hold and not songs:
        e, p = ck[-1]
        log("Nothing to compare the versions on — the last one is taken")
        ctx.mark("evaluate", {"rows": [{"tag": f"e{e}", "pth": p}], "best": f"e{e}", "ranking": [f"e{e}"]})
        return
    index = ctx.state["index"]["path"]
    cands = [(f"e{e}", p, index) for e, p in ck]
    try:
        bp, bi = find_model(ctx.cfg, ctx.name)
        cands.append(("old", bp, bi))
    except (FileNotFoundError, TypeError):
        pass
    log(f"Checking {len(ck)} versions: {len(hold)} held-out phrases, songs: {', '.join(s['song'] for s in songs) or 'none'}")
    voice_mfcc = np.mean([mfcc_mean(h) for h in hold], axis=0) if hold else None
    refs = {h: read48(h) for h in hold}
    p = dict(ctx.cfg["convert"])
    rows = []
    for k, (tag, pth, idx) in enumerate(cands):
        progress("evaluate", k / len(cands), tag)
        d = os.path.join(ev_dir, tag)
        os.makedirs(d)
        sess = RVCSession(ctx.cfg, None, pth=pth, index=idx)
        try:
            mc, hf = [], []
            for h in hold:
                dst = os.path.join(d, os.path.basename(h))
                sess.convert(h, dst, 0, p)
                r = pair_metrics(refs[h], read48(dst))
                if r:
                    mc.append(r[0])
                    hf.append(r[1])
            fl, tb = [], []
            for i, s in enumerate(songs):
                dst = os.path.join(d, f"song{i + 1}.wav")
                sess.convert(s["src"], dst, s["semis"], p)
                fl.append(flatness(dst, f0_track(ctx.cfg, dst)))
                if voice_mfcc is not None:
                    tb.append(float(np.linalg.norm(mfcc_mean(dst) - voice_mfcc)))
        finally:
            sess.close()
        row = {"tag": tag, "pth": pth, "index": idx,
               "mcd": float(np.mean(mc)) if mc else None, "hf": float(np.mean(hf)) if hf else None,
               "flat": float(np.nanmean(fl)) if fl else None, "timbre": float(np.mean(tb)) if tb else None}
        rows.append(row)
        log(f"  {tag:>5}: timbre accuracy {fmtv(row['mcd'])}, highs {fmtv(row['hf'])}, "
            f"metal in songs {fmtv(row['flat'], 4)}, likeness {fmtv(row['timbre'])}")

    score(rows)
    ranked = sorted([r for r in rows if r["tag"] != "old"], key=lambda r: r["score"])
    best = ranked[0]
    write_report(ctx, rows, ranked)
    make_listen(ctx, ev_dir, ranked[:2], hold, songs, any(r["tag"] == "old" for r in rows))
    keep = {r["tag"] for r in ranked[:3]} | {"old"}
    for r in rows:
        if r["tag"] not in keep:
            shutil.rmtree(os.path.join(ev_dir, r["tag"]), ignore_errors=True)
    log(f"Best version: {best['tag']} (second: {ranked[1]['tag'] if len(ranked) > 1 else '—'})")
    ctx.mark("evaluate", {"rows": rows, "best": best["tag"], "ranking": [r["tag"] for r in ranked]})


def load16k(path):
    from my_utils import load_audio  # ffmpeg: reads any format, incl. FLAC without a length header
    return load_audio(path, 16000)


def mfcc_mean(path, sr=16000):
    import librosa
    y = load16k(path)
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    e = librosa.feature.rms(y=y)[0]
    keep = e > np.percentile(e, 40)
    n = min(m.shape[1], len(keep))
    return m[1:, :n][:, keep[:n]].mean(axis=1)


def flatness(path, f0_frames):
    """Noisiness of the vowels at 1–4 kHz on sung frames: a "metallic" model is flatter there."""
    import librosa
    y, sr = load16k(path), 16000
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=160)) + 1e-9
    f = librosa.fft_frequencies(sr=sr, n_fft=1024)
    band = S[(f >= 1000) & (f <= 4000)]
    fl = np.exp(np.log(band).mean(0)) / band.mean(0)
    n = min(len(fl), len(f0_frames))
    voiced = f0_frames[:n] > 50
    return float(fl[:n][voiced].mean()) if voiced.any() else float("nan")


def f0_track(cfg, path):
    from . import pitch
    return pitch._model(cfg).infer_from_audio(load16k(path), thred=0.03)


def fmtv(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


WEIGHTS = {"mcd": 0.35, "hf": 0.35, "flat": 0.15, "timbre": 0.15}


def score(rows):
    """Lower = better. z-scores across the new versions, weighted; the old model is scored on the same scale."""
    new = [r for r in rows if r["tag"] != "old"]
    for r in rows:
        r["score"] = 0.0
    wsum = 0.0
    for k, w in WEIGHTS.items():
        vals = np.array([r[k] for r in new if r[k] is not None and not math.isnan(r[k])], float)
        if len(vals) < 2:
            continue
        mu, sd = vals.mean(), max(vals.std(), 1e-6 + 0.02 * abs(vals.mean()))
        wsum += w
        for r in rows:
            v = r[k]
            if v is not None and not math.isnan(v):
                r["score"] += w * (v - mu) / sd
    for r in rows:
        r["score"] = round(r["score"] / (wsum or 1), 3)


def write_report(ctx, rows, ranked):
    lines = [f"Model {ctx.name}: the saved versions compared (lower = better in every column)", "",
             "version  score   timbre(MCD)  highs 4-16k  metal    likeness",
             "-------  ------  -----------  -----------  -------  --------"]
    order = ranked + [r for r in rows if r["tag"] == "old"]
    for r in order:
        lines.append(f"{r['tag']:>7}  {r['score']:+6.2f}  {fmtv(r['mcd']):>11}  {fmtv(r['hf']):>11}  "
                     f"{fmtv(r['flat'], 4):>7}  {fmtv(r['timbre']):>8}" + ("   ← the installed model" if r["tag"] == "old" else ""))
    lines += ["", "timbre(MCD) — how close a phrase re-sung by the model is to your real one (phrases not used in training)",
              "highs 4-16k — the difference in the highs, where metal and hiss are heard",
              "metal — noisiness of the vowels when re-singing a song",
              "likeness — timbre distance of the re-sung song to your voice"]
    with open(os.path.join(ctx.work, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(ctx.work, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    log("\n" + "\n".join(lines[:4 + len(order)]))


def make_listen(ctx, ev_dir, top, hold, songs, has_old):
    """Blind A/B of the two best versions (the answer is in answer.txt)."""
    import soundfile as sf
    ld = os.path.join(ctx.work, "listen")
    shutil.rmtree(ld, ignore_errors=True)
    os.makedirs(ld)
    pair = list(top)
    random.shuffle(pair)
    letters = ["A", "B"][:len(pair)]
    gap = np.zeros(int(0.6 * SR), np.float32)

    def concat(paths):
        parts = []
        for p in paths:
            parts += [read48(p), gap]
        return np.concatenate(parts) if parts else None
    sample = hold[:: max(1, len(hold) // 5)][:5]
    if sample:
        sf.write(os.path.join(ld, "1_phrases_original.wav"), concat(sample), SR)
        for L, r in zip(letters, pair):
            sf.write(os.path.join(ld, f"1_phrases_{L}.wav"), concat([os.path.join(ev_dir, r["tag"], os.path.basename(h)) for h in sample]), SR)
    for i, s in enumerate(songs):
        for L, r in zip(letters, pair):
            shutil.copyfile(os.path.join(ev_dir, r["tag"], f"song{i + 1}.wav"), os.path.join(ld, f"{i + 2}_song{i + 1}_{L}.wav"))
        if has_old:
            shutil.copyfile(os.path.join(ev_dir, "old", f"song{i + 1}.wav"), os.path.join(ld, f"{i + 2}_song{i + 1}_installed_model.wav"))
    with open(os.path.join(ld, "answer.txt"), "w", encoding="utf-8") as f:
        f.write("Listen first, then open.\n\n")
        for L, r in zip(letters, pair):
            f.write(f"{L} = version {r['tag']}\n")
        f.write(f"\nBy the metrics the best is {top[0]['tag']} — that one is installed.\n"
                f"If you liked the other one more: python -m coversong.train_rvc --voice {ctx.name} --install {top[-1]['tag']}\n")


# ================================================================== 8. install
def step_install(ctx, tag=None):
    ev = ctx.state.get("evaluate") or {}
    tag = tag or ev.get("best")
    row = next((r for r in ev.get("rows", []) if r["tag"] == tag), None)
    if row is None:
        ck = dict((f"e{e}", p) for e, p in ckpt_list(ctx))
        if tag not in ck:
            fail(f"no version {tag}; there are: {', '.join(ck) or 'none'}")
        row = {"tag": tag, "pth": ck[tag]}
    if tag == "old":
        fail('"old" is the installed model — it is already in place')
    mdir = os.path.join(ctx.cfg["paths"]["rvc_models"], ctx.name)
    os.makedirs(mdir, exist_ok=True)
    for f in os.listdir(mdir):
        if f.lower().endswith((".pth", ".index")):
            os.remove(os.path.join(mdir, f))
    shutil.copyfile(row["pth"], os.path.join(mdir, f"{ctx.name}.pth"))
    index = ctx.state["index"]["path"]
    shutil.copyfile(index, os.path.join(mdir, os.path.basename(index)))
    prof = dict((ctx.state["dataset"].get("pitch") or {}))
    prof.pop("seconds_per_note", None)
    if prof:
        prof["files"] = len(ctx.state["dataset"].get("files", []))
        with open(os.path.join(mdir, "voice_profile.json"), "w", encoding="utf-8") as f:
            json.dump(prof, f, ensure_ascii=False, indent=2)
    info = {"version": tag, "base": ctx.state.get("train", {}).get("base"), "train_min": ctx.state["dataset"]["train_min"],
            "metrics": {k: row.get(k) for k in ("mcd", "hf", "flat", "timbre", "score")},
            "installed": time.strftime("%Y-%m-%d %H:%M")}
    with open(os.path.join(mdir, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    log(f"Model {ctx.name} installed (version {tag}) → {mdir}")
    ctx.mark("install", {"tag": tag})


# ================================================================== main
def train(cfg, voice, from_step=None, epochs=None, force=False):
    """Everything in order (the app calls this). Returns the context."""
    ctx = Ctx(cfg, voice)
    t0 = time.time()
    if from_step:
        ctx.reset_from(from_step)
    if epochs and ctx.done("train"):
        ctx.reset_from("train")
    progress("dataset", 0.0)
    if step_dataset(ctx, force):
        log("Recordings updated — the model is trained anew")
    funcs = {"pretrain": step_pretrain, "preprocess": step_preprocess, "extract": step_extract,
             "train": lambda c: step_train(c, epochs), "index": step_index, "evaluate": step_evaluate,
             "install": step_install}
    for s in STEPS[1:]:
        if ctx.done(s):
            continue
        log(f"=== {s} ===")
        progress(s, 0.0)
        funcs[s](ctx)
        progress(s, 1.0)
    log(f"Done in {fmt_dur(time.time() - t0)}. Versions compared: {os.path.join(ctx.work, 'report.txt')}")
    log(f"A blind comparison of the two best: {os.path.join(ctx.work, 'listen')}")
    return ctx


def main():
    ap = argparse.ArgumentParser(description="Train your RVC voice model from the studio's recordings")
    ap.add_argument("--voice", required=False, help="the voice (model) name; default: voice_model in config.yaml")
    ap.add_argument("--from", dest="from_step", choices=STEPS, help="start from this step (the later ones are redone)")
    ap.add_argument("--install", metavar="eNNN", help="install the chosen version of the model")
    ap.add_argument("--epochs", type=int, help="number of epochs (default: from the amount of recordings)")
    ap.add_argument("--force", action="store_true", help="train even with few recordings")
    ap.add_argument("--config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--pitch-of", nargs=2, help=argparse.SUPPRESS)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.pitch_of:
        pitch_job(cfg, *args.pitch_of)
        return
    voice = args.voice or cfg.get("voice_model")
    if not voice:
        raise SystemExit("--voice NAME is needed")
    try:
        if args.install:
            step_install(Ctx(cfg, voice), args.install)
            return
        train(cfg, voice, args.from_step, args.epochs, args.force)
    except TrainError as e:
        raise SystemExit(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
