"""Vocal studio: choose the voice (the model you record for), record takes in the browser, get an instant grade and
plain-language tips, keep the good ones, train the model. Served by the app at /studio.

API (all local): /api/voices, /api/plan, /api/analyze, /api/save, /api/discard, /api/train, /api/open
"""
import json
import os
import re
import threading
import time
import uuid

import numpy as np

from ..config import load_config
from ..i18n import lang_of
from ..util import LOG_HOOKS, log, slugify
from . import analyze, library, plan

STATIC = os.path.join(os.path.dirname(__file__), "static")
SR_SAVE = 48000
MAX_TAKE_S = 900
_lock = threading.Lock()
TRAIN = {"running": False, "voice": None, "step": None, "frac": 0.0, "text": "", "log": [], "error": None,
         "done": None, "started": None}


def make_app():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="CoverSong Studio — vocal studio", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def cfg_now():
        return load_config()

    def voice_of(cfg, name, must_exist=True):
        v = plan.safe_name(name)
        if not v:
            raise HTTPException(400, "no voice chosen")
        if must_exist and not os.path.isdir(plan.voice_dir(cfg, v)) and not plan.has_model(cfg, v):
            raise HTTPException(404, f"no voice {v}")
        return v

    def task_of(cfg, voice, task_id, lang="en"):
        t = next((t for t in plan.build(cfg, voice, lang)["tasks"] if t["id"] == task_id), None)
        if t is None:
            raise HTTPException(404, "no such task")
        return t

    def page(name):
        """A page whose css/js links carry a version mark, so the browser never keeps an old copy after an update."""
        from fastapi.responses import HTMLResponse
        mark = str(int(max(os.path.getmtime(os.path.join(STATIC, f)) for f in os.listdir(STATIC))))
        with open(os.path.join(STATIC, name), encoding="utf-8") as f:
            html = f.read()
        html = re.sub(r'(static/[\w.-]+\.(?:css|js))"', r'\1?v=' + mark + '"', html)
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @app.get("/")
    def index():
        return page("index.html")

    @app.get("/api/voices")
    def get_voices(lang: str = "en"):
        cfg = cfg_now()
        lang = lang_of(lang)
        return {"voices": plan.list_voices(cfg),
                "types": [{"id": k, "name": plan.VOICE_TYPE_NAMES[lang][k]} for k in plan.VOICE_TYPES],
                "default": cfg.get("voice_model") or None}

    @app.post("/api/voices")
    def new_voice(name: str = Form(...), type: str = Form("tenor")):  # noqa: A002
        cfg = cfg_now()
        v = plan.safe_name(name)
        if not v:
            raise HTTPException(400, "Use Latin letters, digits, - and _ in the name")
        plan.create_voice(cfg, v, type)
        log(f"Studio: voice {v} ({type}) — recordings go to {plan.rec_dir(cfg, v)}")
        return {"ok": True, "name": v, "voices": plan.list_voices(cfg)}

    @app.get("/api/plan")
    def get_plan(voice: str, lang: str = "en"):
        cfg = cfg_now()
        v = voice_of(cfg, voice)
        os.makedirs(plan.rec_dir(cfg, v), exist_ok=True)
        clean_tmp(plan.tmp_dir(cfg, v))
        p = plan.build(cfg, v, lang)
        return JSONResponse(dict(p, progress=plan.load_progress(cfg, v), folder=plan.rec_dir(cfg, v),
                                 recorded_min=round(plan.recorded_seconds(cfg, v) / 60, 1),
                                 model=plan.has_model(cfg, v), train=train_status(v)))

    @app.post("/api/analyze")
    def analyze_take(voice: str = Form(...), task_id: str = Form(...), lang: str = Form("en"),
                     audio: UploadFile = File(...)):
        import soundfile as sf
        from ..dsp import to_sr
        cfg = cfg_now()
        v = voice_of(cfg, voice)
        task = task_of(cfg, v, task_id, lang)
        tmp = plan.tmp_dir(cfg, v)
        os.makedirs(tmp, exist_ok=True)
        take = uuid.uuid4().hex[:12]
        raw = os.path.join(tmp, f"{take}_raw.wav")
        with open(raw, "wb") as f:
            f.write(audio.file.read())
        y, sr = sf.read(raw, dtype="float32", always_2d=True)
        os.remove(raw)
        y = y[: int(MAX_TAKE_S * sr)]
        x = to_sr(y.mean(1), sr, SR_SAVE).astype(np.float32)
        path = os.path.join(tmp, f"{take}.wav")
        sf.write(path, x, SR_SAVE, subtype="FLOAT")
        with _lock:
            f0 = f0_track(cfg, x)
        res = analyze.grade(x.astype(np.float64), SR_SAVE, f0, task, lang)
        hist = res.pop("hist", {}) or {}
        with open(os.path.join(tmp, f"{take}.json"), "w", encoding="utf-8") as f:
            json.dump({"hist": {str(k): s for k, s in hist.items()}, "seconds": len(x) / SR_SAVE}, f)
        res.update(take=take, verdict=analyze.verdict(res["score"], lang))
        res["tips"] = [{"kind": k, "text": t} for k, t in res["tips"]]
        return JSONResponse(to_json(res))

    @app.post("/api/save")
    def save_take(voice: str = Form(...), task_id: str = Form(...), take: str = Form(...), score: float = Form(0),
                  label: str = Form("")):
        import soundfile as sf
        cfg = cfg_now()
        v = voice_of(cfg, voice)
        task_of(cfg, v, task_id)
        tmp = plan.tmp_dir(cfg, v)
        take = os.path.basename(take)
        src = os.path.join(tmp, f"{take}.wav")
        if not os.path.exists(src):
            raise HTTPException(404, "take not found")
        x, sr = sf.read(src, dtype="float32")
        x = trim(x, sr)
        tag = slugify(label, 40).replace(" ", "_") if label else ""
        tag = "".join(ch for ch in tag if ch.isalnum() or ch in "_-")
        name = f"studio_{task_id}{'_' + tag if tag else ''}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        d = plan.rec_dir(cfg, v)
        os.makedirs(d, exist_ok=True)
        sf.write(os.path.join(d, name), x, sr, subtype="PCM_24")
        meta = {}
        try:
            with open(os.path.join(tmp, f"{take}.json"), encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            pass
        for ext in (".wav", ".json"):
            if os.path.exists(os.path.join(tmp, take + ext)):
                os.remove(os.path.join(tmp, take + ext))
        plan.add_take(cfg, v, name, len(x) / sr, {int(k): s for k, s in (meta.get("hist") or {}).items()})
        prog = plan.load_progress(cfg, v)
        p = prog.get(task_id, {"takes": 0, "best": 0})
        p["takes"] += 1
        p["best"] = max(p["best"], float(score))
        prog[task_id] = p
        plan.save_progress(cfg, v, prog)
        log(f"Studio: saved {v}/{name} ({score:.0f} points)")
        return {"ok": True, "file": name, "progress": prog,
                "recorded_min": round(plan.recorded_seconds(cfg, v) / 60, 1)}

    @app.post("/api/discard")
    def discard(voice: str = Form(...), take: str = Form(...)):
        cfg = cfg_now()
        v = voice_of(cfg, voice)
        for ext in (".wav", ".json"):
            p = os.path.join(plan.tmp_dir(cfg, v), os.path.basename(take) + ext)
            if os.path.exists(p):
                os.remove(p)
        return {"ok": True}

    @app.post("/api/open")
    def open_folder(voice: str = Form(...)):
        cfg = cfg_now()
        d = plan.rec_dir(cfg, voice_of(cfg, voice))
        os.makedirs(d, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(d)  # noqa: S606  Windows Explorer
        return {"ok": True, "folder": d}

    @app.get("/api/train")
    def get_train(voice: str = ""):
        return train_status(plan.safe_name(voice) or None)

    @app.post("/api/train")
    def start_train(voice: str = Form(...), force: bool = Form(False)):
        cfg = cfg_now()
        v = voice_of(cfg, voice)
        if TRAIN["running"]:
            raise HTTPException(409, f"already training {TRAIN['voice']}")
        threading.Thread(target=run_training, args=(cfg, v, force), daemon=True).start()
        time.sleep(0.2)
        return train_status(v)

    # ------------------------------------------------------------ the covers library
    @app.get("/library")
    def library_page():
        return page("library.html")

    @app.get("/api/library")
    def get_library():
        return {"songs": library.scan(cfg_now()["paths"]["output"])}

    @app.get("/api/media")
    def media(p: str, dl: bool = False):
        path = library.resolve(cfg_now()["paths"]["output"], p)
        if path is None or not os.path.isfile(path):
            raise HTTPException(404, "file not found")
        return FileResponse(path, filename=os.path.basename(path) if dl else None,
                            content_disposition_type="attachment" if dl else "inline")

    @app.post("/api/library/open")
    def library_open(p: str = Form("")):
        root = cfg_now()["paths"]["output"]
        path = library.resolve(root, p) if p else root
        if path is None or not os.path.isdir(path):
            raise HTTPException(404, "folder not found")
        if hasattr(os, "startfile"):
            os.startfile(path)  # noqa: S606  Windows Explorer
        return {"ok": True}

    @app.get("/{rest:path}")
    def fallback(rest: str):
        return RedirectResponse("./")

    return app


def clean_tmp(d, hours=6):
    """Takes that were neither saved nor discarded (the page was closed) are dropped after a while."""
    if not os.path.isdir(d):
        return
    for f in os.listdir(d):
        p = os.path.join(d, f)
        try:
            if time.time() - os.path.getmtime(p) > hours * 3600:
                os.remove(p)
        except OSError:
            pass


def train_status(voice=None):
    st = {k: TRAIN[k] for k in ("running", "voice", "step", "frac", "text", "error", "done", "started")}
    st["log"] = TRAIN["log"][-40:]
    st["mine"] = voice is not None and TRAIN["voice"] == voice
    from .. import train_rvc
    st["steps"] = train_rvc.STEPS
    st["min_minutes"] = train_rvc.MIN_TRAIN_MIN
    st["good_minutes"] = train_rvc.GOOD_TRAIN_MIN
    return st


def run_training(cfg, voice, force=False):
    from .. import train_rvc

    def hook(line):
        TRAIN["log"].append(line)
        del TRAIN["log"][:-400]

    def prog(step, frac, text=""):
        TRAIN.update(step=step, frac=float(frac), text=text or "")

    TRAIN.update(running=True, voice=voice, step="dataset", frac=0.0, text="", log=[], error=None, done=None,
                 started=time.time())
    LOG_HOOKS.append(hook)
    train_rvc.PROGRESS = prog
    try:
        log(f"Studio: training the model {voice}")
        train_rvc.train(cfg, voice, force=force)
        TRAIN["done"] = time.time()
    except Exception as e:  # noqa: BLE001  shown in the studio; the traceback goes to the log
        import traceback
        traceback.print_exc()
        TRAIN["error"] = str(e) or e.__class__.__name__
    finally:
        train_rvc.PROGRESS = None
        LOG_HOOKS.remove(hook)
        TRAIN["running"] = False


def f0_track(cfg, x48, chunk_s=30):
    """RMVPE pitch (Hz per 10 ms) of a take, in 30 s pieces (a whole song does not fit the GPU at once)."""
    from ..dsp import to_sr
    from ..pitch import _model
    x16 = to_sr(x48.astype(np.float32), SR_SAVE, 16000)
    m = _model(cfg)
    hop, step = 160, 16000 * chunk_s
    parts = []
    for i in range(0, len(x16), step):
        f0 = m.infer_from_audio(x16[i:i + step], thred=0.03)
        parts.append(np.asarray(f0)[: (min(step, len(x16) - i)) // hop])
    return np.concatenate(parts) if parts else np.zeros(0)


def trim(x, sr, pad=0.2):
    """Cut silence at both ends (keeps 0.2 s), so the dataset holds singing, not room noise."""
    hop = sr // 100
    n = len(x) // hop
    if n < 2:
        return x
    db = 10 * np.log10((x[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    act = np.flatnonzero(db > db.max() - 40)
    if not len(act):
        return x
    a = max(0, act[0] * hop - int(pad * sr))
    b = min(len(x), (act[-1] + 1) * hop + int(pad * sr))
    return x[a:b]


def to_json(o):
    if isinstance(o, dict):
        return {str(k): to_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_json(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        o = o.item()
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o
