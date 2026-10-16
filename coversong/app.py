"""CoverSong Studio — the whole app on one local address:

    /         covers (English)          /ru/      covers (Russian)
    /studio/  vocal studio: record your voice, train your model

    python -m coversong            (start.bat does this)
    python -m coversong --studio   (opens the studio page)

The console and logs/ are in English; the interface language is chosen in the app.
"""
import os
import socket
import sys
import threading
import webbrowser

from . import ROOT, __version__
from .config import load_config
from .util import log


def free_port(start):
    for p in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"no free port in {start}–{start + 19}")


def gpu_note():
    """Say which device the work runs on (and what to do when an NVIDIA GPU is there but not usable)."""
    import shutil
    try:
        import torch
        if torch.cuda.is_available():
            log(f"GPU: {torch.cuda.get_device_name(0)} (PyTorch {torch.__version__})")
            return
        if shutil.which("nvidia-smi"):
            log("WARNING: there is an NVIDIA GPU, but PyTorch cannot use it — update the NVIDIA driver "
                "(nvidia.com/drivers) and restart. Until then everything runs on the CPU, slowly.")
        else:
            log("No NVIDIA GPU: everything runs on the CPU — covers take several times longer, training is very slow.")
    except Exception as e:  # noqa: BLE001
        log(f"PyTorch could not start: {e}")


# Gradio translates its own texts ("Drop file here", "Click to upload"…) into the browser's language. The page's
# language is chosen by its address instead (/ English, /ru/ Russian), so the browser language is set for the page
# before Gradio starts.
LOCALE_JS = ("<script>(function(l){try{Object.defineProperty(navigator,'language',{get:function(){return l}});"
             "Object.defineProperty(navigator,'languages',{get:function(){return [l]}});}catch(e){}})('%s');</script>")


class ForceLocale:
    """ASGI middleware: puts LOCALE_JS at the top of the covers pages' HTML; everything else (the queue's streams,
    files, the studio) passes through untouched."""

    def __init__(self, app):
        import re
        self.app = app
        self.head = re.compile(rb"<head[^>]*>", re.I)

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] != "http" or scope.get("method") != "GET" or path.startswith("/studio"):
            return await self.app(scope, receive, send)
        lang = "ru" if path == "/ru" or path.startswith("/ru/") else "en"
        state = {"html": False, "start": None, "body": []}

        async def wrapped(msg):
            if msg["type"] == "http.response.start":
                h = {k.lower(): v for k, v in msg.get("headers", [])}
                if b"text/html" in h.get(b"content-type", b"") and b"content-encoding" not in h:
                    state["html"], state["start"] = True, msg
                    return
                return await send(msg)
            if msg["type"] == "http.response.body" and state["html"]:
                state["body"].append(msg.get("body", b""))
                if msg.get("more_body"):
                    return
                body = b"".join(state["body"])
                m = self.head.search(body)
                if m:
                    body = body[:m.end()] + (LOCALE_JS % lang).encode() + body[m.end():]
                start = state["start"]
                headers = [(k, v) for k, v in start.get("headers", []) if k.lower() != b"content-length"]
                headers.append((b"content-length", str(len(body)).encode()))
                await send(dict(start, headers=headers))
                return await send({"type": "http.response.body", "body": body, "more_body": False})
            return await send(msg)

        return await self.app(scope, receive, wrapped)


def make_app():
    import gradio as gr
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse

    from . import ui
    from .studio import server as studio

    cfg = load_config()
    for k in ("output", "logs", "rvc_models", "voices"):
        os.makedirs(cfg["paths"][k], exist_ok=True)
    app = FastAPI(title="CoverSong Studio", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/studio", include_in_schema=False)
    def studio_slash():
        return RedirectResponse("/studio/")

    @app.get("/ru", include_in_schema=False)
    def ru_slash():
        return RedirectResponse("/ru/")

    app.add_middleware(ForceLocale)
    app.mount("/studio", studio.make_app())
    allowed = [cfg["paths"]["output"], cfg["paths"]["voices"], os.path.join(ROOT, "logs")]
    common = dict(theme=ui.theme(), css=ui.css(), allowed_paths=allowed, footer_links=[], ssr_mode=False,
                  show_error=True)
    # the more specific mount first: "/" would take every path
    gr.mount_gradio_app(app, ui.build("ru"), path="/ru", **common)
    gr.mount_gradio_app(app, ui.build("en"), path="/", **common)
    return app


def main(argv=None):
    import uvicorn
    argv = sys.argv[1:] if argv is None else argv
    cfg = load_config()
    port = free_port(int((cfg.get("ui") or {}).get("port", 7861)))
    lang = str((cfg.get("ui") or {}).get("language", "en")).lower()
    page = "studio/" + ("?lang=ru" if lang == "ru" else "") if "--studio" in argv else ("ru/" if lang == "ru" else "")
    url = f"http://127.0.0.1:{port}/"
    log(f"CoverSong Studio {__version__}")
    gpu_note()
    log("Loading the interface…")
    app = make_app()
    log(f"Open in the browser: {url}  (covers: {url} · {url}ru/ · vocal studio: {url}studio/)")
    log("Keep this window open while you work; close it to stop the app.")
    if "--no-browser" not in argv:
        threading.Timer(2.0, lambda: webbrowser.open(url + page)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
