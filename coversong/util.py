import os
import re
import subprocess
import time


LOG_HOOKS = []  # the UI subscribes here to show progress


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    for hook in list(LOG_HOOKS):
        try:
            hook(line)
        except Exception:
            pass


def worker_python():
    """The Python that runs the heavy workers (separation, Whisper) in their own process — the same environment, so
    their memory is freed when they finish. Returns (python, env)."""
    import sys
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    return sys.executable, env


def slugify(text, maxlen=60):
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text).strip().strip(".")
    text = re.sub(r"\s+", " ", text)
    return text[:maxlen] or "song"


def ffmpeg_to_wav(src, dst, sr=44100, channels=2, start=None, duration=None):
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", src]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-ac", str(channels), "-ar", str(sr), "-c:a", "pcm_f32le", dst]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError(f"ffmpeg could not read {src}:\n{r.stderr}")
    return dst


class Timer:
    def __init__(self, what):
        self.what = what

    def __enter__(self):
        self.t = time.time()
        log(f"▶ {self.what}")
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            log(f"✔ {self.what} — {time.time() - self.t:.1f} s")


def parse_excerpt(text):
    """'40:30' -> (40.0, 30.0) = start 40 s, duration 30 s."""
    if not text:
        return None
    a, b = text.split(":", 1)
    return float(a), float(b)
