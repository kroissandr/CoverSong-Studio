"""Stage 0: take a local file or a YouTube link -> 00_input/original.wav (44.1 kHz stereo float)."""
import glob
import os
from urllib.parse import urlparse, parse_qs

from ..cache import file_hash
from ..util import ffmpeg_to_wav, log, slugify


def youtube_id(url):
    q = urlparse(url)
    if q.hostname == "youtu.be":
        return q.path[1:]
    if q.hostname in {"www.youtube.com", "youtube.com", "music.youtube.com", "m.youtube.com"}:
        if q.path == "/watch":
            return parse_qs(q.query).get("v", [None])[0]
        for prefix in ("/watch/", "/embed/", "/v/", "/shorts/"):
            if q.path.startswith(prefix):
                return q.path[len(prefix):].split("/")[0]
    return None


def identify(song_input):
    """Return (kind, song_id, title) without doing heavy work."""
    s = song_input.strip().strip('"')
    if urlparse(s).scheme in ("http", "https"):
        vid = youtube_id(s)
        if not vid:
            raise ValueError(f"Does not look like a YouTube link: {s}")
        return "yt", vid, None
    if not os.path.exists(s):
        raise FileNotFoundError(f"File not found: {s}")
    return "local", file_hash(s), slugify(os.path.splitext(os.path.basename(s))[0])


def run(song_input, kind, stage, excerpt=None):
    s = song_input.strip().strip('"')
    out = stage.path("original.wav")
    cut = {"start": excerpt[0], "duration": excerpt[1]} if excerpt else {}
    if kind == "local":
        log(f"Reading the file: {s}")
        ffmpeg_to_wav(s, out, **cut)
        title = slugify(os.path.splitext(os.path.basename(s))[0])
        src = s
    else:
        import yt_dlp
        log(f"Downloading from YouTube: {s}")
        opts = {
            "format": "bestaudio/best",
            "outtmpl": stage.path("download.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(s, download=True)
        files = [f for f in glob.glob(stage.path("download.*"))]
        if not files:
            raise RuntimeError("yt-dlp downloaded nothing (yt-dlp may need an update: it is updated at the next start)")
        ffmpeg_to_wav(files[0], out, **cut)
        for f in files:
            os.remove(f)
        title = slugify(info.get("title") or "youtube")
        src = s
    return {"original": out}, {"title": title, "source": src}
