"""The covers library: every finished cover in song_output, newest first.

A song folder (song_output/<title>_<id>/) holds its covers as "<title> (<model> Ver).mp3" (+ .wav), one per voice
model it was sung with, and the original in 00_input/original.wav.
"""
import glob
import os
import re

VER = re.compile(r"^(?P<title>.*) \((?P<model>[^()]*) Ver\)\.mp3$")


def resolve(root, rel):
    """A path inside song_output from a relative one (None if it points outside)."""
    root = os.path.realpath(root)
    path = os.path.realpath(os.path.join(root, str(rel or "")))
    return path if path == root or path.startswith(root + os.sep) else None


def scan(root):
    songs = []
    if not os.path.isdir(root):
        return songs
    for d in os.listdir(root):
        sd = os.path.join(root, d)
        if not os.path.isdir(sd):
            continue
        covers = []
        for mp3 in glob.glob(os.path.join(glob.escape(sd), "* Ver).mp3")):
            m = VER.match(os.path.basename(mp3))
            if not m:
                continue
            wav = mp3[:-4] + ".wav"
            st = os.stat(mp3)
            covers.append({"model": m.group("model"), "title": m.group("title"), "time": st.st_mtime,
                           "size_mb": round(st.st_size / 1e6, 1),
                           "mp3": os.path.relpath(mp3, root).replace(os.sep, "/"),
                           "wav": os.path.relpath(wav, root).replace(os.sep, "/") if os.path.exists(wav) else None})
        if not covers:
            continue
        covers.sort(key=lambda c: -c["time"])
        orig = os.path.join(sd, "00_input", "original.wav")
        songs.append({"title": covers[0]["title"], "folder": d, "time": covers[0]["time"], "covers": covers,
                      "original": os.path.relpath(orig, root).replace(os.sep, "/") if os.path.exists(orig) else None})
    songs.sort(key=lambda s: -s["time"])
    return songs
