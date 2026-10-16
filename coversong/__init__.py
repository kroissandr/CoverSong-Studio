"""CoverSong Studio: AI covers in your own voice, a vocal studio to record it and training of the voice model."""
import os
import sys

__version__ = "1.0.0"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    # legacy AICoverGen modules (mdx, rvc, infer_pack, ...) import each other by bare name
    sys.path.insert(0, SRC)
_FFMPEG = os.path.join(ROOT, "tools", "ffmpeg", "bin")  # put there by start.bat
if os.path.isdir(_FFMPEG) and _FFMPEG not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _FFMPEG + os.pathsep + os.environ.get("PATH", "")
