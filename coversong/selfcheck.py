"""python -m coversong.selfcheck [--quick] — checks the environment (start.bat runs the quick check after installing).

--quick: only that the libraries import (exit code 1 if one does not).
"""
import importlib
import os
import shutil
import sys

from .config import load_config

LIBS = ["numpy", "scipy", "torch", "torchaudio", "transformers", "faiss", "librosa", "soundfile", "yaml", "gradio",
        "fastapi", "uvicorn", "audio_separator", "whisper", "yt_dlp", "torchcrepe", "onnxruntime", "sklearn",
        "parselmouth", "pyworld", "noisereduce"]
OPTIONAL = {"parselmouth", "pyworld", "noisereduce"}
ok = True


def check(name, cond, detail="", optional=False):
    global ok
    if not optional:
        ok &= bool(cond)
    tag = "OK  " if cond else ("info" if optional else "ERR ")
    print(f"[{tag}] {name}{(': ' + str(detail)) if detail else ''}", flush=True)


def libs():
    for name in LIBS:
        try:
            m = importlib.import_module(name)
            check(name, True, getattr(m, "__version__", ""), optional=name in OPTIONAL)
        except Exception as e:  # noqa: BLE001
            check(name, False, f"{e.__class__.__name__}: {e}", optional=name in OPTIONAL)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    print(f"Python {sys.version.split()[0]} ({sys.executable})")
    check("Python 3.10+", sys.version_info[:2] >= (3, 10), sys.version.split()[0])
    libs()
    if "--quick" in argv:
        print("\nRESULT:", "libraries OK" if ok else "some libraries are missing (see ERR above)")
        return 0 if ok else 1
    cfg = load_config()
    try:
        import torch
        cuda = torch.cuda.is_available()
        check("GPU (CUDA)", cuda, torch.cuda.get_device_name(0) if cuda else
              "not available — everything works on the CPU, but much slower", optional=True)
    except Exception as e:  # noqa: BLE001
        check("PyTorch", False, e)
    check("ffmpeg", shutil.which("ffmpeg"), shutil.which("ffmpeg") or "not found in PATH (start.bat puts it into tools/)")
    from . import assets
    for name in assets.FILES:
        p = os.path.join(cfg["paths"]["rvc_models"], *name.split("/"))
        check(f"rvc_models/{name}", os.path.exists(p), "" if os.path.exists(p) else "downloaded on first use",
              optional=True)
    from .ui import list_models
    models = list_models()
    check("voice models", bool(models), ", ".join(models) if models else
          "none yet — train one in the Vocal Studio or put a model folder into rvc_models", optional=True)
    print("\nRESULT:", "all good" if ok else "there are errors (see ERR above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
