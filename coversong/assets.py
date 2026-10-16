"""Model files the app needs, downloaded once on first use into rvc_models/ (or models/ for the separator and
Whisper, which download their own):

  - rmvpe.pt — the RMVPE melody tracker (RVC project, MIT): lj1995/VoiceConversionWebUI on Hugging Face
  - contentvec/ — the HuBERT content encoder every RVC v2 model was trained with, in the Hugging Face transformers
    format as published by Applio (MIT): IAHispano/Applio on Hugging Face. The same network as RVC's hubert_base.pt,
    without the fairseq library (which does not install on Windows without a compiler).
"""
import os
import urllib.request

from .util import log

HF = "https://huggingface.co"
FILES = {
    "rmvpe.pt": f"{HF}/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
    "contentvec/config.json": f"{HF}/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/config.json",
    "contentvec/pytorch_model.bin": f"{HF}/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/pytorch_model.bin",
}


def download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    log(f"Downloading {os.path.basename(path)} (once): {url}")
    last = [0]

    def hook(blocks, bs, total):
        done = blocks * bs
        if total > 0 and done - last[0] >= max(total // 10, 1):
            last[0] = done
            log(f"  {os.path.basename(path)}: {min(100, done * 100 // total)}% of {total / 1e6:.0f} MB")
    urllib.request.urlretrieve(url, tmp, hook)
    os.replace(tmp, path)


def ensure(cfg, name):
    """Path of a needed file (downloaded first if missing)."""
    path = os.path.join(cfg["paths"]["rvc_models"], *name.split("/"))
    if not os.path.exists(path):
        download(FILES[name], path)
    return path


def contentvec_dir(cfg):
    for name in ("contentvec/config.json", "contentvec/pytorch_model.bin"):
        ensure(cfg, name)
    return os.path.join(cfg["paths"]["rvc_models"], "contentvec")


def ensure_all(cfg):
    for name in FILES:
        ensure(cfg, name)
