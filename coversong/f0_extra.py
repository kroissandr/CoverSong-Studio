"""Two more ways to read the melody for RVC, next to RMVPE and crepe (convert.f0_method / the auto-tuning):

- swiftf0: SwiftF0 (Lars Nieradzik, MIT, coversong/third_party/swift_f0), a small fast network trained on speech,
  instruments and some singing; 16 ms frames, resampled here to RVC's 10 ms. Runs on the CPU (onnxruntime).
- hpa-rmvpe: HPA-RMVPE (Vietnamese-RVC, MIT, coversong/third_party/hpa_rmvpe), RMVPE retrained with a different
  backbone; same input and output as RMVPE. Its weights (hpa-rmvpe-112000.pt) are downloaded once into rvc_models.

Both give, besides the melody, a looser voicing (what the tracker half-hears) for the note-edge smoothing in
f0fix.py.
"""
import os

import numpy as np

from . import ROOT

SR, HOP = 16000, 160
HPA_FILE = "hpa-rmvpe-112000.pt"
HPA_URL = "https://huggingface.co/AnhP/Vietnamese-RVC-Project/resolve/main/predictors/" + HPA_FILE
_swift = [None]


# ---------------------------------------------------------------- SwiftF0
def _swift_session():
    if _swift[0] is None:
        import onnxruntime as ort
        path = os.path.join(os.path.dirname(__file__), "third_party", "swift_f0", "model.onnx")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        try:
            _swift[0] = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        except Exception as e:  # older onnxruntime (1.17 in the RVC venv) reads ONNX IR 9 at most; the model uses
            if "IR version" not in str(e):  # nothing of IR 10, so the version field is lowered
                raise
            data = bytearray(open(path, "rb").read())
            assert data[0] == 0x08 and data[1] == 10
            data[1] = 9
            _swift[0] = ort.InferenceSession(bytes(data), opts, providers=["CPUExecutionProvider"])
    return _swift[0]


def swift_raw(x, fmin=50.0, fmax=1100.0):
    """SwiftF0 on 16 kHz mono audio: (pitch Hz, confidence 0..1) per 16 ms frame (frame k at k*16 ms)."""
    from .third_party.swift_f0 import HOP as SHOP, LEFT, LOOK, WINDOW
    sess = _swift_session()
    x = np.asarray(x, np.float32)
    n = max(1, len(x) // SHOP)
    pitch, conf = [], []
    for start in range(0, n, WINDOW):
        end = min(start + WINDOW, n)
        left = max(0, start - LEFT)
        w = x[left * SHOP:(end + LOOK) * SHOP] if end < n else x[left * SHOP:]
        if len(w) < SHOP:
            w = np.pad(w, (0, SHOP - len(w)))
        p, c = sess.run(["pitch", "confidence"], {"audio": w[None, :], "fmin": np.asarray(fmin, np.float32),
                                                   "fmax": np.asarray(fmax, np.float32)})
        pitch.append(p[0][start - left:end - left])
        conf.append(c[0][start - left:end - left])
    pitch, conf = np.concatenate(pitch).astype(np.float64), np.concatenate(conf).astype(np.float64)
    hops = x[:len(conf) * SHOP].reshape(-1, SHOP) if len(x) >= len(conf) * SHOP else None
    if hops is not None:
        conf[np.abs(hops).max(1) < 1e-3] = 0.0   # digital silence
    return pitch, conf


def to_rvc_frames(pitch, conf, n, frame_s=0.016, voiced=0.5, loose=0.2):
    """16 ms frames -> RVC's n frames of 10 ms: pitch interpolated on a log scale, voicing from the interpolated
    confidence. Returns (f0, f0_loose) in Hz, 0 = unvoiced."""
    t = np.arange(n) * HOP / SR
    k = np.arange(len(pitch)) * frame_s
    c = np.interp(t, k, conf)
    lp = np.interp(t, k, np.log(np.maximum(pitch, 1.0)))
    f = np.exp(lp)
    return np.where(c >= voiced, f, 0.0), np.where(c >= loose, f, 0.0)


def swift(x, n=None):
    """RVC's get_f0 for "swiftf0": (f0, f0_loose) per 10 ms of x (16 kHz)."""
    n = n or len(x) // HOP + 1
    pitch, conf = swift_raw(x)
    return to_rvc_frames(pitch, conf, n)


# ---------------------------------------------------------------- HPA-RMVPE
def hpa_weights():
    path = os.path.join(ROOT, "rvc_models", HPA_FILE)
    if os.path.exists(path):
        return path
    import urllib.request
    from .util import log
    log(f"Downloading HPA-RMVPE ({HPA_FILE}) once: {HPA_URL}")
    tmp = path + ".part"
    urllib.request.urlretrieve(HPA_URL, tmp)
    os.replace(tmp, path)
    log(f"HPA-RMVPE: {os.path.getsize(path) / 1e6:.0f} MB, saved in rvc_models")
    return path


def load_hpa(is_half, device):
    """An RMVPE object (src/rmvpe.py: its mel, decoding, infer_from_audio) running the HPA network."""
    import torch
    from rmvpe import RMVPE, MelSpectrogram
    from .third_party.hpa_rmvpe.net import E2E

    model = E2E(4, 1, (2, 2), 5, 4, 1, 16, hpa=True)
    model.load_state_dict(torch.load(hpa_weights(), map_location="cpu"))
    model.eval()
    if is_half:
        model = model.half()
    r = RMVPE.__new__(RMVPE)
    r.resample_kernel = {}
    r.is_half = is_half
    r.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    r.model = model.to(r.device)
    r.mel_extractor = MelSpectrogram(is_half, 128, 16000, 1024, 160, None, 30, 8000).to(r.device)
    r.cents_mapping = np.pad(20 * np.arange(360) + 1997.3794084376191, (4, 4))

    def mel2hidden(mel, chunk=8000):  # the hypergraph backbone needs much more memory: 80 s at a time
        import torch.nn.functional as F
        with torch.no_grad():
            n = mel.shape[-1]
            mel = F.pad(mel, (0, 32 * ((n - 1) // 32 + 1) - n), mode="reflect")
            out = [r.model(mel[..., s:s + chunk]) for s in range(0, mel.shape[-1], chunk)]
            return torch.cat(out, dim=1)[:, :n]
    r.mel2hidden = mel2hidden
    return r
