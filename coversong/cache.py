"""Per-song, per-stage cache.

song_output/<song_dir>/<NN_stage>/manifest.json holds a key = hash(stage params + upstream keys).
A stage is skipped when its key matches and all its outputs exist.
"""
import datetime as _dt
import hashlib
import json
import os

STAGES = ["input", "separate", "convert", "clean", "fx", "mix"]
STAGE_NUM = {"input": 0, "separate": 1, "convert": 2, "clean": 3, "fx": 4, "mix": 5}


def file_hash(path, n=11):
    h = hashlib.blake2b()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:n]


def params_key(params: dict, upstream=()):
    blob = json.dumps({"p": params, "up": list(upstream)}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


class Stage:
    def __init__(self, song_dir, name):
        self.name = name
        self.dir = os.path.join(song_dir, f"{STAGE_NUM[name]:02d}_{name}")
        self.manifest_path = os.path.join(self.dir, "manifest.json")

    def path(self, filename):
        return os.path.join(self.dir, filename)

    def manifest(self):
        if not os.path.exists(self.manifest_path):
            return None
        with open(self.manifest_path, encoding="utf-8") as f:
            return json.load(f)

    def is_done(self, key):
        m = self.manifest()
        if not m or m.get("key") != key:
            return False
        return all(os.path.exists(self.path(fn)) for fn in m.get("outputs", {}).values())

    def outputs(self):
        m = self.manifest() or {}
        return {k: self.path(v) for k, v in m.get("outputs", {}).items()}

    def key(self):
        m = self.manifest() or {}
        return m.get("key")

    def begin(self):
        os.makedirs(self.dir, exist_ok=True)
        if os.path.exists(self.manifest_path):
            os.remove(self.manifest_path)

    def finish(self, key, params, outputs: dict, info=None):
        rel = {k: os.path.basename(v) for k, v in outputs.items()}
        data = {
            "stage": self.name,
            "key": key,
            "params": params,
            "outputs": rel,
            "info": info or {},
            "finished": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
