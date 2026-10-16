"""Stage 1: separation — one fixed chain, chosen by comparison on 2026-10-10.

  1. Mel-RoFormer vocals (Kim)            original  -> vocals + instrumental
  2. Mel-RoFormer karaoke (aufr33/viperx) vocals    -> lead_wet + backing
  3. Mel-RoFormer de-reverb (anvuew)      lead_wet  -> lead_dry + lead_reverb
  (the backing is sung as separated: de-reverbing it was tried in 0.19.0 and removed — the model takes a wide
   stereo backing's doubles for reverb, and most of the backing vanished in HOT and Bite Me)

Runs in its own process through coversong.sep_worker. Outputs, 44.1 kHz stereo:
  vocals, instrumental, lead_wet (reference for stage 4), backing (RVC input), lead_dry (RVC input), lead_reverb
"""
import json
import os
import subprocess

from .. import ROOT
from ..util import log

VOCALS_MODEL = "vocals_mel_band_roformer.ckpt"
KARAOKE_MODEL = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
DEREVERB_MODEL = "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"
CHAIN = [VOCALS_MODEL, KARAOKE_MODEL, DEREVERB_MODEL]

ROLES = {
    "vocals": {"vocals": ["vocals"], "instrumental": ["instrumental", "other", "no vocals"]},
    # karaoke models: "vocals" = lead, the rest = backing vocals (input is vocals only)
    "karaoke": {"lead": ["vocals", "lead", "karaoke", "lead vocals"],
                "backing": ["instrumental", "other", "backing", "back", "no vocals", "backing vocals"]},
    "dereverb": {"dry": ["noreverb", "no reverb", "dry", "no echo", "noecho"],
                 "reverb": ["reverb", "no dry", "echo", "other", "instrumental"]},
}


def run_job(job, job_path):
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)
    from ..util import worker_python
    py, env = worker_python()
    proc = subprocess.run([py, "-u", "-m", "coversong.sep_worker", job_path], cwd=ROOT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Separation failed (code {proc.returncode}), see the log above")


def run(cfg, original, stage):
    out = {k: stage.path(f"{k}.wav") for k in ("vocals", "instrumental", "lead_wet", "backing", "lead_dry", "lead_reverb")}
    job = {
        "model_dir": cfg["paths"]["sep_models"],
        "tmp_dir": stage.path("_tmp"),
        "use_autocast": True,
        "steps": [
            {"model": VOCALS_MODEL, "input": original, "roles": ROLES["vocals"],
             "targets": {"vocals": out["vocals"], "instrumental": out["instrumental"]}},
            {"model": KARAOKE_MODEL, "input": out["vocals"], "roles": ROLES["karaoke"],
             "targets": {"lead": out["lead_wet"], "backing": out["backing"]}},
            {"model": DEREVERB_MODEL, "input": out["lead_wet"], "roles": ROLES["dereverb"],
             "targets": {"dry": out["lead_dry"], "reverb": out["lead_reverb"]}},
        ],
        # every model step re-normalises its input: put all stems back on the original's scale
        "restore": [
            {"target": original, "parts": [out["instrumental"], out["vocals"]]},
            {"target": out["vocals"], "parts": [out["lead_wet"], out["backing"]]},
            {"target": out["lead_wet"], "parts": [out["lead_dry"], out["lead_reverb"]]},
        ],
    }
    log("Separation: vocals/instrumental → lead/backing → de-reverb (Mel-RoFormer)")
    run_job(job, stage.path("sep_job.json"))
    os.remove(stage.path("sep_job.json"))
    return out, {"chain": CHAIN, "levels": "restored"}


def backing_src(sep):
    """What RVC sings the backing from: the backing as separated (song folders separated by 0.19.0 also have a
    de-reverbed backing_dry; it is not used)."""
    return sep.get("backing")
