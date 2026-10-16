"""Command line entry point.

    python -m coversong.cli "D:\\Music\\song.mp3" [--model my_voice] [--pitch -1] [--from-stage convert]
                         [--set convert.index_rate=0.7 --set mix.output_format=wav]

Runs itself in a child process so that everything (including native-library messages) is copied to logs/.
"""
import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys

from . import ROOT
from .cache import STAGES


def build_parser():
    p = argparse.ArgumentParser(prog="coversong", description="CoverSong Studio: a cover in your voice, stage by stage")
    p.add_argument("song", help="audio file path or YouTube link")
    p.add_argument("--model", help="voice model folder in rvc_models (default from config.yaml)")
    p.add_argument("--pitch", help="vocal shift in octaves: auto, -1, 0, 1 (default auto)")
    p.add_argument("--from-stage", choices=STAGES, help="recompute from this stage on")
    p.add_argument("--until", choices=STAGES, default="mix", help="stop after this stage")
    p.add_argument("--excerpt", metavar="START:DUR", help="only an excerpt, e.g. 40:30 = 30 s from 0:40")
    p.add_argument("--config", help="another settings file instead of config.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a setting")
    return p


def child_main(argv):
    from .config import load_config
    from .pipeline import run_pipeline
    from .util import parse_excerpt

    args = build_parser().parse_args(argv)
    overrides = list(args.set)
    if args.pitch is not None:
        overrides.append(f"convert.pitch_octaves={args.pitch}")
    cfg = load_config(args.config, overrides)
    run_pipeline(cfg, args.song, voice_model=args.model, from_stage=args.from_stage, until_stage=args.until,
                 excerpt=parse_excerpt(args.excerpt))


def tee_main(argv, module="coversong.cli", prefix="run"):
    logs = os.path.join(ROOT, "logs")
    os.makedirs(logs, exist_ok=True)
    log_path = os.path.join(logs, f"{prefix}_{dt.datetime.now():%Y%m%d_%H%M%S}.txt")
    env = dict(os.environ, COVERSONG_CHILD="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-u", "-m", module, *argv]
    with open(log_path, "wb") as logf:
        logf.write(("$ " + " ".join(cmd) + "\n").encode("utf-8"))
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = sys.stdout.buffer
        while True:
            chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(1)
            if not chunk:
                break
            logf.write(chunk)
            logf.flush()
            try:
                out.write(chunk)
                out.flush()
            except Exception:
                pass
        code = proc.wait()
        logf.write(f"\n[exit code {code}]\n".encode("utf-8"))
    shutil.copyfile(log_path, os.path.join(logs, "last_run.txt"))
    print(f"\nLog: {log_path}")
    return code


def main():
    argv = sys.argv[1:]
    if os.environ.get("COVERSONG_CHILD") == "1":
        child_main(argv)
        return 0
    if not argv or argv[0] in ("-h", "--help"):
        build_parser().print_help()
        return 0
    return tee_main(argv)


if __name__ == "__main__":
    sys.exit(main())
