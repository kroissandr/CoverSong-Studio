"""Runs in its own process: Whisper word timestamps for vocal tracks.

    python -m coversong.asr_worker in.wav out.json models_dir
    python -m coversong.asr_worker --job job.json models_dir      job = [{"in", "out", "language"}, ...]
"""
import json
import sys
import warnings

# without the CUDA toolkit Whisper times words with a slower (equally exact) kernel and warns about it on every song
warnings.filterwarnings("ignore", message="Failed to launch Triton kernels")


def transcribe(model, src, dst, language=None, fp16=True):
    res = model.transcribe(src, word_timestamps=True, condition_on_previous_text=False, language=language,
                           fp16=fp16, verbose=False, no_speech_threshold=0.5)
    words = [{"start": w["start"], "end": w["end"], "word": w["word"].strip(), "p": w.get("probability", 0)}
             for seg in res.get("segments", []) for w in seg.get("words", [])]
    with open(dst, "w", encoding="utf-8") as f:
        json.dump({"language": res.get("language"), "words": words}, f, ensure_ascii=False, indent=1)
    print(f"[asr] language {res.get('language')}, words: {len(words)}", flush=True)


def main(argv):
    import torch
    import whisper

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if argv[0] == "--job":
        with open(argv[1], encoding="utf-8") as f:
            jobs = json.load(f)
        models_dir = argv[2]
    else:
        jobs = [{"in": argv[0], "out": argv[1], "language": None}]
        models_dir = argv[2]
    model = whisper.load_model("turbo", device=dev, download_root=models_dir)
    for j in jobs:
        transcribe(model, j["in"], j["out"], j.get("language"), fp16=dev == "cuda")


if __name__ == "__main__":
    main(sys.argv[1:])
