# Third-party notices

CoverSong Studio is MIT-licensed (see [LICENSE](LICENSE)). It is built on, includes or downloads the work below.
Thank you to all of these authors. Licences are given as published by each project; where a component is only
downloaded at run time, its own terms apply to it.

## Code included in this repository

| Component | Where | Licence | Source |
|---|---|---|---|
| AICoverGen — the starting point of the project; RVC inference files in `src/` | `src/`, parts of `coversong/` | MIT, © 2023 SociallyIneptWeeb | https://github.com/SociallyIneptWeeb/AICoverGen |
| RVC (Retrieval-based Voice Conversion) — model code in `src/infer_pack`, `src/vc_infer_pipeline.py`, `src/rmvpe.py` | `src/` | MIT, © RVC-Project | https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI |
| Applio — RVC training engine (unchanged; real-time, TTS and UI parts left out) | `coversong/third_party/applio/` | MIT, © 2026 AI Hispano (see the LICENSE there) | https://github.com/IAHispano/Applio |
| SwiftF0 — pitch model `model.onnx` (from swift-f0 0.3.0) | `coversong/third_party/swift_f0/` | MIT, © 2025-2026 Lars Nieradzik | https://github.com/lars76/swift-f0 |
| Basic Pitch — note model `nmp.onnx` | `coversong/third_party/basic_pitch/` | Apache-2.0, © 2022 Spotify AB | https://github.com/spotify/basic-pitch |
| HPA-RMVPE network code (from Vietnamese-RVC) | `coversong/third_party/hpa_rmvpe/` | MIT, © 2026 Phạm Huỳnh Anh | https://github.com/PhamHuynhAnh16/Vietnamese-RVC |

## Models downloaded on first use

| Model | Used for | Licence / terms | Source |
|---|---|---|---|
| RMVPE (`rmvpe.pt`) | melody tracking | MIT (RVC release) | https://huggingface.co/lj1995/VoiceConversionWebUI · paper: Wei et al., 2023, "RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music" |
| ContentVec / HuBERT (`contentvec/`, transformers format) | the content features every RVC v2 model uses | MIT | https://github.com/auspicious3000/contentvec · weights as published by Applio: https://huggingface.co/IAHispano/Applio |
| HPA-RMVPE weights | an alternative melody tracker | see the source | https://huggingface.co/AnhP/Vietnamese-RVC-Project |
| TITAN-Medium 48 kHz pretrain | the base model for training | Apache-2.0 | https://huggingface.co/blaise-tk/TITAN |
| Mel-Band RoFormer models: vocals, karaoke (aufr33 & viperx), dereverb (anvuew) | separation | each model's own terms (see the model pages) | downloaded by python-audio-separator from its model repository |
| Whisper | words and their timing (intelligibility, wordless parts) | MIT | https://github.com/openai/whisper |
| WeSpeaker ResNet34 (VoxCeleb) | voiceprints for the "Several voices" tab | CC-BY-4.0 | https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM · https://github.com/wenet-e2e/wespeaker |

## Libraries installed by start.bat

| Library | Licence | Source |
|---|---|---|
| PyTorch, torchaudio | BSD-3-Clause | https://pytorch.org |
| python-audio-separator | MIT | https://github.com/nomadkaraoke/python-audio-separator |
| Hugging Face transformers, huggingface_hub | Apache-2.0 | https://github.com/huggingface/transformers |
| Gradio | Apache-2.0 | https://github.com/gradio-app/gradio |
| FastAPI, Uvicorn | MIT, BSD-3-Clause | https://github.com/fastapi/fastapi · https://github.com/encode/uvicorn |
| faiss | MIT | https://github.com/facebookresearch/faiss |
| librosa | ISC | https://github.com/librosa/librosa |
| NumPy, SciPy, scikit-learn | BSD-3-Clause | https://numpy.org · https://scipy.org · https://scikit-learn.org |
| ONNX Runtime | MIT | https://github.com/microsoft/onnxruntime |
| torchcrepe | MIT | https://github.com/maxrmorrison/torchcrepe |
| torchfcpe | MIT | https://github.com/CNChTu/FCPE |
| soundfile, soxr | BSD-3-Clause, LGPL-2.1 | https://github.com/bastibe/python-soundfile · https://github.com/dofuuz/python-soxr |
| yt-dlp | Unlicense | https://github.com/yt-dlp/yt-dlp |
| Deno (JavaScript runtime for yt-dlp) | MIT | https://deno.com |
| PyYAML, tqdm, requests, matplotlib, tensorboard, einops, stftpitchshift, ffmpeg-python | MIT / BSD / Apache-2.0 (see each) | PyPI |

## Tools downloaded by start.bat

| Tool | Licence | Source |
|---|---|---|
| uv | MIT / Apache-2.0 | https://github.com/astral-sh/uv |
| Python 3.12 (python-build-standalone) | PSF License | https://github.com/astral-sh/python-build-standalone |
| FFmpeg (BtbN Windows build, GPL variant) | GPL-3.0 — downloaded separately, not part of this repository | https://github.com/BtbN/FFmpeg-Builds · https://ffmpeg.org |

## Not used any more

- **pedalboard** (GPL-3.0), used by AICoverGen for effects, was replaced by the project's own DSP code
  (`coversong/dsp.py`, `coversong/vocal_fx.py`), so that the whole project can stay MIT.
- **fairseq** is not needed: the HuBERT/ContentVec encoder is loaded with Hugging Face transformers.
