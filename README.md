# CoverSong Studio

**AI song covers in your own voice — mixed to sound like the original — plus a vocal studio to record your voice and train your own model.**

[Русская версия](README.ru.md)

CoverSong Studio takes a song (a file or a YouTube link), separates the vocals, re-sings them with an
[RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) voice model and puts them back into the
song with the original's tone, reverb and balance. It grew out of [AICoverGen](https://github.com/SociallyIneptWeeb/AICoverGen)
and now has its own pipeline, a multi-singer mode, a browser vocal studio and model training built on
[Applio](https://github.com/IAHispano/Applio)'s engine.

- **One voice** — a queue of songs, each turned into a cover with one model.
- **Several voices** — finds the singers of a song (by voice, or by the singer tags of lyrics from Genius), lets you
  listen to each and give each one its own model and key.
- **Vocal Studio** — record your voice in the browser: sing 5–10 songs you like, plus optional short exercises with
  an instant grade and plain-language tips; then press **Train** to get your own voice model.
- **Library** — every cover you have made: play it next to the original, download MP3/WAV, open its folder.
- English and Russian interface; the console and logs are in English.

> ⚠️ Use only voices you have the right to use — your own, or with the person's consent. Covers of songs are for
> personal use unless you have the rights to the song. See [Responsible use](#responsible-use).

## Quick start (Windows 10/11)

1. On GitHub press **Code → Download ZIP** and unpack it into a folder with ~20 GB free (a path without
   non-Latin letters is the safest, e.g. `D:\CoverSong-Studio`).
2. Double-click **`start.bat`**.

The first run downloads and installs everything into that folder — nothing is installed system-wide:
[uv](https://github.com/astral-sh/uv), Python 3.12, PyTorch (for your NVIDIA GPU, or the CPU build without one),
the other libraries, and ffmpeg. This takes 5–30 minutes. When it is done the browser opens the app by itself;
later runs start in seconds. If the installation stops (e.g. the internet drops), just run `start.bat` again — it
continues. Deleting the folder removes everything.

**Hardware.** An NVIDIA GPU with 6+ GB of memory is recommended (8+ GB for training). Everything also runs on the
CPU, only much slower. GTX 10-series and older cards get PyTorch 2.7.1 automatically.

### First cover

1. You need a voice model. Either train your own (next section), or put an RVC v2 model you have the right to use
   into `rvc_models/<name>/` (`<name>.pth` and, if you have it, the `.index` file) and press **↻**.
2. **One voice** tab: add songs (files and/or YouTube links, one per line), choose the model, press
   **🎤 Make covers**. Keep **Voice key = Auto**: the key is chosen from your range.
3. The cover (MP3 and WAV) and every stage's files land in `song_output/<song>/`. Running again with other sliders
   only redoes the stages that changed — usually just the mix, which takes seconds.

### Your own voice model

1. Open **Vocal Studio** (the third tab, or `http://127.0.0.1:7861/studio/`).
2. **+ New voice** — a name and a voice type (it only sets the starting range).
3. **Songs**: sing 5–10 songs you like, whole, 1–5 minutes each — a cappella, or along with the song **in headphones**
   so the music does not get into the microphone. Each take gets a quick check (level, clipping, noise, how much is
   sung) before you save it.
4. Optional exercises fill the gaps: steady notes, humming, sirens, runs, staccato, soft singing, shouts, English
   words, and **breathing** (your breaths then replace the radio-like hiss RVC makes of the original's breaths).
5. **Train** (at least 3 minutes of singing; 15+ is better). Training takes from an hour to several hours. The best
   version is chosen on held-out phrases of your recordings and installed into `rvc_models/<name>/`; a blind
   comparison of the two best versions is left in `models/train/<name>/listen/`.

Recordings stay on your computer in `voices/<name>/recordings/`.

## How a cover is made

Every stage is cached in `song_output/<song>/`, so a change only redoes what depends on it.

| Stage | What happens |
|---|---|
| input | file or YouTube link → WAV |
| separate | Mel-Band RoFormer vocals → lead / backing (karaoke model) → dereverb, via [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) |
| convert | RVC re-sings the lead and the backing; the melody is tracked with RMVPE / CREPE / SwiftF0 / HPA-RMVPE and the least creaky result wins; note edges are smoothed; phrases outside your range are moved by an octave; `index_rate` and `protect` are tuned per song by how well [Whisper](https://github.com/openai/whisper) still understands the words |
| clean | wordless singing (oo, aa) handled softly, clicks repaired, noise between harmonics lowered, air above 8 kHz restored from the original, choir where the original has chords ([Basic Pitch](https://github.com/spotify/basic-pitch)), your breaths where the original breathes |
| fx | your vocal's tone and reverb matched to the original's, including passages with a "radio" or muffled effect |
| mix | instrumental + your lead and backing at the original's balance and loudness → MP3 + WAV |

The settings with comments are in [`config.yaml`](config.yaml). The command line works too:

```
.venv\Scripts\python -m coversong.cli "D:\Music\song.mp3" --model my_voice --pitch auto
```

## How it differs from the projects it builds on

This project stands on the shoulders of the projects below and does not replace them: they are larger, have their
own communities and do many things this one does not (Applio, for instance, has TTS, real-time conversion and many
more training options). The differences, as of the versions this project was built on:

- **[AICoverGen](https://github.com/SociallyIneptWeeb/AICoverGen)** (the starting point) — one RVC pass over the
  separated vocal and a fixed mix. Here: a staged and cached pipeline; the backing vocals are converted too; the
  melody method and the RVC settings are chosen per song by measurement; clean-up and effects matched to the
  original; several singers per song; recording and training your own voice; one-click setup without
  fairseq/compilers; the GPL `pedalboard` dependency replaced by our own DSP so the whole project is MIT.
- **[RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)** — the original voice
  conversion and training toolkit, working on single audio files. Here RVC is one stage of a song pipeline, and the
  training data come from a guided studio with instant feedback.
- **[Applio](https://github.com/IAHispano/Applio)** — a full-featured RVC toolkit; its training engine is used here
  unchanged (`coversong/third_party/applio`). Added around it: a dataset prepared from the studio takes (clipping
  and missing highs checked, ~8 % of phrases held out), the [TITAN](https://huggingface.co/blaise-tk/TITAN)
  pretrain, an automatic choice of the best epoch on the held-out phrases and a blind A/B of the two best.
- **UVR / python-audio-separator** — separation only; used here as the first stage.

## Folders

```
start.bat          the only launcher: installs on the first run, then starts the app
config.yaml        settings (most are also in the app)
coversong/         the app: pipeline stages, UI (ui.py, ui_multi.py), studio/, training (train_rvc.py)
src/               RVC inference (from AICoverGen / RVC, adapted)
rvc_models/        voice models: rvc_models/<name>/<name>.pth (+ .index); rmvpe.pt and contentvec/ are downloaded here
voices/            studio recordings: voices/<name>/recordings/
song_output/       covers and every stage's files
models/            separation, Whisper, voiceprint and training files (downloaded / created on use)
logs/              everything the app printed (logs/last_run.txt is the latest)
tools/, .venv/     uv, Python, ffmpeg and the libraries (created by start.bat)
```

None of the voice models, recordings, songs or downloaded weights are part of this repository (see `.gitignore`).

## Troubleshooting

- **Setup stopped** — run `start.bat` again; the details are in `logs/setup.txt`.
- **"There is an NVIDIA GPU, but PyTorch cannot use it"** — update the NVIDIA driver and restart.
- **The app stopped** — the full log is in `logs/last_run.txt`.
- **YouTube link fails** — YouTube changes often; update yt-dlp:
  `tools\uv\uv.exe pip install --python .venv\Scripts\python.exe -U "yt-dlp[default,deno]"` or download the song and add the file.
- **Out of GPU memory while training** — set `train.batch_size` to 6 or 4 in `config.yaml`.
- Check the environment: `.venv\Scripts\python -m coversong.selfcheck`.

## Responsible use

- Make covers only with voices you have the right to use: your own, or with the explicit consent of the person.
  Do not imitate real people to deceive, and do not present AI covers as real recordings of the original singer.
- Songs are copyrighted. Covers are for personal use unless you have the rights or a licence; publishing them may
  require permission from the rights holders, depending on your country and platform.
- The software is provided "as is", without warranty (see [LICENSE](LICENSE)). You are responsible for what you make
  with it.

## Credits and licences

CoverSong Studio is released under the [MIT licence](LICENSE). It includes or downloads the work of others — see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for every project, model and its licence.
