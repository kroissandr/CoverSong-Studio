"""Basic Pitch note model (Spotify, https://github.com/spotify/basic-pitch, Apache 2.0 — see LICENSE here):
saved_models/icassp_2022/nmp.onnx and the framing it expects (basic_pitch/constants.py, inference.py).
Used by coversong/choir.py to hear several notes at once (chords in backing vocals)."""
SAMPLE_RATE = 22050
FFT_HOP = 256
WINDOW = 43844          # samples per model call (2 s - one hop)
OVERLAP_FRAMES = 30
BASE_MIDI = 21          # key 0 = A0
