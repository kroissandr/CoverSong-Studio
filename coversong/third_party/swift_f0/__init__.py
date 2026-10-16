"""SwiftF0 model (Lars Nieradzik, https://github.com/lars76/swift-f0, MIT — see LICENSE here): model.onnx from
swift-f0 0.3.0 and the framing it expects (swift_f0/core.py). Used by coversong/f0_extra.py."""
SAMPLE_RATE = 16000
HOP = 256          # 16 ms frames
LEFT = 11          # frames of left context per window
LOOK = 10          # frames of look-ahead per window
WINDOW = 1875      # frames per window (30 s)
