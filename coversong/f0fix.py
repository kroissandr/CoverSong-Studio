"""Melody before RVC sings it: note edges smoothed, phrases below the voice model's range an octave higher.

1. Note edges. RMVPE switches the voice on and off abruptly: a frame it is unsure about becomes "unvoiced", and RVC
   then sings it with noise instead of a tone. Measured on the covers of 13.10: 72–94 % of the creak and rasp
   sits in the first/last 50 ms of a note (in one test song — an older render with crepe, which switches more
   softly — had half as much). So, before RVC sings:
   - short holes inside a note (up to 50 ms) are filled when the tracker still hears a pitch there at a looser
     threshold, the pitch matches both sides and the sound is tonal (not "s"/"sh");
   - the start and end of a note are extended by up to 30 ms under the same conditions (voiced consonants "m", "n",
     "l" and the attack of the vowel);
   - isolated 10–20 ms "notes" in silence and single-frame jumps of the pitch are removed.
   convert.smooth_edges: false switches this off.
2. The song's octave shift is chosen so that most of the lead's notes fit your range. A phrase sung much lower than that — a low
   double in the backing vocal, a spoken-low ad-lib — ends up below anything the model can sing (Runaway Baby, 1:07:
   the backing's "I love you so" an octave under Bruno, shifted down another octave → D#2, while your lowest note
   is G2) and comes out deep, broken and robotic. Such phrases are sung an octave higher instead; phrases far
   above your range (very high ad-libs and vocal chops) an octave lower; single notes below your lowest notes an
   octave higher.
3. Backing vocals: where an octave double makes RMVPE follow the lower voice, the upper one is sung
   (upper_voice; checked with SwiftF0).

(An automatic fix for notes the pitch tracker reads an octave low was tried and removed: by ear the cover was
better without it.)
"""
import numpy as np

SR, HOP = 16000, 160          # RVC's pitch frames: 10 ms at 16 kHz
RANGE = [None]   # (lowest, highest) comfortable MIDI note of the current model; set by RVCSession.convert
ENABLED = [True]  # convert.fix_low_phrases
SMOOTH = [True]   # convert.smooth_edges
UPPER = [False]   # backing vocals: follow the upper voice of an octave double (see upper_voice)
FILL = [True]     # backing vocals: clear tones RMVPE missed are sung (fill_voicing); convert.backing_fill_voicing
LOOSE_THRED = 0.01   # RMVPE's own threshold is 0.03
MAX_GAP, MAX_EXT = 5, 3  # frames (10 ms)
STATS = {}


def set_range(r):
    RANGE[0] = r


def set_enabled(on=True):
    ENABLED[0] = bool(on)


def set_smooth(on=True):
    SMOOTH[0] = bool(on)


def set_upper(on=False, fill=True):
    UPPER[0] = bool(on)
    FILL[0] = bool(fill)


def phrases(f0, gap=25, min_voiced=30):
    """Sung phrases: voiced frames with pauses up to `gap` frames merged; (start, end) frame indices."""
    v = np.flatnonzero(f0 > 0)
    if not len(v):
        return []
    cuts = np.flatnonzero(np.diff(v) > gap)
    starts = np.r_[v[0], v[cuts + 1]]
    ends = np.r_[v[cuts], v[-1]] + 1
    return [(a, b) for a, b in zip(starts, ends) if (f0[a:b] > 0).sum() >= min_voiced]


HIGH_MARGIN = 2.0  # semitones above the top of the comfortable range (which already has +3)


def range_guard(f0, up_key, offset_s=0.0, comfort=False):
    """Phrases below the voice's range an octave up; phrases far above it (very high ad-libs, vocal chops — in
    Standing Next to You the intro has notes around F#6) an octave down. Returns (f0, moved): moved =
    [(start s, end s, median note, +12/-12)].
    comfort (backing vocals): a phrase is lifted already when it sits under the comfortable part of the range
    (p10) and an octave up still fits under p90 — the song's shift is chosen for the lead, and a backing line
    often sits lower (KATSEYE Animal 0:13: a group hum G3–C4, right in your range, shifted down to G2–C3)."""
    r = RANGE[0]
    if not r:
        return f0, []
    lo, hi = r
    low_edge, top = (lo + 3, hi - 2) if comfort else (lo, hi)   # RANGE = (p10 - 3, p90 + 3)
    moved = []
    for a, b in phrases(f0):
        seg = f0[a:b]
        m = float(np.median(12 * np.log2(seg[seg > 0] / 440.0) + 69)) + up_key
        if m < low_edge and m + 12 <= top:
            seg[seg > 0] *= 2
            moved.append((a * HOP / SR - offset_s, b * HOP / SR - offset_s, m, 12))
        elif m > hi + HIGH_MARGIN and m - 12 >= lo:
            seg[seg > 0] /= 2
            moved.append((a * HOP / SR - offset_s, b * HOP / SR - offset_s, m, -12))
    return f0, moved


def notes(f0, jump=1.5, min_len=5):
    """Single notes: voiced runs split where the pitch jumps by more than `jump` semitones; (start, end)."""
    out = []
    v = f0 > 0
    for a, b in _runs(v):
        m = 12 * np.log2(f0[a:b] / 440.0)
        cuts = np.flatnonzero(np.abs(np.diff(m)) > jump) + 1
        for s0, e0 in zip(np.r_[0, cuts], np.r_[cuts, b - a]):
            if e0 - s0 >= min_len:
                out.append((a + s0, a + e0))
    return out


def floor_guard(f0, up_key, offset_s=0.0):
    """Single notes that would land below the bottom of your range (the phrase around them may be fine): an octave
    up — or, for a short scoop into a note (under 150 ms, right before or after it), the pitch of that note.
    Standing Next to You, 0:22: the backing's D4 read as D3 and shifted to D2; the scoop into "you" sung at G2."""
    r = RANGE[0]
    if not r:
        return f0, []
    lo, hi = r
    moved = []
    ns = notes(f0, min_len=3)
    low = lambda a, b: float(np.median(12 * np.log2(f0[a:b] / 440.0) + 69)) + up_key  # noqa: E731
    scoops = []
    for k, (a, b) in enumerate(ns):
        m = low(a, b)
        if m >= lo:
            continue
        nb = [ns[j] for j in (k - 1, k + 1) if 0 <= j < len(ns)]
        nb = [(c, d) for c, d in nb if (d - c) >= 2 * (b - a) and min(abs(a - d), abs(c - b)) <= 12]
        if b - a < 15 and nb:
            scoops.append((a, b, m, max(nb, key=lambda cd: cd[1] - cd[0])))
        elif m + 12 <= hi and b - a >= 5:
            f0[a:b] *= 2
            moved.append((a * HOP / SR - offset_s, b * HOP / SR - offset_s, m, 12))
    for a, b, m, (c, d) in scoops:  # after the lifts: the note it leads into may have moved up
        target = f0[c] if c >= b else f0[d - 1]
        tm = 12 * np.log2(target / 440.0) + 69 + up_key
        if tm >= lo and tm - m <= 9:      # a real scoop: a few semitones under its note
            f0[a:b] = target
            moved.append((a * HOP / SR - offset_s, b * HOP / SR - offset_s, m, 0))
        elif tm - m > 9:                  # far under it: a misread breath or noise, not sung
            f0[a:b] = 0
    return f0, moved


OVERRIDE = [None]  # choir layers (choir.py): the melody to sing, Hz per 10 ms of the file; replaces the tracker
SCALE = [1.0]      # choir layers: detune — a number, or per 10 ms of the file (slow drift, like a real singer)
QUIET = [False]    # choir layers: no log lines


def set_layer(override=None, scale=1.0, quiet=False):
    OVERRIDE[0] = None if override is None else np.asarray(override, np.float64)
    SCALE[0] = float(scale) if np.ndim(scale) == 0 else np.asarray(scale, np.float64)
    QUIET[0] = bool(quiet)


REF = [None]   # backing: the upper of the two channels' readings, per 10 ms of the file (set by convert_stereo)


def set_reference(ref):
    REF[0] = None if ref is None else np.asarray(ref, np.float64)


def upper_voice(f0, x, min_len=8, offset_s=0.0, sw=None):
    """Backing vocals often have an octave double (Standing Next to You, 1:08: falsetto D5 with a D4 under it).
    RMVPE then may follow the lower voice, and after your octave shift you sing two octaves under the line.
    Where SwiftF0 hears the same note an octave higher with confidence for at least 80 ms — or the other stereo
    channel of the backing has it an octave higher (REF) — the whole note follows the upper voice."""
    n = len(f0)
    if sw is None:
        from .f0_extra import swift
        sw, _ = swift(x, n)
    sw = sw[:n] if len(sw) >= n else np.pad(sw, (0, n - len(sw)))
    both = (f0 > 0) & (sw > 0)
    d = np.full(n, np.nan)
    d[both] = 12 * np.log2(sw[both] / f0[both])
    cand = both & (np.abs(d - 12) < 1.5)
    ref = np.zeros(n)
    if REF[0] is not None:  # the file's frames start after RVC's padding
        k = int(round(offset_s * SR / HOP))
        r = REF[0][:max(0, n - k)]
        ref[k:k + len(r)] = r
        rb = (f0 > 0) & (ref > 0)
        dr = np.full(n, np.nan)
        dr[rb] = 12 * np.log2(ref[rb] / f0[rb])
        cand |= rb & (np.abs(dr - 12) < 1.5)
    # close 1–2 frame gaps
    for a, b in _runs(~cand):
        if 0 < a and b < n and b - a <= 2:
            cand[a:b] = True
    moved = []
    done = np.zeros(n, bool)
    orig = f0.copy()
    for a, b in _runs(cand):
        if b - a < min_len or done[a:b].any():
            continue
        med = np.median(orig[a:b])

        def same_note(i):  # the rest of the note, unless SwiftF0 hears it at RMVPE's octave
            return (not done[i] and orig[i] > 0 and abs(12 * np.log2(orig[i] / med)) < 1.5
                    and not (sw[i] > 0 and abs(12 * np.log2(sw[i] / orig[i])) < 1.0)
                    and not (ref[i] > 0 and abs(12 * np.log2(ref[i] / orig[i])) < 1.0))
        while a > 0 and same_note(a - 1):
            a -= 1
        while b < n and same_note(b):
            b += 1
        f0[a:b] = orig[a:b] * 2
        done[a:b] = True
        moved.append((a, b))
    return f0, moved


def breath_gate(f0, x, sw_loose=None):
    """Breaths, sighs and noise that the tracker gave a pitch to: RVC would sing them as a note — a metallic,
    tonal sigh (HOT, 0:45 in the backing). A voiced stretch is left without pitch when most of it is not tonal (its
    energy is not in the voice's low band) and SwiftF0 does not hear a pitch there at all."""
    n = len(f0)
    if sw_loose is None:
        from .f0_extra import swift
        _, sw_loose = swift(x, n)
    sw_loose = sw_loose[:n] if len(sw_loose) >= n else np.pad(sw_loose, (0, n - len(sw_loose)))
    noisy = ~tonal(x, n) & (sw_loose <= 0)
    gone = []
    for a, b in _runs(f0 > 0):
        if noisy[a:b].mean() >= 0.7:
            f0[a:b] = 0
            gone.append((a, b))
    return f0, gone


def fill_voicing(f0, x, sw, swl=None, min_len=5):
    """Backing vocals: where RMVPE says "no note" but the sound is clearly a sung tone — SwiftF0 hears a pitch
    (with confidence, or at least half-hears it where the waveform repeats very cleanly), most energy is in the
    voice's band and the waveform repeats (periodicity >= 0.8) — the note is sung at SwiftF0's pitch, in the
    octave of the note it continues. 72–88 % of the backing's creak was such moments: the original sang a clear
    note, RVC got none and sang noise. Short gaps (from 20 ms) count when they touch a sung note."""
    from .quality import periodicity
    n = len(f0)
    swl = sw if swl is None else swl
    per, _ = periodicity(x, SR)
    pr = np.zeros(n)
    k = min(n - 2, len(per))           # periodicity frame i covers x[i*160 : i*160+640]: centred on RVC frame i+2
    if k > 0:
        pr[2:2 + k] = per[:k]
    ok = (f0 <= 0) & tonal(x, n) & (((sw > 0) & (pr >= 0.8)) | ((swl > 0) & (pr >= 0.85)))
    pitch = np.where(sw > 0, sw, swl)
    added = []
    for a, b in _runs(ok):
        left = f0[a - 1] if a > 0 and f0[a - 1] > 0 else 0.0
        right = f0[b] if b < n and f0[b] > 0 else 0.0
        if b - a < (2 if (left or right) else min_len):
            continue
        seg = pitch[a:b].copy()
        ref = left or right
        if ref:  # keep the octave of the note it continues
            seg *= 2.0 ** np.round(np.log2(ref / np.median(seg)))
        f0[a:b] = seg
        added.append((a, b))
    return f0, added


def unpitched_gate(f0, x, swl, min_len=4):
    """Backing vocals: where the original has no note at all — whispers, breathy or spoken parts, a smear of
    voices — RMVPE still often names one, and RVC sings a clean steady tone where the original is air: the
    "robot" in Bite Me's backing (29 % of its loud backing). A stretch is sung without pitch (breathy, like the
    original) when SwiftF0 does not even half-hear a pitch, Basic Pitch hears no clear note (< 0.5) and the
    waveform hardly repeats (periodicity < 0.75)."""
    from .choir import posteriors
    from .quality import periodicity
    n = len(f0)
    note, fps = posteriors(x, SR)
    idx = np.clip((np.arange(n) / 100 * fps).round().astype(int), 0, len(note) - 1)
    bp = note.max(1)[idx]
    per, _ = periodicity(x, SR)
    pr = np.ones(n)
    k = min(n - 2, len(per))
    if k > 0:
        pr[2:2 + k] = per[:k]
    gate = (f0 > 0) & (swl <= 0) & (bp < 0.5) & (pr < 0.75)
    gone = []
    for a, b in _runs(gate):
        if b - a >= min_len:
            f0[a:b] = 0
            gone.append((a, b))
    return f0, gone


# ---------------------------------------------------------------- note edges
def tonal(x, n):
    """Per pitch frame (centred at i*HOP, like RMVPE): is it loud enough and is most of its energy below 1.5 kHz
    (a vowel or a voiced consonant, not "s"/"sh"/breath)."""
    x = np.asarray(x, np.float32)
    win = 512
    xp = np.pad(x, (win // 2, win + n * HOP))
    f = np.fft.rfftfreq(win, 1 / SR)
    low = (f >= 60) & (f < 1500)
    w = np.hanning(win).astype(np.float32)
    db = np.zeros(n, np.float32)
    ratio = np.zeros(n, np.float32)
    for c in range(0, n, 4000):
        idx = np.arange(c, min(n, c + 4000))
        fr = np.stack([xp[i * HOP:i * HOP + win] for i in idx]) * w
        S = np.abs(np.fft.rfft(fr, axis=1)) ** 2
        tot = S[:, f >= 60].sum(1) + 1e-12
        ratio[idx] = S[:, low].sum(1) / tot
        db[idx] = 10 * np.log10(tot)
    return (db > np.percentile(db, 98) - 45) & (ratio > 0.6)


def _runs(mask):
    """(start, end) of True runs."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _st(a, b):
    return abs(12 * np.log2(a / b))


def smooth_edges(f0, loose, x):
    """See the module docstring. f0, loose: RMVPE f0 at the normal and the loose threshold (Hz, 0 = unvoiced);
    x: the 16 kHz audio they were read from. Returns (f0, stats)."""
    f0 = np.asarray(f0, np.float64).copy()
    n = len(f0)
    st = {"blips": 0, "spikes": 0, "holes": 0, "edges": 0}
    if n < 10:
        return f0, st
    loose = np.asarray(loose, np.float64)[:n] if loose is not None else np.zeros(n)
    if len(loose) < n:
        loose = np.pad(loose, (0, n - len(loose)))
    ok = tonal(x, n) & (loose > 0)
    # 1. isolated 1–2 frame "notes"
    for a, b in _runs(f0 > 0):
        if b - a <= 2 and (a == 0 or not (f0[max(0, a - 5):a] > 0).any()) and not (f0[b:b + 5] > 0).any():
            f0[a:b] = 0
            st["blips"] += 1
    # 2. single-frame pitch jumps
    for i in range(1, n - 1):
        p, c, q = f0[i - 1], f0[i], f0[i + 1]
        if p > 0 and c > 0 and q > 0 and _st(p, q) < 1 and _st(c, p) > 3 and _st(c, q) > 3:
            f0[i] = np.sqrt(p * q)
            st["spikes"] += 1
    # 3. short holes inside a note
    for a, b in _runs(f0 <= 0):
        if a == 0 or b >= n or b - a > MAX_GAP:
            continue
        p, q = f0[a - 1], f0[b]
        if _st(p, q) > 2 or not ok[a:b].all():
            continue
        t = (np.arange(a, b) - (a - 1)) / (b - (a - 1))
        interp = np.exp(np.log(p) + t * (np.log(q) - np.log(p)))
        lz = loose[a:b]
        f0[a:b] = np.where(np.abs(12 * np.log2(lz / interp)) <= 1, lz, interp)
        st["holes"] += 1
    # 4. onsets and offsets
    v = f0 > 0
    for a, b in _runs(v):
        if b - a < 5:
            continue
        ext = 0
        for i in range(a - 1, max(-1, a - 1 - MAX_EXT), -1):
            if v[i] or not ok[i] or _st(loose[i], f0[i + 1]) > 2:
                break
            f0[i] = loose[i]
            ext += 1
        for i in range(b, min(n, b + MAX_EXT)):
            if v[i] or not ok[i] or _st(loose[i], f0[i - 1]) > 2:
                break
            f0[i] = loose[i]
            ext += 1
        st["edges"] += ext > 0
    return f0, st


_last = [None]


def apply(x, f0, offset_s=0.0, up_key=0, model=None, loose=None):
    """Hook for RVC's pitch step (before its octave shift). loose: what the tracker half-hears (f0 at a looser
    voicing threshold), or model: the RMVPE instance that produced f0 (its last salience map is decoded again at
    a looser threshold). Logs once per distinct result (a song is converted several times)."""
    f0 = np.asarray(f0, np.float64).copy()
    def on_frames(a, fill):  # a per-file-frame array placed on this call's frames (RVC pads the audio)
        k = int(round(offset_s * SR / HOP))
        out = np.full(len(f0), fill, np.float64)
        r = np.asarray(a, np.float64)[:max(0, len(f0) - k)]
        out[k:k + len(r)] = r
        return out
    scale = SCALE[0] if np.ndim(SCALE[0]) == 0 else on_frames(SCALE[0], 1.0)
    if OVERRIDE[0] is not None:  # a choir layer: the given line, only the range guards apply
        ov = on_frames(OVERRIDE[0], 0.0)
        tr = f0.copy()
        tr[tr < 50] = 0
        both = (ov > 0) & (tr > 0)
        # the line's notes, but with the singer's own small pitch movements (a flat MIDI line sounds robotic):
        # the tracker's reading moved by the interval between the line's note and the nearest note of the reading
        mt = 12 * np.log2(np.where(both, tr, 440.0) / 440.0)
        ml = 12 * np.log2(np.where(both, ov, 440.0) / 440.0)
        f0 = np.where(both, tr * 2 ** ((np.round(ml) - np.round(mt)) / 12), ov) * scale
        if ENABLED[0]:
            f0, _ = range_guard(f0, up_key, offset_s, comfort=UPPER[0])
            f0, _ = floor_guard(f0, up_key, offset_s)
        return f0
    f0[f0 < 50] = 0  # RVC's own lower limit; readings under it (HPA-RMVPE gives some at 25–45 Hz) are noise
    STATS.clear()
    gone, filled, airy = [], [], []
    sw = swl = None
    try:
        from .f0_extra import swift
        sw, swl = swift(x, len(f0))
        sw, swl = (np.pad(a[:len(f0)], (0, max(0, len(f0) - len(a)))) for a in (sw, swl))
    except Exception as e:  # SwiftF0 missing or failed: the checks that need it are skipped
        print(f"f0fix swift: {e}")
    if sw is not None:
        f0, gone = breath_gate(f0, x, swl)
        if UPPER[0]:
            try:
                f0, airy = unpitched_gate(f0, x, swl)
            except Exception as e:  # Basic Pitch missing or failed: keep the tracker's reading
                print(f"f0fix unpitched_gate: {e}")
        if UPPER[0] and FILL[0]:
            f0, filled = fill_voicing(f0, x, sw, swl)
    if SMOOTH[0]:
        if loose is None and model is not None and getattr(model, "last_hidden", None) is not None:
            loose = model.decode(model.last_hidden, thred=LOOSE_THRED)
        f0, st = smooth_edges(f0, loose, x)
        STATS.update(st)
    moved, notes_up, upper = [], [], []
    if UPPER[0]:
        try:
            f0, upper = upper_voice(f0, x, offset_s=offset_s, sw=sw)
        except Exception as e:  # SwiftF0 missing or failed: keep RMVPE's reading
            print(f"f0fix upper_voice: {e}")
    if ENABLED[0]:
        f0, moved = range_guard(f0, up_key, offset_s, comfort=UPPER[0])
        f0, notes_up = floor_guard(f0, up_key, offset_s)
    if np.ndim(scale) or scale != 1.0:
        f0 = f0 * scale
    if QUIET[0]:
        return f0
    msg = ""
    names = "C C# D D# E F F# G G# A A# B".split()

    def where(items):
        return ", ".join(f"{int(max(a, 0) // 60)}:{max(a, 0) % 60:04.1f} (would be around "
                         f"{names[int(round(m)) % 12]}{int(round(m)) // 12 - 1})"
                         for a, _, m, _ in items[:8]) + (" …" if len(items) > 8 else "")
    up = [mv for mv in moved if mv[3] > 0]
    down = [mv for mv in moved if mv[3] < 0]
    parts = []
    if up:
        parts.append(f"Phrases below the voice's range sung an octave higher: {where(up)}")
    if down:
        parts.append(f"Phrases above the voice's range sung an octave lower: {where(down)}")
    lifted = [mv for mv in notes_up if mv[3] > 0]
    scoops = [mv for mv in notes_up if mv[3] == 0]
    if lifted:
        parts.append(f"Single notes below the voice's range sung an octave higher: {where(lifted)}")
    if scoops:
        parts.append(f"Scoops into notes from below the voice's range sung level, at the note: {where(scoops)}")
    if upper:
        ts = ", ".join(f"{int(max(t, 0) // 60)}:{max(t, 0) % 60:04.1f}" for t in
                       [a * HOP / SR - offset_s for a, _ in upper[:10]]) + (" …" if len(upper) > 10 else "")
        parts.append(f"Octave doubles in the backing — the upper voice sung ({len(upper)} notes): {ts}")
    if filled:
        tot = sum(b - a for a, b in filled) / 100
        parts.append(f"Backing: clean notes RMVPE missed sung from SwiftF0 ({len(filled)} places, {tot:.1f} s)")
    if airy:
        tot = sum(b - a for a, b in airy) / 100
        parts.append(f"Backing: where the original has no notes (whisper, breath, blurred voices) sung without pitch "
                     f"({len(airy)} places, {tot:.1f} s)")
    if gone:
        ts = ", ".join(f"{int(max(t, 0) // 60)}:{max(t, 0) % 60:04.1f}" for t in
                       [a * HOP / SR - offset_s for a, _ in gone[:10]]) + (" …" if len(gone) > 10 else "")
        parts.append(f"Breaths and noises where the model heard a note left without pitch ({len(gone)}): {ts}")
    msg = "\n".join(parts)
    if msg and msg != _last[0]:
        from .util import log
        for line in msg.split("\n"):
            log(line)
    _last[0] = msg or _last[0]
    return f0
