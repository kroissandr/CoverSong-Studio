"""What to record, per voice: a plan built from the voice's range and from what it already has.

Every voice (= the model it is recorded for) has its own folder: voices/<name>/
    recordings/     the saved takes (training data)
    voice.json      voice type, notes sung so far
    progress.json   takes per task

Most tasks are short (2–8 s) with one clear goal in plain words; thin notes come first. "Songs" are free: any songs
the singer likes, sung whole — the model learns most from real singing.
"""
import json
import os
import re
import time

from ..i18n import lang_of, note, note_name

# a voice type gives the starting range until the voice's own range is known (MIDI: usual low, middle, usual high)
VOICE_TYPES = {"bass": (43, 50, 59), "tenor": (48, 55, 64), "alto": (53, 60, 69), "soprano": (60, 67, 76)}
VOICE_TYPE_NAMES = {"en": {"bass": "Low male (bass, baritone)", "tenor": "Male (tenor, baritone)",
                           "alto": "Low female (alto, mezzo)", "soprano": "Female (soprano, mezzo)"},
                    "ru": {"bass": "Низкий мужской (бас, баритон)", "tenor": "Мужской (тенор, баритон)",
                           "alto": "Низкий женский (альт, меццо)", "soprano": "Женский (сопрано, меццо)"}}
SONG_SLOTS = 10
SONGS_GOOD = 5

CATEGORIES = [
    ("songs", {"en": ("Songs", "Sing 5–10 songs you like, whole — a cappella or along with the song in headphones. "
                               "The model learns most from real singing."),
               "ru": ("Песни", "Спойте 5–10 любых песен целиком — без музыки или под песню в наушниках. "
                               "Больше всего модель учится на настоящем пении.")}),
    ("note", {"en": ("Steady notes", "Hold one sound evenly at the right pitch — the base of everything."),
              "ru": ("Ровные ноты", "Тянуть один звук ровно на нужной высоте — основа всего.")}),
    ("hum", {"en": ("Humming", "\"Mmm\" with the mouth closed — the model learns to hum and hold nasal sounds."),
             "ru": ("Мычание", "«М-м-м» с закрытым ртом — модель учится мычать и тянуть носовые звуки.")}),
    ("glide", {"en": ("Sirens", "Slide smoothly from low to high or high to low, without breaks."),
               "ru": ("Сирены", "Плавно проехать голосом снизу вверх или сверху вниз, без срывов.")}),
    ("pattern", {"en": ("Runs", "Short melodies on \"ah\", \"na\", \"oo\" — like the runs in songs."),
                 "ru": ("Распевы", "Короткие мелодии на «а», «на», «у» — как распевы в песнях.")}),
    ("staccato", {"en": ("Staccato", "Short separate sounds, \"ha-ha-ha\", \"na-na-na\", on one note."),
                  "ru": ("Отрывисто", "Короткие отдельные звуки «ха-ха-ха», «на-на-на» на одной ноте.")}),
    ("soft", {"en": ("Soft and airy", "Quiet and breathy, like a whisper on a note."),
              "ru": ("Мягко, с воздухом", "Тихо и с придыханием, как шёпот на ноте.")}),
    ("shout", {"en": ("Shouts", "Short bright \"hey!\", \"yeah!\" — loud, but without rasp."),
               "ru": ("Выкрики", "Короткие яркие «эй!», «е-е!» — громко, но без хрипа.")}),
    ("english", {"en": ("Song words", "Words from songs on one note — the model learns English sounds."),
                 "ru": ("Английские слова", "Слова из песен на одной ноте — модель учится английским звукам.")}),
    ("breath", {"en": ("Breathing", "Inhales and sighs without voice — in covers they replace the radio-like hiss "
                                    "the model makes of the original's breaths."),
                "ru": ("Дыхание", "Вдохи и вздохи без голоса — встанут в каверах на место вздохов оригинала вместо "
                                  "«радио»-шума, который из них делает модель.")}),
]

VOWELS = [("a", "ah", "а"), ("o", "oh", "о"), ("u", "oo", "у"), ("e", "eh", "э"), ("i", "ee", "и")]
SYL = {"a": ("ah", "а"), "u": ("oo", "у"), "na": ("na", "на"), "ha": ("ha", "ха"), "ta": ("ta", "та"),
       "m": ("mmm", "м"), "hey": ("hey", "эй"), "yeah": ("yeah", "е-е"), "oh": ("oh", "о-о")}
REG = {"en": {"chest": "in chest voice — like loud, confident speech",
              "head": "lightly, in head voice — bright, no strain in the throat"},
       "ru": {"chest": "грудным голосом — как громкая уверенная речь",
              "head": "легко, «головой» — светло, без напряжения в горле"}}


def safe_name(s):
    """A voice name usable as a folder and a model name."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", str(s or "").strip()).strip("_")
    return s[:40]


# ================================================================== voices
def voices_root(cfg):
    return cfg["paths"].get("voices") or os.path.join(os.path.dirname(cfg["paths"]["rvc_models"]), "voices")


def voice_dir(cfg, voice):
    return os.path.join(voices_root(cfg), safe_name(voice))


def rec_dir(cfg, voice):
    return os.path.join(voice_dir(cfg, voice), "recordings")


def tmp_dir(cfg, voice):
    return os.path.join(voice_dir(cfg, voice), "_tmp")


def _read(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load_voice(cfg, voice):
    return _read(os.path.join(voice_dir(cfg, voice), "voice.json"), {})


def save_voice(cfg, voice, data):
    _write(os.path.join(voice_dir(cfg, voice), "voice.json"), data)


def has_model(cfg, voice):
    d = os.path.join(cfg["paths"]["rvc_models"], voice)
    return os.path.isdir(d) and any(f.lower().endswith(".pth") for f in os.listdir(d))


def list_voices(cfg):
    """Voices with a studio folder and installed models (a model can get more recordings and be trained again)."""
    names = set()
    for root, need in ((voices_root(cfg), None), (cfg["paths"]["rvc_models"], ".pth")):
        if os.path.isdir(root):
            for d in os.listdir(root):
                p = os.path.join(root, d)
                if os.path.isdir(p) and safe_name(d) == d and (need is None or has_model(cfg, d)):
                    names.add(d)
    out = []
    for n in sorted(names, key=str.lower):
        v = load_voice(cfg, n)
        out.append({"name": n, "type": v.get("type"), "model": has_model(cfg, n),
                    "recorded_min": round(recorded_seconds(cfg, n) / 60, 1)})
    return out


def create_voice(cfg, name, vtype="tenor"):
    name = safe_name(name)
    if not name:
        raise ValueError("empty name")
    os.makedirs(rec_dir(cfg, name), exist_ok=True)
    v = load_voice(cfg, name)
    v.setdefault("created", time.strftime("%Y-%m-%d %H:%M"))
    if vtype in VOICE_TYPES:
        v["type"] = vtype
    save_voice(cfg, name, v)
    return name


def recorded_seconds(cfg, voice):
    """Seconds of saved takes (from voice.json; files you put into recordings/ yourself are counted by training)."""
    v = load_voice(cfg, voice)
    return float(sum((v.get("takes") or {}).values()))


def add_take(cfg, voice, fname, seconds, hist):
    """Remember a saved take: its length and the seconds sung per note (for the range and the plan)."""
    v = load_voice(cfg, voice)
    v.setdefault("takes", {})[fname] = round(float(seconds), 2)
    if hist:
        v.setdefault("notes", {})[fname] = {str(k): round(float(s), 2) for k, s in hist.items() if s > 0}
    save_voice(cfg, voice, v)


def sung_per_note(cfg, voice):
    """{midi: seconds} over all saved takes."""
    tot = {}
    for h in (load_voice(cfg, voice).get("notes") or {}).values():
        for k, s in h.items():
            tot[int(k)] = tot.get(int(k), 0.0) + float(s)
    return tot


def _percentile(hist, q):
    keys = sorted(hist)
    total = sum(hist.values())
    acc = 0.0
    for k in keys:
        acc += hist[k]
        if acc >= q / 100 * total:
            return k
    return keys[-1]


def voice_range(cfg, voice):
    """The range to plan in. From the voice's model (voice_profile.json), else from what was sung in the studio
    (2+ minutes), else from the voice type."""
    path = os.path.join(cfg["paths"]["rvc_models"], voice, "voice_profile.json")
    src = "type"
    lo, mid, hi = VOICE_TYPES.get(load_voice(cfg, voice).get("type") or "tenor", VOICE_TYPES["tenor"])
    sung = sung_per_note(cfg, voice)
    if os.path.exists(path):
        p = _read(path, {})
        if p.get("median") is not None:
            lo, mid, hi = int(round(p["p10"])), int(round(p["median"])), int(round(p["p90"]))
            src = "model"
    elif sum(sung.values()) >= 120:
        lo, mid, hi = _percentile(sung, 10), _percentile(sung, 50), _percentile(sung, 90)
        src = "studio"
    hi = max(hi, lo + 7)
    chest = min(hi, mid + 8)  # above this, head voice is asked for
    return {"lo": max(36, lo - 3), "comfort_lo": lo, "mid": mid, "comfort_hi": hi, "hi": min(84, hi + 4),
            "chest_top": chest, "source": src}


def coverage(cfg, voice):
    """Seconds of singing per note name: the training set's count and the studio takes, whichever is more."""
    from .. import ROOT
    st = _read(os.path.join(ROOT, "models", "train", voice, "pitch.json"), {}) or {}
    cov = dict(st.get("seconds_per_note") or {})
    for m, s in sung_per_note(cfg, voice).items():
        cov[note_name(m)] = max(cov.get(note_name(m), 0.0), s)
    return cov


def load_progress(cfg, voice):
    return _read(os.path.join(voice_dir(cfg, voice), "progress.json"), {})


def save_progress(cfg, voice, prog):
    _write(os.path.join(voice_dir(cfg, voice), "progress.json"), prog)


# ================================================================== the plan
def build(cfg, voice, lang="en"):
    lang = lang_of(lang)
    ru = lang == "ru"
    r = voice_range(cfg, voice)
    cov = coverage(cfg, voice)
    tasks = []
    N = lambda m: note(m, lang)  # noqa: E731

    def say(en, ru_):
        return ru_ if ru else en

    def syl(k):
        return SYL[k][1] if ru else SYL[k][0]

    def add(cat, tid, title, how, target, seconds, text, reg=None, need=1, prio=0.0):
        tasks.append({"id": tid, "cat": cat, "title": title, "how": how, "target": target, "seconds": seconds,
                      "text": text, "register": reg, "need": need, "prio": prio})

    def gap(m):  # 0 (nothing recorded on this note) … 4 (a minute or more): thin notes come first
        return min(cov.get(note_name(m), 0.0), 60.0) / 15

    def reg_of(m):
        return "chest" if m <= r["chest_top"] else "head"

    lo, hi, mid = r["lo"], r["hi"], r["mid"]
    for k in range(1, SONG_SLOTS + 1):
        add("songs", f"song_{k:02d}", say(f"Song {k}", f"Песня {k}"),
            say("Sing any song you like, whole or a big part of it (1–5 minutes). A cappella, or along with the song "
                "or a karaoke track — but only in headphones, so the music does not get into the microphone. "
                "Press Stop when you finish.",
                "Спойте любую песню целиком или большой кусок (1–5 минут). Без музыки или под песню / минусовку — "
                "но только в наушниках, чтобы музыка не попала в микрофон. Закончили — нажмите «Стоп»."),
            None, 600.0, "", prio=-10 + k * 0.01)
    for m in range(lo, hi + 1):
        reg = reg_of(m)
        for i, (vid, ven, vru) in enumerate(VOWELS):
            v = vru if ru else ven
            add("note", f"note_{note_name(m)}_{vid}", say(f"\"{v}\" on {N(m)}", f"«{v}» на ноте {N(m)}"),
                say(f"Listen to the note, then hold \"{v}\" evenly for 4 seconds, {REG['en'][reg]}.",
                    f"Послушайте ноту, затем тяните «{v}» ровно 4 секунды, {REG['ru'][reg]}."),
                m, 4.0, v, reg, prio=gap(m) + i * 0.1)
    for m in range(lo, hi + 1, 2):
        add("hum", f"hum_{note_name(m)}", say(f"Hum \"mmm\" on {N(m)}", f"Мычание «м-м-м» на ноте {N(m)}"),
            say("Close your mouth and hold \"mmm\" on this note for 4 seconds — your lips should tickle a little.",
                "Закройте рот и тяните «м-м-м» на этой ноте 4 секунды — губы слегка щекочет."),
            m, 4.0, syl("m"), reg_of(m), prio=gap(m) + 1)
    scale = [0, 2, 4, 5, 7, 5, 4, 2, 0]
    arp = [0, 4, 7, 12, 7, 4, 0]
    for m in range(lo, hi - 6, 3):  # hums in backing vocals are melodies, not held notes
        add("hum", f"humscale_{note_name(m)}", say(f"Hummed run \"mmm\" up and down from {N(m)}",
                                                   f"Мычание-распев «м-м-м» вверх-вниз от {N(m)}"),
            say("Listen to the 9-note melody and hum it with your mouth closed — connected, like the \"mmm\" "
                "of backing vocals.",
                "Послушайте мелодию из 9 нот и промычите её с закрытым ртом — связно, как «м-м-м» в бэк-вокале."),
            [m + k for k in scale], 4.5, syl("m"), prio=gap(m + 3) + 1.5)
    for m in range(lo, hi - 6, 2):
        for vid in ("u", "a"):
            v = syl(vid)
            add("glide", f"glide_up_{note_name(m)}_{vid}",
                say(f"Siren up on \"{v}\": {N(m)} → {N(m + 7)}", f"Сирена вверх на «{v}»: {N(m)} → {N(m + 7)}"),
                say(f"Start on the low note and in 3 seconds slide smoothly, without steps, up to the high one on "
                    f"\"{v}\".",
                    f"Начните на нижней ноте и за 3 секунды плавно, без ступенек, доедьте до верхней на «{v}»."),
                [m, m + 7], 3.0, v, prio=gap(m + 4) + 2)
            add("glide", f"glide_down_{note_name(m)}_{vid}",
                say(f"Siren down on \"{v}\": {N(m + 7)} → {N(m)}", f"Сирена вниз на «{v}»: {N(m + 7)} → {N(m)}"),
                say(f"Start on the high note and slide smoothly down in 3 seconds on \"{v}\".",
                    f"Начните на верхней ноте и плавно съедьте вниз за 3 секунды на «{v}»."),
                [m + 7, m], 3.0, v, prio=gap(m + 4) + 2.5)
    for m in range(lo, hi - 6, 2):
        for sid in ("a", "na", "u"):
            s = syl(sid)
            add("pattern", f"scale_{note_name(m)}_{sid}",
                say(f"Run on \"{s}\" up and down from {N(m)}", f"Распев «{s}» вверх-вниз от {N(m)}"),
                say(f"Listen to the 9-note melody and sing it on \"{s}\" — each note separate, but connected.",
                    f"Послушайте мелодию из 9 нот и спойте её на «{s}» — каждая нота отдельно, но связно."),
                [m + k for k in scale], 4.5, s, prio=gap(m + 3) + 3)
    for m in range(lo, hi - 11, 3):
        s = syl("u")
        add("pattern", f"arp_{note_name(m)}_u", say(f"Chord run on \"{s}\" from {N(m)}", f"Распев-аккорд «{s}» от {N(m)}"),
            say(f"Listen to the melody and sing it on \"{s}-{s}-{s}\" — like the runs in choruses.",
                f"Послушайте мелодию и спойте её на «{s}-{s}-{s}» — это как распевы в припевах."),
            [m + k for k in arp], 3.5, s, prio=gap(m + 7) + 3.5)
    for m in range(max(lo, mid - 5), min(hi, mid + 6) + 1, 3):
        for sid in ("ha", "na", "ta"):
            s = syl(sid)
            add("staccato", f"stacc_{note_name(m)}_{sid}",
                say(f"Staccato \"{s}-{s}-{s}-{s}\" on {N(m)}", f"Отрывисто «{s}-{s}-{s}-{s}» на ноте {N(m)}"),
                say(f"Four short separate \"{s}\" on one note, like a chuckle, with pauses between them.",
                    f"Четыре коротких отдельных «{s}» на одной ноте, как смешок, с паузами между ними."),
                m, 3.0, s, prio=gap(m) + 4.5)
    for m in range(max(lo, mid - 3), min(hi, mid + 7) + 1, 3):
        s = syl("u")
        add("soft", f"soft_{note_name(m)}", say(f"Soft airy \"{s}\" on {N(m)}", f"Мягкое «{s}» с воздухом на ноте {N(m)}"),
            say("Quietly, almost a whisper, but on the note — with breath clearly in the sound. 3 seconds.",
                "Тихо, почти шёпотом, но на ноте — с заметным дыханием в звуке. 3 секунды."),
            m, 3.0, s, prio=gap(m) + 5)
    for m in range(max(lo, r["chest_top"] - 5), r["chest_top"] + 1, 2):
        for sid in ("hey", "yeah", "oh"):
            s = syl(sid)
            add("shout", f"shout_{note_name(m)}_{sid}", say(f"Shout \"{s}!\" on {N(m)}", f"Выкрик «{s}!» на ноте {N(m)}"),
                say("Short and bright, as if calling a friend across the street, but without rasp. "
                    "Move a hand's width away from the microphone so it does not overload.",
                    "Коротко и ярко, как будто зовёте друга через улицу, но без хрипа. "
                    "Отодвиньтесь от микрофона на ладонь, чтобы не было перегруза."),
                m, 2.0, s, "chest", prio=gap(m) + 6)
    words = ["baby", "yeah", "this is it", "say my name", "the light", "stay", "so sweet", "with you", "think",
             "together"]
    for i, w in enumerate(words):
        m = mid + (i % 3) * 2 - 2
        add("english", f"eng_{i}_{note_name(m)}", say(f"Sing \"{w}\" on {N(m)}", f"Спеть «{w}» на ноте {N(m)}"),
            say(f"Sing \"{w}\" on one note, pronouncing every sound clearly. 2–3 seconds.",
                f"Пропойте «{w}» на одной ноте, чётко проговаривая все звуки. 2–3 секунды."),
            m, 3.0, w, reg_of(m), prio=gap(m) + 7)
    breaths = [
        ("in_mouth", say("Inhale through the mouth before a line", "Вдох ртом перед строчкой"),
         say("Like before you start singing: a calm, audible breath in through the mouth, 1 second. Keep the "
             "microphone a hand's width away and a little to the side, so the air does not hit it.",
             "Как перед тем, как запеть: спокойный слышный вдох ртом, 1 секунда. Микрофон на расстоянии ладони и "
             "чуть сбоку, чтобы воздух не бил в него."), 2.5),
        ("in_quick", say("Quick short inhale", "Быстрый короткий вдох"),
         say("A sharp catch-breath between fast lines — half a second.",
             "Резкий «подхват» воздуха между быстрыми строчками — полсекунды."), 2.0),
        ("in_deep", say("Deep inhale", "Глубокий вдох"),
         say("A slow deep breath in through the mouth, as before a long note — 1.5–2 seconds.",
             "Медленный глубокий вдох ртом, как перед длинной нотой — 1.5–2 секунды."), 3.0),
        ("in_nose", say("Inhale through the nose", "Вдох носом"),
         say("A quiet breath in through the nose, 1 second.", "Тихий вдох носом, 1 секунда."), 2.5),
        ("out_sigh", say("Sigh \"haah\" without voice", "Вздох «ха-а» без голоса"),
         say("Breathe out with a sigh, as if tired, with no voice in it — 1 second.",
             "Выдохните со вздохом, как от усталости, без звука голоса — 1 секунда."), 2.5),
        ("out_soft", say("Soft exhale after a phrase", "Мягкий выдох после фразы"),
         say("A quiet breath out through the mouth, as when you finish a line — half a second to a second.",
             "Тихий выдох ртом, как когда допели строчку, — полсекунды-секунда."), 2.0),
    ]
    for bid, title, how, sec in breaths:
        add("breath", f"breath_{bid}", title,
            how + say(" You can do it 2–3 times in a row with pauses.", " Можно сделать 2–3 раза подряд с паузами."),
            None, sec, "", need=3, prio=0.5 if bid.startswith("in") else 0.7)
    tasks.sort(key=lambda t: (t["prio"], t["id"]))
    for i, t in enumerate(tasks):
        t["order"] = i
    cats = [{"id": c, "title": d[lang][0], "about": d[lang][1]} for c, d in CATEGORIES]
    rng = {k: [v, note(v, lang)] for k, v in r.items() if k != "source"}
    return {"voice": voice, "tasks": tasks, "range": rng, "range_source": r["source"], "categories": cats}
