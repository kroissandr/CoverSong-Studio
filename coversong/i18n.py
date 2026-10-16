"""Interface languages: English (the main one) and Russian. The console and the logs are always in English."""
LANGS = ("en", "ru")
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTES_RU = ["до", "до-диез", "ре", "ре-диез", "ми", "фа", "фа-диез", "соль", "соль-диез", "ля", "ля-диез", "си"]


def lang_of(x):
    x = str(x or "").lower()[:2]
    return x if x in LANGS else "en"


def note_name(m):
    m = int(round(m))
    return f"{NOTES[m % 12]}{m // 12 - 1}"


def note(m, lang="en"):
    """A note for people: "A3" in English, "ля (A3)" in Russian."""
    m = int(round(m))
    return f"{NOTES_RU[m % 12]} ({note_name(m)})" if lang_of(lang) == "ru" else note_name(m)


def tr(table, lang, key, **kw):
    """table = {"en": {key: text}, "ru": {...}}; falls back to English."""
    s = table.get(lang_of(lang), {}).get(key)
    if s is None:
        s = table["en"][key]
    return s.format(**kw) if kw else s
