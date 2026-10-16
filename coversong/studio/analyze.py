"""Grade a take right after recording and say in plain words what to change.

Pitch comes from RMVPE (the tracker RVC uses) at 10 ms. Everything is measured against the task's target notes.
Tips are short, concrete and for someone who never studied music: "a little higher", "don't slide in from below",
"hold it steady to the end", "fuller, less air", "move back — it overloads". Songs (free singing) are graded only
on what makes a good training recording: level, clipping, noise and music in the pauses, how much is sung.
"""
import numpy as np

from ..i18n import lang_of, note

FPS = 100
SR16 = 16000
IN_TUNE = 35.0  # cents: counted as "on the note"

T = {
    "en": {
        "clip": "Overload: the sound hits the ceiling. Move a hand's width back from the microphone or turn the Gain "
                "knob on your audio interface down.",
        "very_quiet": "Very quiet. Come closer to the microphone (5–10 cm) and sing in full voice.",
        "quiet": "It came out quiet — you can come a little closer to the microphone.",
        "noise": "There is noise in the pauses — check for music in the headphones or a fan nearby.",
        "noise_song": "Noise or music is heard in the pauses. If you sing along to a track, keep the headphones "
                      "quieter and closer to your ears; turn off fans nearby.",
        "tiny": "Just a tiny bit {way} — very little.",
        "semitone": "Go {way} by about a semitone: listen to the note again and sing it \"in unison\".",
        "octave": "You sang an octave {way} — it is the same note in another register. Try exactly the pitch "
                  "you hear in the prompt.",
        "off": "Off the note: you need to go clearly {way} (by {n:.0f} semitones). Listen to the prompt and match it.",
        "lower": "lower", "higher": "higher",
        "not_note": "This was not recognised as singing on a note. Hold the vowel steadily for at least {s:.0f} "
                    "seconds.",
        "scoop_up": "You slide into the note from below. Start right on the pitch — imagine the note is already "
                    "sounding and just join it.",
        "scoop_down": "The start is higher than the note and then slides down. Start softer, right on the note.",
        "drift": "The note drifts {way} towards the end. Hold the pitch to the very end, like drawing a straight line.",
        "up": "up", "down": "down",
        "wobble": "The pitch shakes. Sing calmer, on an even breath, without a bleat in the voice.",
        "short": "Too short ({d:.1f} s). Hold the sound for at least {s:.0f} seconds.",
        "level_jumps": "The loudness jumps. Try to sing equally loud from start to end.",
        "too_bright": "It came out too ringing. Here it should be softer, with air — as if humming at night so as "
                      "not to wake anyone.",
        "airy": "There is a lot of air in the sound. Sing fuller — as if calling someone from the next room, "
                "but without shouting.",
        "falsetto": "It sounds too light, almost falsetto. Chest voice is needed here: like confident loud speech, "
                    "the sound \"from the chest\".",
        "strained": "The voice is strained at this height. Try lighter, brighter, \"in the head\" — like a surprised "
                    "\"oo?\" — without pressure in the throat.",
        "good_note": "Steady and right on the note. That's it!",
        "no_glide": "The siren was not recognised. Slide your voice smoothly on a vowel for 2–3 seconds.",
        "no_top": "Not enough top: reach {n}.",
        "no_bottom": "Not enough bottom: start/finish on {n}.",
        "break": "The voice breaks around {n}. Go through that place slower and a little quieter — the transition "
                 "will become smooth.",
        "uneven": "The line is uneven — move the voice one way, without going back, like a lift.",
        "good_glide": "Smooth and without breaks. Excellent!",
        "no_melody": "The melody was not recognised. Sing each note separately and a little longer.",
        "note_sharp": "Note {i} is a bit high (it should be {n}).",
        "note_flat": "Note {i} is a bit low (it should be {n}).",
        "missed": "Notes missed: {n}. Sing each note separately, without merging them.",
        "good_pattern": "All the notes are in place. Excellent!",
        "no_stacc": "No short \"{t}\" on the note is heard — four separate sounds with pauses.",
        "stacc_few": "{n} sounds instead of 4 — make short pauses between the \"{t}\".",
        "stacc_many": "The sounds merge or there are too many — four short ones, with pauses.",
        "stacc_spread": "The sounds are at different pitches — all four should be on one note.",
        "good_stacc": "Clear and detached. Excellent!",
        "br_quiet": "The breath is hardly heard — breathe a bit louder or closer to the microphone (but from the side).",
        "br_noise": "The breath drowns in the room noise — closer to the microphone, turn the fan off.",
        "br_voice": "Voice is heard — only air is needed here, no sound.",
        "br_pop": "Air hits the microphone (rumble, puffing) — turn a little to the side.",
        "good_breath": "A clean, clearly heard breath. Excellent!",
        "shout_quiet": "Quiet for a shout — brighter and louder (but moved back from the microphone).",
        "good_shout": "Bright and clean. Excellent!",
        "song_short": "Only {s:.0f} s of singing — sing at least a minute (a whole song is best).",
        "song_shortish": "{s:.0f} s of singing — fine, but a whole song (2–4 minutes) gives the model more.",
        "song_little_voice": "Singing takes only {p:.0%} of the recording — long silences or talking. "
                             "Fine, but it is better to sing more.",
        "song_range": "Sung in this take: {lo}–{hi}, mostly around {mid}.",
        "song_good": "A clean recording — good for training.",
        "v_great": "Excellent — let's save it", "v_ok": "Not bad — save it or try again", "v_bad": "Let's try again",
    },
    "ru": {
        "clip": "Перегруз: звук упирается в потолок. Отодвиньтесь от микрофона на ладонь или убавьте ручку Gain "
                "на звуковой карте.",
        "very_quiet": "Очень тихо. Подойдите к микрофону ближе (5–10 см) и пойте в полный голос.",
        "quiet": "Тихо записалось — можно подойти чуть ближе к микрофону.",
        "noise": "В паузах слышен шум — проверьте, нет ли музыки в наушниках, вентилятора рядом.",
        "noise_song": "В паузах слышен шум или музыка. Если поёте под песню — сделайте наушники тише и плотнее к "
                      "ушам; выключите вентиляторы рядом.",
        "tiny": "Чуть-чуть {way} — совсем немного.",
        "semitone": "Возьмите {way} примерно на полтона: попробуйте ещё раз послушать ноту и спеть её «в унисон».",
        "octave": "Вы спели на октаву {way} — это та же нота, но в другом регистре. Попробуйте именно ту высоту, "
                  "что звучит в подсказке.",
        "off": "Мимо ноты: нужно заметно {way} (на {n:.0f} полутона). Послушайте подсказку и подстройтесь голосом.",
        "lower": "ниже", "higher": "выше",
        "not_note": "Звук не распознан как пение на ноте. Тяните гласную ровно хотя бы {s:.0f} секунды.",
        "scoop_up": "Вы заходите на ноту снизу («подъезд»). Начинайте сразу с нужной высоты — представьте, что нота "
                    "уже звучит, и просто включитесь в неё.",
        "scoop_down": "Начало звучит выше нужного и потом съезжает. Начинайте мягче, сразу на ноте.",
        "drift": "К концу нота уплывает {way}. Держите высоту до самого конца, как будто ведёте ровную линию.",
        "up": "вверх", "down": "вниз",
        "wobble": "Высота дрожит. Пойте спокойнее, на ровном выдохе, без «барашка» в голосе.",
        "short": "Слишком коротко ({d:.1f} с). Тяните звук не меньше {s:.0f} секунд.",
        "level_jumps": "Громкость скачет. Старайтесь петь одинаково громко от начала до конца.",
        "too_bright": "Получилось слишком «звонко». Здесь нужно мягче, с воздухом — как будто напеваете ночью, "
                      "чтобы никого не разбудить.",
        "airy": "В звуке много воздуха. Пойте полнее — как будто зовёте человека из соседней комнаты, но не кричите.",
        "falsetto": "Звучит слишком легко, почти фальцетом. Здесь нужен грудной голос: как уверенная громкая речь, "
                    "звук «из груди».",
        "strained": "На этой высоте голос напряжён. Попробуйте легче, светлее, «головой» — как удивлённое «у?» — "
                    "без давления в горле.",
        "good_note": "Ровно и точно в ноту. Так и надо!",
        "no_glide": "Сирена не распознана. Проведите голосом плавно на гласной 2–3 секунды.",
        "no_top": "Не хватило верха: доезжайте до {n}.",
        "no_bottom": "Не хватило низа: начинайте/заканчивайте на {n}.",
        "break": "Голос «ломается» около {n}. Пройдите это место медленнее и чуть тише — переход станет плавным.",
        "uneven": "Линия неровная — ведите голос в одну сторону, без возвратов, как лифт.",
        "good_glide": "Плавно и без срывов. Отлично!",
        "no_melody": "Мелодия не распознана. Пойте каждую ноту отдельно и чуть дольше.",
        "note_sharp": "{i}-я нота высоковата (нужна {n}).",
        "note_flat": "{i}-я нота низковата (нужна {n}).",
        "missed": "Пропущено нот: {n}. Пойте каждую ноту отдельно, не сливая их.",
        "good_pattern": "Все ноты на месте. Отлично!",
        "no_stacc": "Не слышно коротких «{t}» на ноте — четыре отдельных звука с паузами.",
        "stacc_few": "Слышно {n} звука вместо 4 — делайте короткие паузы между «{t}».",
        "stacc_many": "Звуки сливаются или их слишком много — четыре коротких, с паузами.",
        "stacc_spread": "Звуки на разной высоте — все четыре должны быть на одной ноте.",
        "good_stacc": "Чётко и отрывисто. Отлично!",
        "br_quiet": "Вдох почти не слышен — дышите чуть громче или ближе к микрофону (но сбоку).",
        "br_noise": "Вдох тонет в шуме комнаты — ближе к микрофону, выключите вентилятор.",
        "br_voice": "Слышен голос — здесь нужен только воздух, без звука.",
        "br_pop": "Воздух бьёт в микрофон (гул, «пыхтение») — отвернитесь чуть в сторону.",
        "good_breath": "Чистый, хорошо слышный вдох. Отлично!",
        "shout_quiet": "Для выкрика тихо — ярче и громче (но отодвинувшись от микрофона).",
        "good_shout": "Ярко и чисто. Отлично!",
        "song_short": "Пения всего {s:.0f} с — спойте хотя бы минуту (лучше всего — песню целиком).",
        "song_shortish": "Пения {s:.0f} с — неплохо, но песня целиком (2–4 минуты) даст модели больше.",
        "song_little_voice": "Пение занимает только {p:.0%} записи — долгие паузы или разговоры. Можно сохранить, "
                             "но лучше петь больше.",
        "song_range": "В этой записи спето: {lo}–{hi}, чаще всего около {mid}.",
        "song_good": "Чистая запись — подходит для обучения.",
        "v_great": "Отлично — сохраняем", "v_ok": "Неплохо — можно сохранить или попробовать ещё раз",
        "v_bad": "Давайте ещё раз",
    },
}


class Tips(list):
    """Tips in the take's language: tips.add("warn", "key", n=…)."""

    def __init__(self, lang):
        super().__init__()
        self.lang = lang_of(lang)

    def t(self, key, **kw):
        s = T[self.lang].get(key) or T["en"][key]
        return s.format(**kw) if kw else s

    def add(self, kind, key, **kw):
        self.append((kind, self.t(key, **kw)))

    def note(self, m):
        return note(m, self.lang)


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def frames_db(x, sr):
    hop = sr // FPS
    n = len(x) // hop
    fr = x[: n * hop].reshape(n, hop)
    return 10 * np.log10((fr ** 2).mean(1) + 1e-12)


def harmonic_stats(x, sr, f0, idx):
    """Clarity (harmonics vs. what lies between them, 2–4 kHz) and H1–H2 (light/breathy vs. chest) on given frames."""
    n = 4096
    hop = sr // FPS
    clar, h12 = [], []
    win = np.hanning(n)
    f = np.fft.rfftfreq(n, 1 / sr)
    df = f[1]
    for i in idx[::2]:
        a = i * hop - n // 2
        if a < 0 or a + n > len(x) or f0[i] <= 0:
            continue
        S = np.abs(np.fft.rfft(x[a:a + n] * win)) ** 2
        F = f0[i]
        ks = np.arange(np.ceil(2000 / F), np.floor(4000 / F))
        if len(ks) >= 2:
            hb = np.round(ks * F / df).astype(int)
            ib = np.round((ks + 0.5) * F / df).astype(int)
            clar.append(10 * np.log10(S[hb].mean() / (S[ib].mean() + 1e-20)))

        def peak(fr):
            b = int(round(fr / df))
            return S[max(0, b - 2):b + 3].max()
        h12.append(10 * np.log10(peak(F) / (peak(2 * F) + 1e-20)))
    return (float(np.median(clar)) if clar else None), (float(np.median(h12)) if h12 else None)


def base_metrics(x, sr, f0):
    db = frames_db(x, sr)
    n = min(len(db), len(f0))
    db, f0 = db[:n], f0[:n]
    loud = db > db.max() - 30
    voiced = (f0 > 40) & loud
    peak = float(20 * np.log10(np.abs(x).max() + 1e-9))
    clip = float((np.abs(x) > 0.99).mean() * 100)
    noise = float(np.percentile(db, 5))
    return {"db": db, "f0": f0, "voiced": voiced, "peak_db": peak, "clip_pct": clip, "noise_db": noise,
            "voiced_s": float(voiced.sum() / FPS)}


def note_hist(f0, voiced=None):
    """Seconds sung per MIDI note (voiced frames)."""
    f0 = np.asarray(f0, np.float64)
    v = (f0 > 40) if voiced is None else (voiced & (f0 > 40))
    if not v.any():
        return {}
    m = np.round(69 + 12 * np.log2(f0[v] / 440.0)).astype(int)
    keys, cnt = np.unique(m, return_counts=True)
    return {int(k): float(c) / FPS for k, c in zip(keys, cnt)}


def level_tips(m, task, tips):
    ok = True
    if m["clip_pct"] > 0.02 or m["peak_db"] > -0.3:
        tips.add("bad", "clip")
        ok = False
    elif m["peak_db"] < -30:
        tips.add("bad", "very_quiet")
        ok = False
    elif m["peak_db"] < -20 and task["cat"] not in ("soft",):
        tips.add("warn", "quiet")
    if m["noise_db"] > -50:
        tips.add("warn", "noise_song" if task["cat"] == "songs" else "noise")
    return ok


def steady_part(voiced, start_skip=0.2, end_skip=0.1):
    idx = np.flatnonzero(voiced)
    if len(idx) < 20:
        return idx, idx[:0]
    # the longest run of voiced frames (gaps under 60 ms are bridged)
    runs, a = [], idx[0]
    for p, q in zip(idx[:-1], idx[1:]):
        if q - p > 6:
            runs.append((a, p))
            a = q
    runs.append((a, idx[-1]))
    a, b = max(runs, key=lambda r: r[1] - r[0])
    core = np.arange(a, b + 1)
    s = core[int(start_skip * FPS):len(core) - int(end_skip * FPS)]
    return core, s[voiced[s]] if len(s) else s


def pitch_word(d, tips):
    """Plain words for a pitch error in cents (positive = sharp)."""
    a = abs(d)
    if a < IN_TUNE:
        return None
    way = tips.t("lower" if d > 0 else "higher")
    if a < 70:
        return tips.t("tiny", way=way)
    if a < 150:
        return tips.t("semitone", way=way)
    if a > 900:
        return tips.t("octave", way=tips.t("higher" if d > 0 else "lower"))
    return tips.t("off", way=way, n=a / 100)


def grade_note(x, sr, f0, task, tips):
    m = base_metrics(x, sr, f0)
    level_ok = level_tips(m, task, tips)
    target = float(task["target"])
    need_s = task["seconds"] * 0.6
    core, st = steady_part(m["voiced"])
    curve_t = (np.arange(len(m["f0"])) / FPS).tolist()
    cents_all = np.where(m["f0"] > 40, 1200 * np.log2(np.maximum(m["f0"], 1) / hz(target)), np.nan)
    res = {"curve": {"t": curve_t, "cents": [None if np.isnan(c) else float(np.clip(c, -700, 700)) for c in cents_all]},
           "peak_db": m["peak_db"], "voiced_s": m["voiced_s"], "hist": note_hist(m["f0"], m["voiced"])}
    if len(st) < 30:
        tips.add("bad", "not_note", s=need_s)
        return dict(res, score=0)
    c = cents_all[st]
    c = c[~np.isnan(c)]
    # octave slips are judged as such, not as a huge error
    med = float(np.median(c))
    oct_off = 1200 * round(med / 1200)
    if oct_off and task["cat"] != "shout":
        tips.append(("bad", pitch_word(med, tips)))
    c_rel = c - oct_off
    med = float(np.median(c_rel))
    in_tune = float((np.abs(c_rel) < IN_TUNE).mean())
    smooth = np.convolve(c_rel, np.ones(15) / 15, mode="same")
    wobble = float(np.median(np.abs(c_rel - smooth)) * 1.48)
    drift = float(np.median(c_rel[-30:]) - np.median(c_rel[:30])) if len(c_rel) > 80 else 0.0
    head = cents_all[core[:15]] - oct_off
    head = head[~np.isnan(head)]
    scoop = float(np.median(head) - med) if len(head) > 5 else 0.0
    dur = len(core) / FPS
    clarity, h12 = harmonic_stats(x, sr, m["f0"], st)
    lvl = m["db"][st]
    lvl_jump = float(np.percentile(lvl, 90) - np.percentile(lvl, 10))

    w = pitch_word(med, tips)
    if w and not oct_off:
        tips.append(("bad" if abs(med) > 70 else "warn", w))
    if scoop < -80:
        tips.add("warn", "scoop_up")
    elif scoop > 80:
        tips.add("warn", "scoop_down")
    if abs(drift) > 40 and task["cat"] != "shout":
        tips.add("warn", "drift", way=tips.t("up" if drift > 0 else "down"))
    if wobble > 25 and task["cat"] != "shout":
        tips.add("warn", "wobble")
    if dur < need_s and task["cat"] != "shout":
        tips.add("bad", "short", d=dur, s=need_s)
    if lvl_jump > 8 and task["cat"] in ("note", "hum"):
        tips.add("warn", "level_jumps")
    if task["cat"] == "soft":
        if clarity is not None and clarity > 18:
            tips.add("warn", "too_bright")
    elif clarity is not None and clarity < 10 and task["cat"] in ("note", "hum", "english"):
        tips.add("warn", "airy")
    reg = task.get("register")
    if h12 is not None and reg and level_ok and task["cat"] in ("note", "hum"):
        if reg == "chest" and h12 > 9:
            tips.add("warn", "falsetto")
        elif reg == "head" and h12 < 0 and target >= 64:
            tips.add("warn", "strained")
    pitch_ok = 0.5 * in_tune + 0.5 * max(0.0, 1 - max(0.0, abs(med) - 10) / 80)
    score = 100 * (0.45 * pitch_ok + 0.2 * max(0, 1 - wobble / 40) + 0.15 * min(1, dur / task["seconds"]) +
                   0.1 * (1 if abs(drift) <= 40 else 0.4) + 0.1 * (1 if scoop >= -80 else 0.4))
    if oct_off and task["cat"] != "shout":
        score *= 0.5
    if dur < need_s and task["cat"] != "shout":
        score = min(score, 55)
    if not level_ok:
        score = min(score, 40)
    if not tips:
        tips.add("good", "good_note")
    return dict(res, score=round(score), stats={"cents": med, "in_tune": in_tune, "wobble": wobble, "scoop": scoop,
                                                "drift": drift, "clarity": clarity, "h1h2": h12, "seconds": dur})


def grade_glide(x, sr, f0, task, tips):
    m = base_metrics(x, sr, f0)
    level_ok = level_tips(m, task, tips)
    a, b = task["target"]
    v = np.flatnonzero(m["voiced"])
    midi = np.where(m["f0"] > 40, 69 + 12 * np.log2(np.maximum(m["f0"], 1) / 440), np.nan)
    res = {"curve": {"t": (np.arange(len(midi)) / FPS).tolist(),
                     "midi": [None if np.isnan(q) else float(q) for q in midi], "from": a, "to": b},
           "peak_db": m["peak_db"], "voiced_s": m["voiced_s"], "hist": note_hist(m["f0"], m["voiced"])}
    if len(v) < 40:
        tips.add("bad", "no_glide")
        return dict(res, score=0)
    seg = midi[v[0]:v[-1] + 1]
    ok = ~np.isnan(seg)
    s0, s1 = np.nanmedian(seg[:15]), np.nanmedian(seg[-15:])
    up = b > a
    covered = (min(s0, s1) <= min(a, b) + 1.0) and (max(s0, s1) >= max(a, b) - 1.0)
    if not covered:
        if (up and s1 < b - 1) or (not up and s0 < a - 1):
            tips.add("warn", "no_top", n=tips.note(max(a, b)))
        if (up and s0 > a + 1) or (not up and s1 > b + 1):
            tips.add("warn", "no_bottom", n=tips.note(min(a, b)))
    steps = np.diff(seg[ok])
    jumps = np.flatnonzero(np.abs(steps) > 2.5)
    gaps = np.flatnonzero(~ok)
    breaks = len(jumps) + (1 if len(gaps) > 8 else 0)
    if breaks:
        where = seg[ok][jumps[0]] if len(jumps) else np.nanmedian(seg)
        tips.add("warn", "break", n=tips.note(where))
    mono = float((np.sign(steps) == (1 if up else -1)).mean()) if len(steps) else 0.0
    if mono < 0.6:
        tips.add("warn", "uneven")
    score = 100 * (0.4 * covered + 0.3 * min(1, mono / 0.8) + 0.3 * (1 if not breaks else 0.3))
    if not level_ok:
        score = min(score, 40)
    if not tips:
        tips.add("good", "good_glide")
    return dict(res, score=round(score))


def plateaus(midi, min_len=10):
    """Stable stretches (each note of a melody): runs where pitch stays within ±0.6 semitone for ≥100 ms."""
    out, i, n = [], 0, len(midi)
    while i < n:
        if np.isnan(midi[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and not np.isnan(midi[j + 1]) and abs(midi[j + 1] - np.median(midi[i:j + 1])) < 0.6:
            j += 1
        if j - i + 1 >= min_len:
            out.append((i, j, float(np.median(midi[i:j + 1]))))
        i = j + 1
    return out


def grade_pattern(x, sr, f0, task, tips):
    m = base_metrics(x, sr, f0)
    level_ok = level_tips(m, task, tips)
    target = list(task["target"])
    midi = np.where(m["voiced"], 69 + 12 * np.log2(np.maximum(m["f0"], 1) / 440), np.nan)
    res = {"curve": {"t": (np.arange(len(midi)) / FPS).tolist(),
                     "midi": [None if np.isnan(q) else float(q) for q in midi], "notes": target},
           "peak_db": m["peak_db"], "voiced_s": m["voiced_s"], "hist": note_hist(m["f0"], m["voiced"])}
    pl = plateaus(midi)
    if len(pl) < max(2, len(target) // 2):
        tips.add("bad", "no_melody")
        return dict(res, score=0)
    sung = np.array([p[2] for p in pl])
    # align sung notes to the target melody (dynamic programming, notes may be missed or extra)
    nT, nS = len(target), len(sung)
    D = np.full((nT + 1, nS + 1), 1e9)
    D[0, :] = np.arange(nS + 1) * 1.5
    D[:, 0] = np.arange(nT + 1) * 1.5
    for i in range(1, nT + 1):
        for j in range(1, nS + 1):
            D[i, j] = min(D[i - 1, j - 1] + min(abs(sung[j - 1] - target[i - 1]), 3), D[i - 1, j] + 1.5,
                          D[i, j - 1] + 1.5)
    i, j, pairs = nT, nS, []
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + min(abs(sung[j - 1] - target[i - 1]), 3):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1.5:
            i -= 1
        else:
            j -= 1
    pairs = pairs[::-1]
    errs = {ti: (sung[sj] - target[ti]) * 100 for ti, sj in pairs}
    missed = nT - len(pairs)
    good = sum(1 for e in errs.values() if abs(e) < 50)
    for ti in range(nT):
        if ti in errs and abs(errs[ti]) >= 50:
            tips.add("warn", "note_sharp" if errs[ti] > 0 else "note_flat", i=ti + 1, n=tips.note(target[ti]))
    if missed:
        tips.add("warn", "missed", n=missed)
    del tips[5:]
    res["curve"]["sung"] = [{"t0": p[0] / FPS, "t1": p[1] / FPS, "midi": p[2]} for p in pl]
    score = 100 * good / nT
    if not level_ok:
        score = min(score, 40)
    if not tips:
        tips.add("good", "good_pattern")
    return dict(res, score=round(score))


def grade_staccato(x, sr, f0, task, tips):
    m = base_metrics(x, sr, f0)
    level_ok = level_tips(m, task, tips)
    target = float(task["target"])
    db = m["db"]
    on = db > db.max() - 25
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    bursts = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= 6]
    cents_all = np.where(m["f0"] > 40, 1200 * np.log2(np.maximum(m["f0"], 1) / hz(target)), np.nan)
    res = {"curve": {"t": (np.arange(len(cents_all)) / FPS).tolist(),
                     "cents": [None if np.isnan(c) else float(np.clip(c, -700, 700)) for c in cents_all]},
           "peak_db": m["peak_db"], "voiced_s": m["voiced_s"], "hist": note_hist(m["f0"], m["voiced"])}
    meds = []
    for a, b in bursts:
        c = cents_all[a + 2:b - 1]
        c = c[~np.isnan(c)]
        if len(c) >= 3:
            meds.append(float(np.median(c)))
    if not meds:
        tips.add("bad", "no_stacc", t=task["text"])
        return dict(res, score=0)
    med = float(np.median(meds))
    w = pitch_word(med, tips)
    if w:
        tips.append(("warn", w))
    if len(bursts) < 3:
        tips.add("warn", "stacc_few", n=len(bursts), t=task["text"])
    elif len(bursts) > 6:
        tips.add("warn", "stacc_many")
    spread = float(np.std(meds)) if len(meds) > 1 else 0.0
    if spread > 50:
        tips.add("warn", "stacc_spread")
    count_ok = 3 <= len(bursts) <= 6
    score = 100 * (0.5 * max(0.0, 1 - max(0.0, abs(med) - 15) / 80) + 0.25 * count_ok + 0.25 * max(0, 1 - spread / 80))
    if not level_ok:
        score = min(score, 40)
    if not tips:
        tips.add("good", "good_stacc")
    return dict(res, score=round(score))


def grade_breath(x, sr, f0, task, tips):
    """A breath take: heard clearly above the room, without voice, not blown into the microphone."""
    m = base_metrics(x, sr, f0)
    level_ok = level_tips(m, task, tips)
    db = m["db"]
    act = db > db.max() - 25
    breath_s = float(act.sum() / FPS)
    snr = float(np.percentile(db, 95) - m["noise_db"])
    voiced = float(m["voiced"].sum() / max(1, act.sum()))
    lo = (np.abs(np.fft.rfft(x)) ** 2)
    fq = np.fft.rfftfreq(len(x), 1 / sr)
    pop = float(10 * np.log10(lo[(fq > 20) & (fq < 150)].sum() / (lo[(fq >= 150) & (fq < 8000)].sum() + 1e-20)))
    score = 100.0
    if breath_s < 0.15:
        tips.add("bad", "br_quiet")
        score -= 50
    if snr < 20:
        tips.add("warn", "br_noise")
        score -= 25
    if voiced > 0.25:
        tips.add("warn", "br_voice")
        score -= 30
    if pop > -6:
        tips.add("warn", "br_pop")
        score -= 25
    if not level_ok:
        score = min(score, 40)
    if not tips:
        tips.add("good", "good_breath")
    return {"score": round(max(0.0, score)), "voiced_s": breath_s, "peak_db": m["peak_db"], "curve": {"t": []},
            "hist": {}}


def grade_song(x, sr, f0, task, tips):
    """Free singing: is it a good training recording? Level, clipping, noise or music in the pauses, how much is
    sung. The pitch line is shown, nothing is "wrong" about the notes."""
    m = base_metrics(x, sr, f0)
    level_ok = level_tips(m, task, tips)
    sung = m["voiced_s"]
    dur = len(x) / sr
    score = 100.0
    if sung < 30:
        tips.add("bad", "song_short", s=sung)
        score -= 60
    elif sung < 60:
        tips.add("warn", "song_shortish", s=sung)
        score -= 15
    share = sung / max(dur, 1e-6)
    if dur > 20 and share < 0.3:
        tips.add("warn", "song_little_voice", p=share)
        score -= 10
    if m["noise_db"] > -50:
        score -= 20
    if m["clip_pct"] > 0.02:
        score -= 20
    hist = note_hist(m["f0"], m["voiced"])
    if hist:
        keys = sorted(hist)
        tot = sum(hist.values())
        acc, q = 0.0, {}
        for k in keys:
            acc += hist[k]
            for p in (5, 50, 95):
                if p not in q and acc >= p / 100 * tot:
                    q[p] = k
        tips.add("good", "song_range", lo=tips.note(q[5]), hi=tips.note(q[95]), mid=tips.note(q[50]))
    if not level_ok:
        score = min(score, 40)
    if score >= 80:
        tips.add("good", "song_good")
    step = 5  # 20 points a second is enough to see the melody of a whole song
    midi = np.where(m["voiced"], 69 + 12 * np.log2(np.maximum(m["f0"], 1) / 440), np.nan)[::step]
    curve = {"t": (np.arange(len(midi)) * step / FPS).tolist(),
             "midi": [None if np.isnan(q_) else float(q_) for q_ in midi], "free": True}
    return {"score": round(max(0.0, score)), "voiced_s": sung, "peak_db": m["peak_db"], "curve": curve, "hist": hist}


def grade(x, sr, f0, task, lang="en"):
    """{score 0–100, tips [(good|warn|bad, text)], curve, hist {midi: s}, voiced_s, peak_db}."""
    tips = Tips(lang)
    cat = task["cat"]
    if cat == "songs":
        out = grade_song(x, sr, f0, task, tips)
    elif cat == "breath":
        out = grade_breath(x, sr, f0, task, tips)
    elif cat == "glide":
        out = grade_glide(x, sr, f0, task, tips)
    elif cat == "pattern" or (cat == "hum" and isinstance(task["target"], list)):
        out = grade_pattern(x, sr, f0, task, tips)
    elif cat == "staccato":
        out = grade_staccato(x, sr, f0, task, tips)
    elif cat == "shout":
        out = grade_note(x, sr, f0, dict(task, seconds=1.0), tips)
        if out.get("peak_db", -99) < -12 and out["score"] > 0:
            tips.insert(0, ("warn", tips.t("shout_quiet")))
        if not tips:
            tips.add("good", "good_shout")
    else:
        out = grade_note(x, sr, f0, task, tips)
    out["tips"] = list(tips)
    return out


def verdict(score, lang="en"):
    t = T[lang_of(lang)]
    return t["v_great"] if score >= 80 else t["v_ok"] if score >= 60 else t["v_bad"]
