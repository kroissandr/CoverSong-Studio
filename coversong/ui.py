"""CoverSong Studio — the covers interface (Gradio 6), in English and Russian. The app (coversong/app.py) serves it at
"/" (English) and "/ru" (Russian), next to the vocal studio at "/studio".
"""
import os
import threading
import time
import traceback

import gradio as gr

from . import __version__
from .config import load_config
from .i18n import tr
from .util import LOG_HOOKS

RUN_LOCK = threading.Lock()  # one cover job at a time on the GPU (the English and Russian pages share it)

# The look of the vocal studio (coversong/studio/static/style.css): warm paper background, white cards, one red
# accent, system font — for the whole app.
PAL = {"bg": ("#f6f4ef", "#151413"), "card": ("#ffffff", "#1f1d1b"), "ink": ("#1d1b18", "#f1ede6"),
       "muted": ("#6f6a61", "#a39d93"), "line": ("#e4dfd5", "#34312d"), "soft": ("#efebe3", "#2a2825"),
       "input": ("#fcfbf8", "#24221f"), "accent": ("#d2452f", "#e0573f"), "accent_hover": ("#b93a26", "#ef6a52"),
       "accent_soft": ("#f8e2dd", "#3a2420")}

CSS = """
.gradio-container {max-width: none !important; width: 100% !important; margin: 0 !important}
.gradio-container > .main {max-width: 1440px !important; width: 100% !important; margin: 0 auto !important;
  padding: 8px 28px 32px !important}
footer {display: none !important}
#sc-head {padding: 18px 0 6px}
.sc-head {display: flex; gap: 24px; align-items: flex-end; justify-content: space-between; flex-wrap: wrap}
.sc-head h1 {margin: 0; font-size: 28px; letter-spacing: -.01em; color: var(--body-text-color)}
.sc-head p {margin: 4px 0 0; color: var(--body-text-color-subdued)}
.sc-head .lang a {color: var(--body-text-color-subdued); font-weight: 600; text-decoration: none; font-size: 14px}
.sc-head .lang a.active {color: var(--body-text-color); text-decoration: underline}
.sc-head .lang {color: var(--body-text-color-subdued)}
/* tabs as the studio's category buttons */
.tab-wrapper {height: auto !important; padding-bottom: 0 !important; margin-bottom: 8px !important}
.tab-container:not(.visually-hidden) {height: auto !important; overflow: visible !important; border-bottom: none !important;
  gap: 8px !important; padding: 3px 3px 14px !important}
button[role="tab"] {border: 1px solid var(--border-color-primary) !important; border-radius: 10px !important;
  background: var(--block-background-fill) !important; padding: 9px 16px !important; font-size: 15px !important;
  color: var(--body-text-color) !important; margin-right: 8px !important}
button[role="tab"][aria-selected="true"] {border-color: var(--color-accent) !important;
  box-shadow: 0 0 0 2px var(--color-accent-soft) !important; font-weight: 600 !important}
button[role="tab"][aria-selected="true"]::after, button[role="tab"]::after {display: none !important}
#go {min-height: 56px; font-size: 18px; font-weight: 650}
.sc-frame {width: 100%; height: 900px; border: 0; display: block; background: transparent}
.sc-sub {color: var(--body-text-color-subdued); margin: 0 0 6px}
@media (max-width: 700px) { .gradio-container > .main {padding: 4px 12px 24px !important} }
"""

# the header's language links keep the open tab (#tab-…) and switch the page in the same window
HEAD_JS = r"""
const tabs = () => [...document.querySelectorAll('button[role="tab"]')];
element.querySelectorAll('.lang a').forEach(a => a.addEventListener('click', e => {
  e.preventDefault();
  const i = tabs().findIndex(b => b.getAttribute('aria-selected') === 'true');
  window.location.assign(a.getAttribute('href') + (i > 0 ? '#tab' + i : ''));
}));
const m = /^#tab(\d+)$/.exec(window.location.hash);
if (m) { let n = 0; const go = () => { const b = tabs()[+m[1]]; if (b) b.click(); else if (n++ < 40) setTimeout(go, 100); }; go(); }
"""
# a studio page inside a tab: its height follows its content (the page posts it, coversong/studio/static/embed.js)
FRAME_JS = r"""
const f = element.querySelector('iframe');
window.addEventListener('message', e => {
  if (f && e.source === f.contentWindow && e.data && e.data.scHeight) f.style.height = Math.max(420, e.data.scHeight) + 'px';
});
"""

TXT = {
    "en": {
        "subtitle": "A cover in your voice, mixed like the original",
        "links": '<a href="/studio/?lang=en">🎙 Vocal Studio — record and train your voice</a> · <a href="/ru/">Русский</a>',
        "tab_one": "One voice", "tab_multi": "Several voices", "tab_studio": "Vocal Studio", "tab_library": "Library",
        "songs": "Songs (mp3, flac, wav, m4a…) — several at once is fine",
        "links_box": "…and/or YouTube links — one per line",
        "model": "Voice model",
        "model_info": "Folders in rvc_models. A new model: put its folder with the .pth and .index there, then ↻",
        "pitch": "Voice key", "pitch_info": "Auto — from your range. −1 — if a woman sings and your voice is male.",
        "p_auto": "Auto", "p_down": "−1 octave", "p_none": "No shift", "p_up": "+1 octave",
        "voc": "Wordless singing (oo, aa)", "voc_soft": "In your voice", "voc_remove": "Remove",
        "reverb": "Reverb", "reverb_info": "1 — like the original, 0 — dry",
        "vocal_db": "Voice level, dB", "vocal_db_info": "0 — like the vocal in the original",
        "choir_acc": "Choir in the backing vocals", "choir": "Where to sing as a choir",
        "choir_info": "Auto — where the original has a chord and in the backing's wordless parts. Your voice is sung "
                      "in several voices and mixed into a choir.",
        "c_auto": "Auto", "c_chords": "Chords only", "c_manual": "Only by hand", "c_off": "Off",
        "choir_at": "Choir by hand (time)", "choir_at_info": "Places where there surely is a choir — added to those found",
        "crowd": "A crowd of your voice (test)",
        "crowd_info": "Where a crowd sings in the original (unclear, many voices) — 6 more takes of your voice, spread "
                      "in pitch, time and panorama",
        "crowd_at": "Crowd by hand (time)",
        "go": "🎤 Make covers",
        "status0": "Add one or more songs and press \"Make covers\" — they go one after another. Running again with "
                   "other sliders only redoes the mix — that is quick.",
        "player": "Cover", "downloads": "Download (MP3 and WAV)", "open": "📂 Open the result folder",
        "stems": "Separate tracks", "st_v": "Your vocal", "st_b": "Backing vocals", "st_i": "Instrumental",
        "log": "Log", "log_box": "Progress",
        "need_songs": "Add songs: files and/or links (one per line)", "need_model": "Choose a voice model",
        "no_models": "No voice models yet: train one in the Vocal Studio, or put a model folder into rvc_models and "
                     "press ↻",
        "h_num": "#", "h_song": "Song", "h_res": "Result", "h_intel": "Intelligibility", "h_tone": "Tone difference",
        "working": "⏳ working", "queued": "queued",
        "song_n": "⏳ Song {i} of {n} · model {m} · {s:.0f} s", "work1": "⏳ Working… model {m} · {s:.0f} s",
        "done": "✅ Done: {k} of {n} in {s:.0f} s · model {m}", "errors": " · errors: {e} (details in the log and in "
                                                                          "the logs folder)",
        "none_ok": "❌ Not a single song worked out.", "folder": "Folder: `{d}`",
        "each_folder": "Each song is in its own folder inside song_output.",
        "s": "s", "db": "dB",
        "oct": {None: "", 0: "no shift", -1: "−1 oct.", 1: "+1 oct.", -2: "−2 oct.", 2: "+2 oct."},
        "studio_md": "### 🎙 Vocal Studio\nRecord your voice in the browser and train your own voice model:\n\n"
                     "1. Create a voice (the model you record for).\n"
                     "2. **Songs** — sing 5–10 songs you like, whole (in headphones if you sing along to a track).\n"
                     "3. Optionally: short exercises (steady notes, sirens, runs, breaths) — they fill the gaps in "
                     "your range.\n"
                     "4. Press **Train** — after training the model appears in the list on the covers tabs.\n\n"
                     "The browser will ask for the microphone the first time.",
        "studio_btn": "Open the Vocal Studio",
        "footer": "CoverSong Studio v{v} · use only voices you have the right to use",
    },
    "ru": {
        "subtitle": "Кавер вашим голосом со сведением как в оригинале",
        "links": '<a href="/studio/?lang=ru">🎙 Вокальная студия — запись и обучение голоса</a> · <a href="/">English</a>',
        "tab_one": "Один голос", "tab_multi": "Несколько голосов", "tab_studio": "Вокальная студия",
        "tab_library": "Библиотека",
        "songs": "Песни (mp3, flac, wav, m4a…) — можно сразу несколько",
        "links_box": "…и/или ссылки на YouTube — по одной в строке",
        "model": "Голосовая модель",
        "model_info": "Папки в rvc_models. Новая модель: папка с .pth и .index туда, затем ↻",
        "pitch": "Тональность голоса",
        "pitch_info": "Авто — по вашему диапазону. −1 — если поёт женщина, а у вас мужской голос.",
        "p_auto": "Авто", "p_down": "−1 октава", "p_none": "Без сдвига", "p_up": "+1 октава",
        "voc": "Распевы без слов (у-у, а-а)", "voc_soft": "Вашим голосом", "voc_remove": "Убрать",
        "reverb": "Реверберация", "reverb_info": "1 — как в оригинале, 0 — сухо",
        "vocal_db": "Громкость голоса, дБ", "vocal_db_info": "0 — как вокал в оригинале",
        "choir_acc": "Хор в бэк-вокале", "choir": "Где петь хором",
        "choir_info": "Авто — где в оригинале аккорд и в распевах бэка без слов. Ваш голос поётся несколькими "
                      "голосами и сводится в хор.",
        "c_auto": "Авто", "c_chords": "Только аккорды", "c_manual": "Только вручную", "c_off": "Выкл.",
        "choir_at": "Хор вручную (время)", "choir_at_info": "Места, где хор точно есть — добавятся к найденным",
        "crowd": "Толпа из вашего голоса (тест)",
        "crowd_info": "Где в оригинале поёт толпа (неразборчиво, много голосов) — ещё 6 дублей вашего голоса, "
                      "разбросанных по высоте, времени и панораме",
        "crowd_at": "Толпа вручную (время)",
        "go": "🎤 Сделать каверы",
        "status0": "Добавьте одну или несколько песен и нажмите «Сделать каверы» — они пойдут по очереди. Повторный "
                   "запуск с другими ползунками пересчитывает только сведение — это быстро.",
        "player": "Кавер", "downloads": "Скачать (MP3 и WAV)", "open": "📂 Открыть папку с результатом",
        "stems": "Дорожки по отдельности", "st_v": "Ваш вокал", "st_b": "Бэк-вокал", "st_i": "Минус",
        "log": "Журнал", "log_box": "Ход работы",
        "need_songs": "Добавьте песни: файлы и/или ссылки (по одной в строке)", "need_model": "Выберите голосовую модель",
        "no_models": "Голосовых моделей пока нет: обучите свою в вокальной студии или положите папку модели в "
                     "rvc_models и нажмите ↻",
        "h_num": "#", "h_song": "Песня", "h_res": "Результат", "h_intel": "Разборчивость", "h_tone": "Отличие по тону",
        "working": "⏳ в работе", "queued": "в очереди",
        "song_n": "⏳ Песня {i} из {n} · модель {m} · {s:.0f} с", "work1": "⏳ Работаю… модель {m} · {s:.0f} с",
        "done": "✅ Готово: {k} из {n} за {s:.0f} с · модель {m}",
        "errors": " · ошибок: {e} (подробности в журнале и в папке logs)",
        "none_ok": "❌ Не получилось ни одной песни.", "folder": "Папка: `{d}`",
        "each_folder": "Каждая песня — в своей папке внутри song_output.",
        "s": "с", "db": "дБ",
        "oct": {None: "", 0: "без сдвига", -1: "−1 окт.", 1: "+1 окт.", -2: "−2 окт.", 2: "+2 окт."},
        "studio_md": "### 🎙 Вокальная студия\nЗапишите свой голос прямо в браузере и обучите свою голосовую модель:\n\n"
                     "1. Создайте голос (модель, для которой записываете).\n"
                     "2. **Песни** — спойте 5–10 любых песен целиком (в наушниках, если поёте под песню).\n"
                     "3. По желанию: короткие упражнения (ровные ноты, сирены, распевы, вдохи) — они закрывают "
                     "пробелы в диапазоне.\n"
                     "4. Нажмите **Обучить** — после обучения модель появится в списке на вкладках каверов.\n\n"
                     "В первый раз браузер спросит разрешение на микрофон.",
        "studio_btn": "Открыть вокальную студию",
        "footer": "CoverSong Studio v{v} · используйте только голоса, на которые у вас есть права",
    },
}


def t(lang, key, **kw):
    return tr(TXT, lang, key, **kw)


def pitch_choices(lang):
    return [(t(lang, "p_auto"), "auto"), (t(lang, "p_down"), "-1"), (t(lang, "p_none"), "0"), (t(lang, "p_up"), "1")]


def voc_choices(lang):
    return [(t(lang, "voc_soft"), "soft"), (t(lang, "voc_remove"), "remove")]


def choir_choices(lang):
    return [(t(lang, "c_auto"), "auto"), (t(lang, "c_chords"), "chords"), (t(lang, "c_manual"), "manual"),
            (t(lang, "c_off"), "off")]


def _path(f):
    return f if isinstance(f, str) else getattr(f, "path", None) or getattr(f, "name", None) or str(f)


def _songs(files, links, lang):
    """Queue: uploaded files first, then links (one per line)."""
    out = [_path(f) for f in (files or [])]
    for line in (links or "").splitlines():
        line = line.strip().strip('"')
        if line:
            out.append(line)
    if not out:
        raise gr.Error(t(lang, "need_songs"))
    return out


def _name(song):
    base = os.path.basename(song.rstrip("/\\")) or song
    return os.path.splitext(base)[0] if os.path.exists(song) else song


def list_models():
    """Voice models = folders in rvc_models with a .pth inside (your trained models and any you add)."""
    root = load_config()["paths"]["rvc_models"]
    out = []
    for d in sorted(os.listdir(root), key=str.lower) if os.path.isdir(root) else []:
        p = os.path.join(root, d)
        if os.path.isdir(p) and any(f.lower().endswith(".pth") for f in os.listdir(p)):
            out.append(d)
    return out


def refresh_models(current):
    names = list_models()
    return gr.update(choices=names, value=current if current in names else (names[0] if names else None))


def _table(rows, lang):
    """Markdown table of the queue: done, current and waiting songs."""
    h = [t(lang, k) for k in ("h_num", "h_song", "h_res", "h_intel", "h_tone")]
    out = ["| " + " | ".join(h) + " |", "|---|---|---|---|---|"]
    octs = TXT[lang]["oct"]
    for i, (name, state, r) in enumerate(rows, 1):
        if state == "ok":
            intel = r.get("intelligibility") or {}
            near = r.get("closeness") or {}
            res = f"✅ {r['seconds']:.0f} {t(lang, 's')}, {octs.get(r.get('octaves'), r.get('octaves'))}"
            it = f"{intel['after']:.0%}" if intel.get("after") is not None else "—"
            tone = f"±{near['tone_max_db']:.1f} {t(lang, 'db')}" if near else "—"
        elif state == "error":
            msg = (str(r).strip().splitlines() or [r.__class__.__name__])[0].replace("|", "/")
            res, it, tone = f"❌ {msg[:120]}", "—", "—"
        elif state == "work":
            res, it, tone = t(lang, "working"), "", ""
        else:
            res, it, tone = t(lang, "queued"), "", ""
        name = name[:60].replace("|", "/")
        out.append(f"| {i} | {name} | {res} | {it} | {tone} |")
    return "\n".join(out)


def make_covers(lang):
    def run(files, links, model, pitch, vocalizations, reverb, vocal_db, choir="auto", choir_at="", crowd=False,
            crowd_at=""):
        songs = _songs(files, links, lang)
        if not model:
            raise gr.Error(t(lang, "need_model"))
        cfg = load_config(overrides=[
            f"convert.pitch_octaves={pitch}",
            f"clean.vocalizations={vocalizations}",
            f"mix.reverb={reverb}",
            f"mix.vocal_gain_db={vocal_db}",
            f"clean.choir={choir or 'auto'}",
        ])
        cfg.setdefault("clean", {})["choir_ranges"] = choir_at or ""  # set directly: YAML would read 1:20 as a number
        if crowd:
            cfg["clean"]["crowd"] = "auto"
        cfg["clean"]["crowd_ranges"] = crowd_at or ""
        rows = [[_name(s), "wait", None] for s in songs]
        lines = []
        done = threading.Event()

        def hook(line):
            lines.append(line)

        def work():
            from .pipeline import run_pipeline
            from .util import log
            for i, song in enumerate(songs):
                rows[i][1] = "work"
                if len(songs) > 1:
                    log(f"===== Song {i + 1} of {len(songs)}: {rows[i][0]} =====")
                try:
                    with RUN_LOCK:
                        rows[i][2] = run_pipeline(cfg, song, voice_model=model)
                    rows[i][1] = "ok"
                except Exception as e:  # noqa: BLE001  one bad song does not stop the queue; traceback to the log
                    traceback.print_exc()
                    rows[i][1], rows[i][2] = "error", e
            done.set()

        LOG_HOOKS.append(hook)
        t0 = time.time()
        threading.Thread(target=work, daemon=True).start()
        nothing = gr.update()
        try:
            while not done.is_set():
                time.sleep(0.7)
                cur = next((i for i, r in enumerate(rows) if r[1] == "work"), 0)
                head = (t(lang, "song_n", i=cur + 1, n=len(songs), m=model, s=time.time() - t0) if len(songs) > 1
                        else t(lang, "work1", m=model, s=time.time() - t0))
                yield (head + "\n\n" + _table(rows, lang), "\n".join(lines[-300:]), *([nothing] * 6))
        finally:
            LOG_HOOKS.remove(hook)
        log_text = "\n".join(lines[-300:])
        ok = [r[2] for r in rows if r[1] == "ok"]
        n_err = sum(r[1] == "error" for r in rows)
        head = t(lang, "done", k=len(ok), n=len(songs), s=time.time() - t0, m=model) + (
            t(lang, "errors", e=n_err) if n_err else "")
        if not ok:
            yield (t(lang, "none_ok") + "\n\n" + _table(rows, lang), log_text, None, None, None, None, None, None)
            return
        last = ok[-1]
        files_out = [p for r in ok for p in (r.get("final"), r.get("final_wav")) if p and os.path.exists(p)]
        stems = last["stems"]

        def stem(n):
            p = os.path.join(stems, f"{n}.wav")
            return p if os.path.exists(p) else None
        where = "\n\n" + (t(lang, "folder", d=last["song_dir"]) if len(ok) == 1 else t(lang, "each_folder"))
        yield (head + "\n\n" + _table(rows, lang) + where, log_text, last.get("final"), files_out,
               stem("vocal"), stem("backing"), stem("instrumental"), last["song_dir"] if len(ok) == 1 else None)
    return run


def open_folder(path):
    target = path if path and os.path.isdir(path) else load_config()["paths"]["output"]
    os.makedirs(target, exist_ok=True)
    if hasattr(os, "startfile"):
        os.startfile(target)  # noqa: S606  Windows Explorer


def build(lang="en"):
    from . import ui_multi
    models = list_models()
    default_model = load_config().get("voice_model")
    default_model = default_model if default_model in models else (models[0] if models else None)
    with gr.Blocks(title="CoverSong Studio") as demo:
        en, ru = ("active", "") if lang == "en" else ("", "active")
        gr.HTML(f'<div class="sc-head"><div><h1>CoverSong Studio</h1><p>{t(lang, "subtitle")}</p></div>'
                f'<div class="lang"><a href="/" class="{en}">EN</a> · <a href="/ru/" class="{ru}">RU</a></div></div>',
                elem_id="sc-head", js_on_load=HEAD_JS)
        with gr.Tab(t(lang, "tab_one")):
            folder = gr.State(None)
            with gr.Row():
                with gr.Column(scale=3):
                    file = gr.File(label=t(lang, "songs"), file_types=["audio", "video"], type="filepath",
                                   file_count="multiple")
                    link = gr.Textbox(label=t(lang, "links_box"), lines=3,
                                      placeholder="https://www.youtube.com/watch?v=…\nhttps://www.youtube.com/watch?v=…")
                with gr.Column(scale=2):
                    with gr.Row():
                        model = gr.Dropdown(models, value=default_model, label=t(lang, "model"), scale=5,
                                            info=t(lang, "model_info") if models else t(lang, "no_models"))
                        refresh = gr.Button("↻", scale=1, min_width=40)
                    pitch = gr.Radio(pitch_choices(lang), value="auto", label=t(lang, "pitch"),
                                     info=t(lang, "pitch_info"))
                    voc = gr.Radio(voc_choices(lang), value="soft", label=t(lang, "voc"))
                    reverb = gr.Slider(0, 2, value=1.0, step=0.1, label=t(lang, "reverb"), info=t(lang, "reverb_info"))
                    vocal_db = gr.Slider(-6, 6, value=0, step=0.5, label=t(lang, "vocal_db"),
                                         info=t(lang, "vocal_db_info"))
                    with gr.Accordion(t(lang, "choir_acc"), open=False):
                        choir = gr.Radio(choir_choices(lang), value="auto", label=t(lang, "choir"),
                                         info=t(lang, "choir_info"))
                        choir_at = gr.Textbox(label=t(lang, "choir_at"), placeholder="0:10-0:16, 1:20-1:30",
                                              info=t(lang, "choir_at_info"))
                        crowd = gr.Checkbox(value=False, label=t(lang, "crowd"), info=t(lang, "crowd_info"))
                        crowd_at = gr.Textbox(label=t(lang, "crowd_at"), placeholder="0:44-0:46, 1:51-1:54")
            go = gr.Button(t(lang, "go"), variant="primary", elem_id="go")
            status = gr.Markdown(t(lang, "status0"))
            with gr.Row():
                player = gr.Audio(label=t(lang, "player"), type="filepath", interactive=False)
                downloads = gr.File(label=t(lang, "downloads"), file_count="multiple", interactive=False)
            open_btn = gr.Button(t(lang, "open"))
            with gr.Accordion(t(lang, "stems"), open=False):
                with gr.Row():
                    st_v = gr.Audio(label=t(lang, "st_v"), type="filepath", interactive=False)
                    st_b = gr.Audio(label=t(lang, "st_b"), type="filepath", interactive=False)
                    st_i = gr.Audio(label=t(lang, "st_i"), type="filepath", interactive=False)
            with gr.Accordion(t(lang, "log"), open=False):
                log_box = gr.Textbox(label=t(lang, "log_box"), lines=16, max_lines=30, autoscroll=True,
                                     buttons=["copy"])
            refresh.click(refresh_models, [model], [model])
            go.click(make_covers(lang), [file, link, model, pitch, voc, reverb, vocal_db, choir, choir_at, crowd,
                                         crowd_at],
                     [status, log_box, player, downloads, st_v, st_b, st_i, folder])
            open_btn.click(open_folder, [folder], None)
        with gr.Tab(t(lang, "tab_multi")):
            ui_multi.build_tab(lang)
        with gr.Tab(t(lang, "tab_studio")):
            gr.HTML(f'<iframe class="sc-frame" src="/studio/?embed=1&lang={lang}" allow="microphone; autoplay" '
                    f'title="{t(lang, "tab_studio")}"></iframe>', js_on_load=FRAME_JS)
        with gr.Tab(t(lang, "tab_library")):
            gr.HTML(f'<iframe class="sc-frame" src="/studio/library?embed=1&lang={lang}" allow="autoplay" '
                    f'title="{t(lang, "tab_library")}"></iframe>', js_on_load=FRAME_JS)
        gr.Markdown(f"<div style='text-align:center;opacity:.5'>{t(lang, 'footer', v=__version__)}</div>")
    demo.queue(default_concurrency_limit=1)
    return demo


def theme():
    def both(name, key):
        if name == "color_accent":  # one value for both themes
            return {name: PAL[key][0]}
        return {name: PAL[key][0], name + "_dark": PAL[key][1]}
    v = {}
    for name, key in [("body_background_fill", "bg"), ("background_fill_secondary", "bg"), ("panel_background_fill", "bg"),
                      ("body_text_color", "ink"), ("block_title_text_color", "ink"), ("body_text_color_subdued", "muted"),
                      ("block_info_text_color", "muted"), ("block_label_text_color", "muted"),
                      ("background_fill_primary", "card"), ("block_background_fill", "card"),
                      ("button_secondary_background_fill", "card"), ("block_label_background_fill", "soft"),
                      ("button_secondary_background_fill_hover", "soft"), ("checkbox_label_background_fill", "input"),
                      ("block_border_color", "line"), ("border_color_primary", "line"), ("input_border_color", "line"),
                      ("button_secondary_border_color", "line"), ("checkbox_label_border_color", "line"),
                      ("table_border_color", "line"), ("input_background_fill", "input"),
                      ("color_accent", "accent"), ("border_color_accent", "accent"), ("link_text_color", "accent"),
                      ("link_text_color_hover", "accent_hover"), ("link_text_color_visited", "accent"),
                      ("button_primary_background_fill", "accent"), ("button_primary_border_color", "accent"),
                      ("button_primary_background_fill_hover", "accent_hover"),
                      ("button_primary_border_color_hover", "accent_hover"), ("slider_color", "accent"),
                      ("checkbox_background_color_selected", "accent"), ("checkbox_border_color_selected", "accent"),
                      ("loader_color", "accent"), ("input_border_color_focus", "accent"),
                      ("color_accent_soft", "accent_soft"), ("checkbox_label_background_fill_selected", "accent_soft"),
                      ("checkbox_label_border_color_selected", "accent")]:
        v.update(both(name, key))
    v.update(button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
             button_secondary_text_color=PAL["ink"][0], button_secondary_text_color_dark=PAL["ink"][1],
             checkbox_label_text_color_selected=PAL["ink"][0], checkbox_label_text_color_selected_dark=PAL["ink"][1],
             block_radius="16px", container_radius="16px", input_radius="10px", button_large_radius="10px",
             button_medium_radius="10px", button_small_radius="10px", checkbox_label_border_width="1px",
             block_shadow="0 1px 2px rgba(0,0,0,.05), 0 6px 20px rgba(0,0,0,.05)", block_shadow_dark="none",
             block_label_radius="8px", block_title_text_weight="600", button_primary_shadow="none",
             layout_gap="16px", block_padding="16px 18px")
    return gr.themes.Base(font=["Segoe UI", "system-ui", "-apple-system", "sans-serif"],
                          font_mono=["Consolas", "ui-monospace", "monospace"]).set(**v)


def css():
    from . import ui_multi
    return CSS + ui_multi.EDITOR_CSS


if __name__ == "__main__":
    from .app import main
    main()

