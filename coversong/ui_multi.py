"""The "Several voices" tab: find the singers of a song, listen to each, give each one a voice model."""
import json
import os
import re
import threading
import time
import traceback

import gradio as gr

from . import voices
from .config import load_config
from .i18n import note, tr
from .util import LOG_HOOKS

MAX_V = voices.MAX_VOICES
ORIG, MUTE, BY_VOICES = "__orig__", "__mute__", "__voices__"

TXT = {
    "en": {
        "orig": "Original voice (keep)", "mute": "Remove this voice",
        "by_voices": "Each voice its own (as in the lead)",
        "o_auto": "Auto", "o_0": "No shift", "o_-1": "−1 octave (female → male)", "o_1": "+1 octave (male → female)",
        "o_-2": "−2 octaves", "o_2": "+2 octaves",
        "lyrics_ph": "Optional. Open the song on Genius, select the lyrics on the page and paste them here "
                     "(Ctrl+C → Ctrl+V): bold and italics are kept, and they show who sings a line.\n\nOr type them "
                     "yourself:\n[Verse 1: Nora, Mia]\nNora's line\nMia's line — make it and the name \"Mia\" in the "
                     "header italic (button I)\n[Chorus: All]\n…",
        "b": "Bold (Ctrl+B)", "i": "Italic (Ctrl+I)", "u": "Underline (Ctrl+U)", "bl": "B", "il": "I", "ul": "U",
        "plain": "No format", "plain_t": "Remove formatting from the selection", "clear": "Clear",
        "clear_t": "Erase all the text",
        "legend": "As on Genius: in the [header] each singer's name is written in the same style (plain, <i>italic</i>, "
                  "<b>bold</b>, <u>underlined</u>) as their lines.",
        "need_song": "Choose a song or paste a link", "need_find": "Press \"Find voices\" first",
        "manual_oct": "Shift chosen by hand: {o}.",
        "unknown_range": "This model's range is unknown — \"Auto\" keeps the key as it is. If the voice in the song is "
                         "female and the model male, choose \"−1 octave\"; the other way round, \"+1 octave\".",
        "auto_is": "\"Auto\": {what} (the voice in the song is around {a}, the model around {b}).",
        "what": {0: "no shift", -1: "an octave lower", 1: "an octave higher", -2: "2 octaves lower",
                 2: "2 octaves higher"},
        "voice_n": "Voice {i}", "sings": "sings {p:.0%} of the song ({n} lines)", "together": "{n} more lines together "
                                                                                             "with others",
        "around": "on average around {n}", "no_words": "(no words)",
        "found": "**Voices found: {k}**", "by_text": "from the lyrics", "auto": "automatically",
        "found_tail": ". Listen to each and choose whose voice sings it and whether a shift is needed (e.g. if a girl "
                      "sings and the model is male). If the program got it wrong, fix the lines in the table below "
                      "(voice number; \"1+2\" — sing together; \"all\"; \"—\" — remove) and press \"Update voices "
                      "from the table\".",
        "finding": "⏳ Separating the voices and finding who sings where…", "error": "❌ Error: {e}",
        "found_ok": "✅ Voices found — choose the models below.", "updated": "✅ Voices updated from the table.",
        "too_many": "There are {k} voices, and the tab shows up to {n} — merge some in the lines table",
        "need_model": "Choose a model for at least one voice", "who": "voice {i} → {m}",
        "making": "⏳ Making a cover for several voices…",
        "ready": "✅ Done in {s:.0f} s · {who}", "intel": " · intelligibility {p:.0%}", "folder": "Folder: `{d}`",
        "about": "For songs where several people sing: the program finds the voices, lets you listen to each, and "
                 "you choose which model sings for each one and in which key. It can confuse the voices of similar "
                 "singers (e.g. members of one group) — then paste the lyrics with the singers' names or fix the "
                 "lines in the table.",
        "song": "Song", "link": "…or a YouTube link", "know_k": "I know how many voices there are",
        "k": "Voices in the song", "find": "🔍 Find voices", "lyrics_acc": "Lyrics with the singers' names (optional)",
        "status0": "Choose a song and press \"Find voices\".", "whose": "Whose voice sings it", "key": "Key",
        "backing": "Backing vocals",
        "backing_info": "By default the backing under a line is sung by the same voice as the line, with its model. "
                        "Or keep the original, or sing all the backing with one model",
        "backing_key": "Backing key", "table_acc": "Song lines: who sings which (you can fix it)",
        "table_help": "In the \"Voice\" column: a voice number (2), several together (1+3 — a choir is sung like "
                      "this too), \"all\" or \"—\" (remove the line).",
        "h_time": "Time", "h_words": "Words", "h_voice": "Voice", "upd": "↻ Update voices from the table",
        "voc": "Wordless singing", "reverb": "Reverb", "vocal_db": "Voices' level, dB",
        "go": "🎤 Make a cover for several voices", "player": "Cover", "downloads": "Download", "log": "Log",
        "log_box": "Progress",
    },
    "ru": {
        "orig": "Оригинальный голос (не менять)", "mute": "Убрать этот голос",
        "by_voices": "У каждого голоса свой (как в основном вокале)",
        "o_auto": "Авто", "o_0": "Без сдвига", "o_-1": "−1 октава (женский → мужской)",
        "o_1": "+1 октава (мужской → женский)", "o_-2": "−2 октавы", "o_2": "+2 октавы",
        "lyrics_ph": "Необязательно. Откройте песню на Genius, выделите текст на странице и вставьте сюда "
                     "(Ctrl+C → Ctrl+V): жирный и курсив сохранятся, а по ним видно, кто поёт строку.\n\nИли "
                     "наберите сами:\n[Куплет 1: Нора, Мия]\nстрока Норы\nстрока Мии — выделите её и имя «Мия» в "
                     "заголовке курсивом (кнопка К)\n[Припев: Все]\n…",
        "b": "Жирный (Ctrl+B)", "i": "Курсив (Ctrl+I)", "u": "Подчёркнутый (Ctrl+U)", "bl": "Ж", "il": "К", "ul": "Ч",
        "plain": "Без формата", "plain_t": "Убрать форматирование у выделенного", "clear": "Очистить",
        "clear_t": "Стереть весь текст",
        "legend": "Как на Genius: в [заголовке] имя каждого исполнителя написано тем же стилем (обычный, "
                  "<i>курсив</i>, <b>жирный</b>, <u>подчёркнутый</u>), что и его строки.",
        "need_song": "Выберите песню или вставьте ссылку", "need_find": "Сначала нажмите «Найти голоса»",
        "manual_oct": "Сдвиг выбран вручную: {o}.",
        "unknown_range": "Диапазон этой модели неизвестен — «Авто» оставит тональность как есть. Если голос в песне "
                         "женский, а модель мужская, выберите «−1 октава», если наоборот — «+1 октава».",
        "auto_is": "«Авто»: {what} (голос в песне около {a}, модель — около {b}).",
        "what": {0: "без сдвига", -1: "на октаву ниже", 1: "на октаву выше", -2: "на 2 октавы ниже",
                 2: "на 2 октавы выше"},
        "voice_n": "Голос {i}", "sings": "поёт {p:.0%} песни ({n} строк)",
        "together": "ещё {n} строк вместе с другими", "around": "в среднем около {n}", "no_words": "(без слов)",
        "found": "**Найдено голосов: {k}**", "by_text": "по тексту песни", "auto": "автоматически",
        "found_tail": ". Послушайте каждый и выберите, чьим голосом его петь и нужен ли сдвиг (например, если поёт "
                      "девушка, а модель мужская). Если программа ошиблась, поправьте строки в таблице ниже (номер "
                      "голоса; «1+2» — поют вместе; «все»; «—» — убрать) и нажмите «Обновить голоса по таблице».",
        "finding": "⏳ Разделяю голоса и ищу, кто где поёт…", "error": "❌ Ошибка: {e}",
        "found_ok": "✅ Голоса найдены — выберите модели ниже.", "updated": "✅ Голоса обновлены по таблице.",
        "too_many": "Голосов {k}, а вкладка показывает до {n} — объедините кого-то в таблице строк",
        "need_model": "Выберите модель хотя бы для одного голоса", "who": "голос {i} → {m}",
        "making": "⏳ Делаю кавер на несколько голосов…",
        "ready": "✅ Готово за {s:.0f} с · {who}", "intel": " · разборчивость {p:.0%}", "folder": "Папка: `{d}`",
        "about": "Для песен, где поют несколько человек: программа находит голоса, даёт послушать каждый, а вы "
                 "выбираете, какой моделью петь за каждого и в какой тональности. Голоса похожих исполнителей "
                 "(например, участниц одной группы) программа может спутать — тогда вставьте текст с именами "
                 "исполнителей или поправьте строки в таблице.",
        "song": "Песня", "link": "…или ссылка на YouTube", "know_k": "Я знаю, сколько голосов",
        "k": "Голосов в песне", "find": "🔍 Найти голоса", "lyrics_acc": "Текст песни с именами исполнителей "
                                                                       "(необязательно)",
        "status0": "Выберите песню и нажмите «Найти голоса».", "whose": "Чьим голосом петь", "key": "Тональность",
        "backing": "Бэк-вокал (подпевки)",
        "backing_info": "По умолчанию подпевки под строкой поёт тот же голос, что и строку, его моделью. Или оставить "
                        "оригинал, или петь все подпевки одной моделью",
        "backing_key": "Тональность бэк-вокала", "table_acc": "Строки песни: кто какую поёт (можно исправить)",
        "table_help": "В колонке «Голос»: номер голоса (2), несколько вместе (1+3 — так поётся и хор), «все» или «—» "
                      "(убрать строку).",
        "h_time": "Время", "h_words": "Слова", "h_voice": "Голос", "upd": "↻ Обновить голоса по таблице",
        "voc": "Распевы без слов", "reverb": "Реверберация", "vocal_db": "Громкость голосов, дБ",
        "go": "🎤 Сделать кавер на несколько голосов", "player": "Кавер", "downloads": "Скачать", "log": "Журнал",
        "log_box": "Ход работы",
    },
}
OCT_VALUES = ["auto", "0", "-1", "1", "-2", "2"]


def t(lang, key, **kw):
    return tr(TXT, lang, key, **kw)


def oct_choices(lang):
    return [(t(lang, f"o_{v}"), v) for v in OCT_VALUES]


def _oct(v):
    return "auto" if v in (None, "", "auto") else int(v)


def editor_html(lang):
    ph = t(lang, "lyrics_ph").replace("<", "‹").replace(">", "›").replace('"', "'")
    return f"""
<div class="kl-wrap">
  <div class="kl-bar">
    <button type="button" data-cmd="bold" title="{t(lang, 'b')}"><b>{t(lang, 'bl')}</b></button>
    <button type="button" data-cmd="italic" title="{t(lang, 'i')}"><i>{t(lang, 'il')}</i></button>
    <button type="button" data-cmd="underline" title="{t(lang, 'u')}"><u>{t(lang, 'ul')}</u></button>
    <button type="button" data-cmd="removeFormat" title="{t(lang, 'plain_t')}">{t(lang, 'plain')}</button>
    <button type="button" data-cmd="clear" title="{t(lang, 'clear_t')}">{t(lang, 'clear')}</button>
    <span class="kl-legend">{t(lang, 'legend')}</span>
  </div>
  <div class="kl-ed" contenteditable="true" spellcheck="false" data-ph="{ph}"></div>
</div>
"""


EDITOR_CSS = """
.kl-wrap {border:1px solid var(--border-color-primary); border-radius:var(--radius-lg, 8px); overflow:hidden;
  background:var(--input-background-fill)}
.kl-bar {display:flex; flex-wrap:wrap; gap:6px; align-items:center; padding:6px 8px;
  border-bottom:1px solid var(--border-color-primary); background:var(--block-background-fill)}
.kl-bar button {min-width:34px; padding:4px 10px; border-radius:6px; border:1px solid var(--border-color-primary);
  background:var(--button-secondary-background-fill); color:var(--body-text-color); cursor:pointer; font-size:14px}
.kl-bar button:hover {border-color:var(--body-text-color-subdued)}
.kl-legend {font-size:12px; color:var(--body-text-color-subdued); flex:1 1 280px}
.kl-ed {min-height:220px; max-height:480px; overflow:auto; padding:10px 12px; outline:none; white-space:pre-wrap;
  color:var(--body-text-color); font-size:14px; line-height:1.55}
.kl-ed:empty:before {content:attr(data-ph); color:var(--body-text-color-subdued); white-space:pre-wrap}
"""
# Runs when the editor appears. Toolbar buttons format the selection; a paste from a web page keeps only line
# breaks and bold / italic / underline (Genius marks singers this way), whether the page used tags or inline styles;
# everything else (links, fonts, colours) is dropped.
EDITOR_JS = r"""
const ed = element.querySelector('.kl-ed');
element.querySelectorAll('.kl-bar button').forEach(b => {
  b.addEventListener('mousedown', e => {
    e.preventDefault();
    const cmd = b.dataset.cmd;
    if (cmd === 'clear') ed.innerHTML = ''; else document.execCommand(cmd);
  });
});
const BLOCK = new Set(['DIV','P','LI','TR','H1','H2','H3','H4','H5','H6','BLOCKQUOTE','SECTION','ARTICLE']);
const esc = t => t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
function flags(el, inh) {
  const f = Object.assign({}, inh), t = el.tagName;
  if (t === 'B' || t === 'STRONG') f.b = true;
  if (t === 'I' || t === 'EM') f.i = true;
  if (t === 'U' || t === 'INS') f.u = true;
  const s = el.style;
  if (s) {
    if (s.fontWeight) f.b = s.fontWeight === 'bold' || s.fontWeight === 'bolder' || parseInt(s.fontWeight) >= 600;
    if (s.fontStyle) f.i = s.fontStyle === 'italic' || s.fontStyle === 'oblique';
    const td = s.textDecorationLine || s.textDecoration;
    if (td) f.u = td.indexOf('underline') >= 0;
  }
  return f;
}
function clean(root) {
  let out = '', has = false;
  (function walk(node, f) {
    for (const n of node.childNodes) {
      if (n.nodeType === 3) {
        let t = n.nodeValue.replace(/\s+/g, ' ');
        if (!has) t = t.replace(/^ +/, '');  // no indentation from the page's markup
        if (!t) continue;
        let h = esc(t);
        if (t.trim()) {
          if (f.u) h = '<u>' + h + '</u>';
          if (f.i) h = '<i>' + h + '</i>';
          if (f.b) h = '<b>' + h + '</b>';
          has = true;
        }
        out += h;
      } else if (n.nodeType === 1) {
        const tag = n.tagName;
        if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'HEAD' || tag === 'TITLE') continue;
        if (tag === 'BR') { out += '<br>'; has = false; continue; }
        const blk = BLOCK.has(tag);
        if (blk && has) { out += '<br>'; has = false; }
        walk(n, flags(n, f));
        if (blk && has) { out += '<br>'; has = false; }
      }
    }
  })(root, {});
  return out;
}
ed.addEventListener('paste', e => {
  const cd = e.clipboardData;
  if (!cd) return;
  const html = cd.getData('text/html'), text = cd.getData('text/plain');
  if (!html && !text) return;
  e.preventDefault();
  const out = html ? clean(new DOMParser().parseFromString(html, 'text/html').body) : esc(text).replace(/\r?\n/g, '<br>');
  document.execCommand('insertHTML', false, out);
});
"""
# Reads the editor's formatted text when "Find voices" is pressed (the editor is plain HTML, not a Gradio input).
READ_LYRICS_JS = r"""
(f, l, t, k, m) => {
  const ed = document.querySelector('.kl-ed');
  return [f, l, ed ? ed.innerHTML : (t || ''), k, m];
}
"""


def fmt(s):
    return f"{int(s // 60)}:{s % 60:04.1f}"


def _run(fn):
    """Run fn in a thread; yield (finished, result_or_exception, log_text) every 0.7 s."""
    lines, box, done = [], {}, threading.Event()

    def hook(line):
        lines.append(line)

    def work():
        from .ui import RUN_LOCK
        try:
            with RUN_LOCK:
                box["r"] = fn()
        except Exception as e:  # noqa: BLE001  shown in the UI; traceback goes to the console/log
            traceback.print_exc()
            box["e"] = e
        done.set()

    LOG_HOOKS.append(hook)
    threading.Thread(target=work, daemon=True).start()
    try:
        while not done.is_set():
            time.sleep(0.7)
            yield False, None, "\n".join(lines[-200:])
    finally:
        LOG_HOOKS.remove(hook)
    yield True, box.get("e", box.get("r")), "\n".join(lines[-200:])


def _song(file, link, lang):
    if file is not None:
        return file if isinstance(file, str) else getattr(file, "path", None) or getattr(file, "name", None)
    link = (link or "").strip().strip('"')
    if not link:
        raise gr.Error(t(lang, "need_song"))
    return link


def _models():
    from .ui import list_models
    return list_models()


def _choices(lang):
    return [(m, m) for m in _models()] + [(t(lang, "orig"), ORIG), (t(lang, "mute"), MUTE)]


def _values(choices):
    return [v for _, v in choices]


def octave_hint(model, octave, median_midi, lang):
    """One line under a voice: what "Auto" will do with this model, or how to choose by hand."""
    if not model or model in (ORIG, MUTE):
        return ""
    if _oct(octave) != "auto":
        return t(lang, "manual_oct", o=t(lang, f"o_{octave}"))
    cfg = load_config()
    path = os.path.join(cfg["paths"]["rvc_models"], model, "voice_profile.json")
    if not os.path.exists(path) or median_midi is None:
        return t(lang, "unknown_range")
    from .pitch import suggest_octaves
    with open(path, encoding="utf-8") as f:
        prof = json.load(f)
    k = suggest_octaves(median_midi, prof["median"])
    return t(lang, "auto_is", what=TXT[lang]["what"][k], a=note(median_midi, lang), b=note(prof["median"], lang))


def _voice_md(v, model, octave, lang):
    i = v["id"]
    title = "**" + t(lang, "voice_n", i=i + 1) + (f" — {v['name']}" if v.get("name") else "") + "**"
    parts = [t(lang, "sings", p=v["share"], n=v["lines"])]
    if v.get("together"):
        parts.append(t(lang, "together", n=v["together"]))
    if v["median_midi"] is not None:
        parts.append(t(lang, "around", n=note(v["median_midi"], lang)))
    hint = octave_hint(model, octave, v["median_midi"], lang)
    return title + " · " + " · ".join(parts) + (f"  \n<small>{hint}</small>" if hint else "")


def _table(res, lang):
    return [[f"{fmt(r['start'])}–{fmt(r['end'])}", r["text"] or t(lang, "no_words"), voices.show(r["voice"])]
            for r in res["lines"]]


def _render(res, state, lang):
    """Updates for: summary, k, table, (audio, info, model, octave) × MAX_V, backing model, backing octave."""
    k = res["k"]
    choices = _choices(lang)
    vals = _values(choices)
    default = load_config().get("voice_model")
    default = default if default in vals else next((v for v in vals if v not in (ORIG, MUTE)), ORIG)
    st = state or {}
    prev, prev_oct = st.get("assign", {}), st.get("octaves", {})
    how = t(lang, "by_text") if res.get("by_text") else (t(lang, "auto") if res.get("auto") else "")
    out = [t(lang, "found", k=k) + (f" ({how})" if how else "") + t(lang, "found_tail"), gr.update(value=k),
           _table(res, lang)]
    for i in range(MAX_V):
        if i < k:
            v = res["voices"][i]
            val = prev.get(i) if prev.get(i) in vals else (default if i == 0 else ORIG)
            octv = prev_oct.get(i) if prev_oct.get(i) in OCT_VALUES else "auto"
            out += [gr.update(value=v["sample"], visible=True,
                              label=t(lang, "voice_n", i=i + 1) + (f" — {v['name']}" if v.get("name") else "")),
                    gr.update(value=_voice_md(v, val, octv, lang), visible=True),
                    gr.update(choices=choices, value=val, visible=True),
                    gr.update(value=octv, visible=True)]
        else:
            out += [gr.update(value=None, visible=False), gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False)]
    bchoices = [(t(lang, "by_voices"), BY_VOICES), (t(lang, "orig"), ORIG)] + [(m, m) for m in _models()]
    bk = st.get("backing") if st.get("backing") in _values(bchoices) else BY_VOICES
    out += [gr.update(choices=bchoices, value=bk),
            gr.update(value=st.get("backing_oct") or "auto", visible=bk not in (BY_VOICES, ORIG))]
    return out


N_RENDER = 3 + 4 * MAX_V + 2


def find_voices(lang):
    def run(file, link, lyrics, k_manual, manual):
        song = _song(file, link, lang)
        cfg = load_config()
        state = {"song": song}
        if not re.sub(r"<[^>]*>|&nbsp;|\s", "", lyrics or ""):
            lyrics = ""  # empty editor

        def work():
            from .pipeline import run_pipeline
            sep = run_pipeline(cfg, song, until_stage="separate")
            song_dir = os.path.dirname(os.path.dirname(sep["lead_dry"]))
            state.update(song_dir=song_dir, lead=sep["lead_dry"])
            title = os.path.basename(song_dir).rsplit("_", 1)[0]
            artist = title.split(" - ", 1)[0].strip() if " - " in title else None
            return voices.analyze(cfg, song_dir, sep["lead_dry"], k=int(k_manual) if manual else None,
                                  lyrics=lyrics, song_artist=artist)

        for finished, res, log_text in _run(work):
            if not finished:
                yield [t(lang, "finding"), log_text, state] + [gr.update()] * N_RENDER
            elif isinstance(res, Exception):
                yield [t(lang, "error", e=res), log_text, state] + [gr.update()] * N_RENDER
            else:
                state["analysis"] = res
                yield [t(lang, "found_ok"), log_text, state] + _render(res, state, lang)
    return run


def _split_ctl(ctl):
    dd, oc = list(ctl[0:2 * MAX_V:2]), list(ctl[1:2 * MAX_V:2])
    return dd, oc, list(ctl[2 * MAX_V:])


def _remember(state, ctl):
    """Keep the user's choices (models, octaves) across re-analysis."""
    dd, oc, (bk, bo) = _split_ctl(ctl)
    state["assign"] = {i: v for i, v in enumerate(dd) if v}
    state["octaves"] = {i: v for i, v in enumerate(oc) if v}
    state.update(backing=bk, backing_oct=bo)


def apply_table(lang):
    def run(state, table, *ctl):
        if not state or "analysis" not in state:
            raise gr.Error(t(lang, "need_find"))
        cfg = load_config()
        _remember(state, ctl)
        res = voices.relabel(cfg, state["song_dir"], state["lead"], _rows(table))
        state["analysis"] = res
        return [t(lang, "updated"), state] + _render(res, state, lang)
    return run


def _rows(table):
    """The Dataframe's value as a list of rows (Gradio may hand over a DataFrame or a dict)."""
    if table is None:
        return []
    if hasattr(table, "values") and hasattr(table, "columns"):
        return table.values.tolist()
    if isinstance(table, dict) and "data" in table:
        return table["data"]
    return table


def voice_info(i, lang):
    def f(state, model, octave):
        res = (state or {}).get("analysis")
        if not res or i >= res["k"]:
            return gr.update()
        return gr.update(value=_voice_md(res["voices"][i], model, octave, lang))
    return f


def _special(m):
    return {ORIG: "orig", MUTE: "mute", BY_VOICES: "voices"}.get(m, m)


def build_plan(res, ctl, lang):
    """(plan, main model, description) from the tab's controls; raises gr.Error on an impossible choice."""
    k = res["k"]
    dd, oc, (bk, bo) = _split_ctl(ctl)
    if k > len(dd):
        raise gr.Error(t(lang, "too_many", k=k, n=len(dd)))
    assign = {i: _special(dd[i]) or "orig" for i in range(k)}
    octaves = {i: _oct(oc[i]) for i in range(k)}
    octaves["backing"] = _oct(bo)
    labels = [r["voice"] for r in res["lines"]]
    sung = {v for lab in labels for v in voices.members(lab)}
    real = [(m, res["voices"][i]["seconds"]) for i, m in assign.items() if m not in ("orig", "mute") and i in sung]
    if not real:
        raise gr.Error(t(lang, "need_model"))
    main = max(real, key=lambda r: r[1])[0]
    plan = voices.plan(res, assign, _special(bk) or "voices", octaves)
    names = {ORIG: t(lang, "orig"), MUTE: t(lang, "mute")}
    who = ", ".join(t(lang, "who", i=i + 1, m=names.get(dd[i], dd[i])) +
                    ("" if _oct(oc[i]) == "auto" else f" ({t(lang, 'o_' + str(oc[i]))})")
                    for i in range(k) if i in sung)
    return plan, main, who


def make_multi(lang):
    def run(state, table, voc, reverb, vocal_db, *ctl):
        if not state or "analysis" not in state:
            raise gr.Error(t(lang, "need_find"))
        cfg = load_config(overrides=[f"clean.vocalizations={voc}", f"mix.reverb={reverb}",
                                     f"mix.vocal_gain_db={vocal_db}"])
        _remember(state, ctl)
        res = state["analysis"]
        rows = _rows(table)
        if rows and [voices.show(r["voice"]) for r in res["lines"]] != [str(row[2]).strip() for row in rows]:
            res = voices.relabel(cfg, state["song_dir"], state["lead"], rows)
            state["analysis"] = res
        plan, main, who = build_plan(res, ctl, lang)

        def work():
            from .pipeline import run_pipeline
            return run_pipeline(cfg, state["song"], voice_model=main, voice_plan=plan)

        for finished, r, log_text in _run(work):
            if not finished:
                yield t(lang, "making"), log_text, gr.update(), gr.update(), state
            elif isinstance(r, Exception):
                yield t(lang, "error", e=r), log_text, None, None, state
            else:
                files = [p for p in (r.get("final"), r.get("final_wav")) if p and os.path.exists(p)]
                intel = r.get("intelligibility") or {}
                yield (t(lang, "ready", s=r["seconds"], who=who) +
                       (t(lang, "intel", p=intel["after"]) if intel.get("after") is not None else "") +
                       "\n\n" + t(lang, "folder", d=r["song_dir"]), log_text, r.get("final"), files, state)
    return run


def build_tab(lang="en"):
    from .ui import voc_choices
    state = gr.State({})
    gr.Markdown(t(lang, "about"))
    with gr.Row():
        with gr.Column(scale=3):
            file = gr.File(label=t(lang, "song"), file_types=["audio", "video"], type="filepath")
            link = gr.Textbox(label=t(lang, "link"), placeholder="https://www.youtube.com/watch?v=…")
        with gr.Column(scale=2):
            manual = gr.Checkbox(label=t(lang, "know_k"), value=False)
            k = gr.Slider(1, MAX_V, value=2, step=1, label=t(lang, "k"))
            find = gr.Button(t(lang, "find"), variant="primary")
    with gr.Accordion(t(lang, "lyrics_acc"), open=False):
        gr.HTML(editor_html(lang), js_on_load=EDITOR_JS, apply_default_css=False)
        lyrics = gr.Textbox(visible=False)  # filled from the editor by READ_LYRICS_JS
    status = gr.Markdown(t(lang, "status0"))
    summary = gr.Markdown()
    ctl, row_out = [], []
    for i in range(MAX_V):
        with gr.Row():
            a = gr.Audio(label=t(lang, "voice_n", i=i + 1), type="filepath", interactive=False, visible=False)
            with gr.Column():
                info = gr.Markdown(visible=False)
                with gr.Row():
                    dd = gr.Dropdown([], label=t(lang, "whose"), visible=False, scale=3)
                    octv = gr.Dropdown(oct_choices(lang), value="auto", label=t(lang, "key"), visible=False, scale=2)
        row_out += [a, info, dd, octv]
        ctl += [dd, octv]
        dd.change(voice_info(i, lang), [state, dd, octv], [info])
        octv.change(voice_info(i, lang), [state, dd, octv], [info])
    with gr.Row():
        backing = gr.Dropdown([(t(lang, "by_voices"), BY_VOICES), (t(lang, "orig"), ORIG)], value=BY_VOICES,
                              label=t(lang, "backing"), scale=3, info=t(lang, "backing_info"))
        backing_oct = gr.Dropdown(oct_choices(lang), value="auto", label=t(lang, "backing_key"), scale=2,
                                  visible=False)
    backing.change(lambda b: gr.update(visible=b not in (BY_VOICES, ORIG)), [backing], [backing_oct])
    with gr.Accordion(t(lang, "table_acc"), open=False):
        gr.Markdown(t(lang, "table_help"))
        table = gr.Dataframe(headers=[t(lang, "h_time"), t(lang, "h_words"), t(lang, "h_voice")],
                             datatype=["str", "str", "str"], type="array", interactive=True, wrap=True,
                             column_count=3)
        upd = gr.Button(t(lang, "upd"))
    with gr.Row():
        voc = gr.Radio(voc_choices(lang), value="soft", label=t(lang, "voc"))
        reverb = gr.Slider(0, 2, value=1.0, step=0.1, label=t(lang, "reverb"))
        vocal_db = gr.Slider(-6, 6, value=0, step=0.5, label=t(lang, "vocal_db"))
    go = gr.Button(t(lang, "go"), variant="primary")
    out_status = gr.Markdown()
    with gr.Row():
        player = gr.Audio(label=t(lang, "player"), type="filepath", interactive=False)
        downloads = gr.File(label=t(lang, "downloads"), file_count="multiple", interactive=False)
    with gr.Accordion(t(lang, "log"), open=False):
        log_box = gr.Textbox(lines=14, max_lines=30, autoscroll=True, buttons=["copy"], label=t(lang, "log_box"))

    all_ctl = ctl + [backing, backing_oct]
    rendered = [summary, k, table] + row_out + [backing, backing_oct]
    assert len(rendered) == N_RENDER
    find.click(find_voices(lang), [file, link, lyrics, k, manual], [status, log_box, state] + rendered,
               js=READ_LYRICS_JS)
    upd.click(apply_table(lang), [state, table] + all_ctl, [status, state] + rendered)
    go.click(make_multi(lang), [state, table, voc, reverb, vocal_db] + all_ctl,
             [out_status, log_box, player, downloads, state])
