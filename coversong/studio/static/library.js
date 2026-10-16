// The covers library: lists song_output, plays covers and originals, downloads, opens folders.
"use strict";
const LI18N = {
  en: { title: "Covers library", subtitle: "Every cover you have made, newest first.", search: "Search by song or voice",
    refresh: "↻ Refresh", openAll: "Open the song_output folder", original: "Original", voice: "Voice",
    folder: "Folder", count: n => `${n} songs`, covers: n => n === 1 ? "1 cover" : `${n} covers`,
    empty: "No covers yet — make one on the \"One voice\" or \"Several voices\" tab.", nothing: "Nothing found.",
    multi: "several voices" },
  ru: { title: "Библиотека каверов", subtitle: "Все сделанные каверы, сначала новые.", search: "Поиск по песне или голосу",
    refresh: "↻ Обновить", openAll: "Открыть папку song_output", original: "Оригинал", voice: "Голос",
    folder: "Папка", count: n => `песен: ${n}`, covers: n => `каверов: ${n}`,
    empty: "Каверов пока нет — сделайте первый на вкладке «Один голос» или «Несколько голосов».",
    nothing: "Ничего не найдено.", multi: "несколько голосов" },
};
const qs = new URLSearchParams(location.search);
let LANG = qs.get("lang") || (() => { try { return localStorage.getItem("studio.lang"); } catch (e) { return null; } })() || "en";
if (!LI18N[LANG]) LANG = "en";
const T = LI18N[LANG];
const $ = id => document.getElementById(id);
let SONGS = [], SIG = "";

document.documentElement.lang = LANG;
document.querySelectorAll("[data-i18n]").forEach(el => { if (typeof T[el.dataset.i18n] === "string") el.textContent = T[el.dataset.i18n]; });
document.querySelectorAll("[data-i18n-ph]").forEach(el => { el.placeholder = T[el.dataset.i18nPh]; });

const media = (p, dl) => `api/media?p=${encodeURIComponent(p)}${dl ? "&dl=1" : ""}`;
const when = t => new Date(t * 1000).toLocaleString(LANG === "ru" ? "ru-RU" : "en-GB",
  { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
function openFolder(p) { const fd = new FormData(); fd.append("p", p || ""); fetch("api/library/open", { method: "POST", body: fd }); }

function row(name, sub, src, links, cls) {
  const r = el("div", "ver" + (cls ? " " + cls : ""));
  const who = el("div", "who"); who.appendChild(el("b", null, name)); if (sub) who.appendChild(el("small", "muted", sub));
  const a = el("audio"); a.controls = true; a.preload = "none"; a.src = src;
  a.addEventListener("play", () => document.querySelectorAll("audio").forEach(o => { if (o !== a) o.pause(); }));
  const act = el("div", "act");
  for (const [label, href] of links) { const l = el("a", null, label); l.href = href; l.setAttribute("download", ""); act.appendChild(l); }
  r.append(who, a, act);
  return r;
}

function render() {
  const q = $("q").value.trim().toLowerCase();
  const list = $("list"); list.innerHTML = "";
  const shown = SONGS.filter(s => !q || s.title.toLowerCase().includes(q) || s.covers.some(c => c.model.toLowerCase().includes(q)));
  $("count").textContent = T.count(SONGS.length);
  if (!SONGS.length) { list.appendChild(el("div", "card empty", T.empty)); return; }
  if (!shown.length) { list.appendChild(el("div", "card empty", T.nothing)); return; }
  for (const s of shown) {
    const card = el("article", "card song");
    const head = el("div", "song-head");
    head.appendChild(el("h3", null, s.title));
    const meta = el("div", "act");
    const open = el("button", null, "📂 " + T.folder); open.onclick = () => openFolder(s.folder);
    meta.appendChild(open);
    head.appendChild(meta);
    card.appendChild(head);
    card.appendChild(el("p", "muted small", T.covers(s.covers.length)));
    for (const c of s.covers) {
      const links = [["MP3", media(c.mp3, true)]];
      if (c.wav) links.push(["WAV", media(c.wav, true)]);
      card.appendChild(row(c.model === "multi" ? T.multi : c.model, when(c.time) + ` · ${c.size_mb} MB`, media(c.mp3), links));
    }
    if (s.original) card.appendChild(row(T.original, "", media(s.original), [], "orig"));
    list.appendChild(card);
  }
}

async function load() {
  try {
    const r = await fetch("api/library");
    const j = await r.json();
    const sig = JSON.stringify(j.songs.map(s => [s.folder, s.covers.map(c => c.mp3 + c.time)]));
    if (sig === SIG) return;
    const playing = [...document.querySelectorAll("audio")].some(a => !a.paused);
    SONGS = j.songs; SIG = sig;
    if (!playing) render();
  } catch (e) { /* the app is busy; next time */ }
}

$("q").addEventListener("input", render);
$("refreshBtn").onclick = () => { SIG = ""; load(); };
$("openRoot").onclick = () => openFolder("");
load();
setInterval(load, 10000);
