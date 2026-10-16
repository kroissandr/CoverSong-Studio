// Vocal studio front end: voice choice, raw microphone capture (no browser processing), reference tones, live pitch,
// upload for grading, save / redo, model training. Everything runs locally against coversong.studio.server.
"use strict";

const I18N = {
  en: {
    title: "Vocal Studio", subtitle: "Recordings for a voice model: sing — get a grade and a tip in plain words right away.",
    covers: "← Covers", voice: "Voice (model)", newVoice: "+ New voice", openFolder: "Open folder", recorded: "Recorded:",
    min: "min", tasksDone: "tasks", mic: "Microphone", micPick: "— press \"Turn on the microphone\" —",
    micOn: "Turn on the microphone", micIsOn: "Microphone on ✓", level: "level",
    micHint: "Sing in headphones. The browser's sound processing is off — the clean signal is recorded, as in Audacity.",
    trainTitle: "Train the model", trainBtn: "Train", trainLog: "Log", songName: "Song (optional, for you)",
    listen: "▶ Listen", record: "● Record", stop: "■ Stop", skip: "Skip →", lower: "low", onNote: "on the note",
    higher: "high", save: "✓ Save and next", again: "↻ Again",
    keys: "Takes from 60 points are saved. Space — record, Enter — save, R — again.",
    newVoiceTitle: "New voice", voiceName: "Name (Latin letters, digits, - and _)",
    voiceType: "Voice type (the starting range; it adjusts as you sing)", create: "Create", cancel: "Cancel",
    noVoices: "No voices yet — press \"+ New voice\" to start.", of: "of", notRecorded: "not recorded yet",
    takesBest: (n, b) => `recorded: ${n}, best ${b}`, noNote: "no note", airOnly: "air only",
    freeSinging: "free singing", anySong: "any song you like", smooth: "smoothly, no steps", notes: n => `${n} notes`,
    head: "lightly, in head voice", chest: "chest voice",
    pressRecord: "Press \"Record\": a prompt plays, then a 3-2-1 count — and sing.",
    pressRecordSong: "Press \"Record\": after a 3-2-1 count, sing. Press \"Stop\" when the song ends.",
    getReady: "Get ready…", listenNote: "Listen to the note…", breathe: "Breathe!", sing: "Sing!",
    singing: t => `Recording ${t} — press "Stop" when you finish`, grading: "Grading…",
    inTune: n => `On the note ✓  ${n}`, goLower: n => `A bit lower ↓  (${n})`, goHigher: n => `A bit higher ↑  (${n})`,
    nowNote: n => `Now: ${n}`, noMic: "No access to the microphone: ", gradeErr: "Grading error: ",
    meta: (s, p) => `singing ${s} s · peak ${p} dB`, canSave: "Done — you can save it.",
    tryAgain: "Try again — tips below.", band: "green band — the note you need", bands: "green bands — the notes you need",
    melody: "your melody", notSaved: "Not saved: ", overload: "overload!", quiet: "quiet",
    range: (lo, hi, src) => `Planned range: ${lo}–${hi} (${src}).`,
    src: { model: "from the model", studio: "from your songs", type: "from the voice type — sing a few songs to refine it" },
    recordFor: v => `Recordings go to the voice "${v}".`,
    trainInfo: (m, lo, good) => `Recorded: ${m} min of takes. Training needs at least ${lo} min of singing, ${good}+ is better. ` +
      `It takes from an hour to several hours; the app keeps working, but covers will be slower meanwhile.`,
    trainHasModel: " The voice already has a model: training again replaces it with the new best version.",
    trainRunning: (v, s, p, t) => `Training ${v}: ${s} ${p}%${t ? " · " + t : ""}`,
    trainOther: v => `Training the voice "${v}" is in progress — one model at a time.`,
    trainDone: "Done — the model is installed and appears in the covers' model list (press ↻ there).",
    trainErr: e => `Stopped: ${e}`, trainAsk: "Start training this voice's model? It can take hours.",
    trainForce: "There are few recordings. Train anyway (the model will be weak)?",
    steps: { dataset: "preparing the recordings", pretrain: "base model", preprocess: "slicing", extract: "features",
      train: "training", index: "index", evaluate: "choosing the best version", install: "installing" },
  },
  ru: {
    title: "Вокальная студия", subtitle: "Записи для голосовой модели: спели — сразу оценка и подсказка простыми словами.",
    covers: "← Каверы", voice: "Голос (модель)", newVoice: "+ Новый голос", openFolder: "Открыть папку",
    recorded: "Записано:", min: "мин", tasksDone: "заданий", mic: "Микрофон",
    micPick: "— нажмите «Включить микрофон» —", micOn: "Включить микрофон", micIsOn: "Микрофон включён ✓",
    level: "уровень",
    micHint: "Пойте в наушниках. Обработка звука браузером выключена — записывается чистый сигнал, как в Audacity.",
    trainTitle: "Обучить модель", trainBtn: "Обучить", trainLog: "Журнал", songName: "Песня (необязательно, для вас)",
    listen: "▶ Послушать", record: "● Записать", stop: "■ Стоп", skip: "Пропустить →", lower: "ниже",
    onNote: "в ноту", higher: "выше", save: "✓ Сохранить и дальше", again: "↻ Ещё раз",
    keys: "Сохраняются записи от 60 баллов. Пробел — записать, Enter — сохранить, R — ещё раз.",
    newVoiceTitle: "Новый голос", voiceName: "Имя (латинские буквы, цифры, - и _)",
    voiceType: "Тип голоса (начальный диапазон; уточнится, пока вы поёте)", create: "Создать", cancel: "Отмена",
    noVoices: "Голосов пока нет — нажмите «+ Новый голос».", of: "из", notRecorded: "ещё не записано",
    takesBest: (n, b) => `записано: ${n}, лучший результат ${b}`, noNote: "без ноты", airOnly: "только воздух",
    freeSinging: "свободное пение", anySong: "любая песня", smooth: "плавно, без ступенек", notes: n => `${n} нот`,
    head: "легко, «головой»", chest: "грудным голосом",
    pressRecord: "Нажмите «Записать»: прозвучит подсказка, затем отсчёт 3-2-1 — и пойте.",
    pressRecordSong: "Нажмите «Записать»: после отсчёта 3-2-1 пойте. Песня закончилась — нажмите «Стоп».",
    getReady: "Приготовьтесь…", listenNote: "Слушайте ноту…", breathe: "Дышите!", sing: "Пойте!",
    singing: t => `Запись ${t} — закончите кнопкой «Стоп»`, grading: "Оцениваю…",
    inTune: n => `В ноту ✓  ${n}`, goLower: n => `Чуть ниже ↓  (${n})`, goHigher: n => `Чуть выше ↑  (${n})`,
    nowNote: n => `Сейчас: ${n}`, noMic: "Нет доступа к микрофону: ", gradeErr: "Ошибка оценки: ",
    meta: (s, p) => `пение ${s} с · пик ${p} дБ`, canSave: "Готово — можно сохранить.",
    tryAgain: "Попробуйте ещё раз — подсказки ниже.", band: "зелёная полоса — нужная нота",
    bands: "зелёные полосы — нужные ноты", melody: "ваша мелодия", notSaved: "Не сохранилось: ",
    overload: "перегруз!", quiet: "тихо",
    range: (lo, hi, src) => `Диапазон заданий: ${lo}–${hi} (${src}).`,
    src: { model: "по модели", studio: "по вашим песням", type: "по типу голоса — спойте пару песен, и он уточнится" },
    recordFor: v => `Записи идут в голос «${v}».`,
    trainInfo: (m, lo, good) => `Записано: ${m} мин. Для обучения нужно хотя бы ${lo} мин пения, лучше ${good}+. ` +
      `Обучение идёт от часа до нескольких часов; приложением можно пользоваться, но каверы в это время медленнее.`,
    trainHasModel: " У голоса уже есть модель: новое обучение заменит её лучшей новой версией.",
    trainRunning: (v, s, p, t) => `Обучение ${v}: ${s} ${p}%${t ? " · " + t : ""}`,
    trainOther: v => `Идёт обучение голоса «${v}» — по одной модели за раз.`,
    trainDone: "Готово — модель установлена и появится в списке моделей для каверов (нажмите там ↻).",
    trainErr: e => `Остановлено: ${e}`, trainAsk: "Начать обучение модели этого голоса? Это может занять несколько часов.",
    trainForce: "Записей мало. Всё равно обучить (модель выйдет слабой)?",
    steps: { dataset: "подготовка записей", pretrain: "базовая модель", preprocess: "нарезка", extract: "признаки",
      train: "обучение", index: "индекс", evaluate: "выбор лучшей версии", install: "установка" },
  },
};

const NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const RU = ["до", "до-диез", "ре", "ре-диез", "ми", "фа", "фа-диез", "соль", "соль-диез", "ля", "ля-диез", "си"];
const noteName = m => NOTES[((Math.round(m) % 12) + 12) % 12] + (Math.floor(Math.round(m) / 12) - 1);
const hz = m => 440 * Math.pow(2, (m - 69) / 12);
const $ = id => document.getElementById(id);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
};
const params = new URLSearchParams(location.search);
let LANG = params.get("lang") || store.get("studio.lang") || "en";
if (!I18N[LANG]) LANG = "en";
const L = I18N[LANG];
const noteLabel = m => LANG === "ru" ? `${RU[((Math.round(m) % 12) + 12) % 12]} (${noteName(m)})` : noteName(m);
const fmtTime = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

const S = { voice: null, voices: [], plan: null, progress: {}, cat: null, task: null, ctx: null, stream: null,
  node: null, src: null, recording: false, chunks: [], ring: new Float32Array(4096), ringPos: 0, livePts: [],
  result: null, timer: null, recStart: 0, train: null, trainPoll: null };
const isSong = t => t && t.cat === "songs";

function applyLang() {
  document.documentElement.lang = LANG;
  document.title = `${L.title} · CoverSong Studio`;
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const v = L[el.dataset.i18n];
    if (typeof v === "string") el.textContent = v;
  });
  document.querySelectorAll("[data-lang]").forEach(a => {
    a.classList.toggle("active", a.dataset.lang === LANG);
    a.onclick = e => { e.preventDefault(); store.set("studio.lang", a.dataset.lang);
      const p = new URLSearchParams(location.search); p.set("lang", a.dataset.lang); location.search = p.toString(); };
  });
  $("coversLink").href = LANG === "ru" ? "../ru/" : "../";
}

// ---------------------------------------------------------------- voices
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = await r.text();
    try { msg = JSON.parse(msg).detail || msg; } catch (e) { /* plain text */ }
    throw new Error(msg);
  }
  return r.json();
}
function form(obj) { const fd = new FormData(); for (const k in obj) fd.append(k, obj[k]); return fd; }

async function loadVoices(pick) {
  const j = await api(`api/voices?lang=${LANG}`);
  S.voices = j.voices;
  const sel = $("voiceSelect");
  sel.innerHTML = "";
  for (const v of j.voices) {
    const o = document.createElement("option");
    o.value = v.name; o.textContent = v.name + (v.model ? " ✓" : "") + (v.recorded_min ? ` · ${v.recorded_min} ${L.min}` : "");
    sel.appendChild(o);
  }
  const t = $("newVoiceType"); t.innerHTML = "";
  for (const ty of j.types) { const o = document.createElement("option"); o.value = ty.id; o.textContent = ty.name; t.appendChild(o); }
  t.value = "tenor";
  const want = pick || params.get("voice") || store.get("studio.voice") || j.default;
  const names = j.voices.map(v => v.name);
  S.voice = names.includes(want) ? want : (names[0] || null);
  if (S.voice) sel.value = S.voice;
  if (!S.voice) {
    $("voiceHint").textContent = L.noVoices;
    $("taskTitle").textContent = L.noVoices; $("taskHow").textContent = "";
    $("recBtn").disabled = true; $("trainBtn").disabled = true;
    $("newVoiceDlg").showModal();
    return;
  }
  store.set("studio.voice", S.voice);
  $("recBtn").disabled = false;
  await loadPlan();
}

$("voiceSelect").onchange = () => { S.voice = $("voiceSelect").value; store.set("studio.voice", S.voice); loadPlan(); };
$("newVoiceBtn").onclick = () => { $("newVoiceName").value = ""; $("newVoiceDlg").showModal(); };
$("newVoiceDlg").addEventListener("close", async () => {
  if ($("newVoiceDlg").returnValue !== "ok") return;
  const name = $("newVoiceName").value.trim();
  if (!name) return;
  try {
    const j = await api("api/voices", { method: "POST", body: form({ name, type: $("newVoiceType").value }) });
    await loadVoices(j.name);
  } catch (e) { alert(e.message); }
});
$("openBtn").onclick = () => { if (S.voice) fetch("api/open", { method: "POST", body: form({ voice: S.voice }) }); };

// ---------------------------------------------------------------- plan & navigation
async function loadPlan() {
  S.plan = await api(`api/plan?voice=${encodeURIComponent(S.voice)}&lang=${LANG}`);
  S.progress = S.plan.progress || {};
  const saved = store.get("studio.cat");
  S.cat = S.plan.categories.some(c => c.id === saved) ? saved : "songs";
  const r = S.plan.range;
  $("voiceHint").textContent = L.recordFor(S.voice) + " " +
    L.range(r.comfort_lo[1], r.comfort_hi[1], L.src[S.plan.range_source] || "");
  $("recMin").textContent = S.plan.recorded_min;
  renderCats();
  nextTask();
  showTrain(S.plan.train);
}

function doneTask(t) { return (S.progress[t.id] || {}).takes >= t.need; }

function renderCats() {
  const nav = $("cats");
  nav.innerHTML = "";
  let all = 0, done = 0;
  for (const c of S.plan.categories) {
    const ts = S.plan.tasks.filter(t => t.cat === c.id);
    const d = ts.filter(doneTask).length;
    all += ts.length; done += d;
    const b = document.createElement("button");
    b.className = "cat" + (c.id === S.cat ? " active" : "");
    b.innerHTML = `<b></b><small class="about"></small><small> · ${d} ${L.of} ${ts.length}</small>` +
      `<div class="bar"><div style="width:${ts.length ? 100 * d / ts.length : 0}%"></div></div>`;
    b.querySelector("b").textContent = c.title;
    b.querySelector(".about").textContent = c.about;
    b.onclick = () => { S.cat = c.id; store.set("studio.cat", c.id); renderCats(); nextTask(); };
    nav.appendChild(b);
  }
  $("doneCount").textContent = done;
  $("totalCount").textContent = all;
  $("overallBar").style.width = (all ? 100 * done / all : 0) + "%";
}

function nextTask(afterId) {
  const ts = S.plan.tasks.filter(t => t.cat === S.cat);
  let pool = ts.filter(t => !doneTask(t));
  if (afterId) {
    const i = pool.findIndex(t => t.id === afterId);
    if (i >= 0) pool = pool.slice(i + 1).concat(pool.slice(0, i));
  }
  showTask(pool[0] || ts[0]);
}

function showTask(t) {
  S.task = t;
  $("result").classList.add("hidden");
  if (!t) return;
  const cat = S.plan.categories.find(c => c.id === t.cat);
  $("taskCat").textContent = cat ? cat.title : t.cat;
  const p = S.progress[t.id];
  $("taskProgress").textContent = p ? L.takesBest(p.takes, Math.round(p.best)) : L.notRecorded;
  $("taskTitle").textContent = t.title;
  $("taskHow").textContent = t.how;
  $("songLabelRow").classList.toggle("hidden", !isSong(t));
  $("songLabel").value = "";
  $("listenBtn").classList.toggle("hidden", t.target == null);
  $("needleWrap").classList.toggle("hidden", t.target == null);
  $("liveCanvas").classList.toggle("hidden", t.target == null && !isSong(t));
  if (isSong(t)) {
    $("targetNote").textContent = L.freeSinging;
    $("targetRange").textContent = L.anySong;
  } else if (t.target == null) {
    $("targetNote").textContent = L.noNote;
    $("targetRange").textContent = L.airOnly;
  } else if (Array.isArray(t.target)) {
    $("targetNote").textContent = t.cat === "glide" ? `${noteName(t.target[0])} → ${noteName(t.target[1])}` :
      t.target.map(noteName).join(" ");
    $("targetRange").textContent = t.cat === "glide" ? L.smooth : L.notes(t.target.length);
  } else {
    $("targetNote").textContent = noteLabel(t.target);
    $("targetRange").textContent = t.register === "head" ? L.head : t.register === "chest" ? L.chest : "";
  }
  $("liveText").textContent = isSong(t) ? L.pressRecordSong : L.pressRecord;
  S.livePts = [];
  drawLive();
}

// ---------------------------------------------------------------- audio: microphone without processing
async function listMics() {
  try {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "audioinput");
    const sel = $("micSelect");
    if (!devs.length || !devs[0].label) return;  // no permission yet: labels are hidden
    sel.innerHTML = "";
    for (const d of devs) {
      const o = document.createElement("option");
      o.value = d.deviceId; o.textContent = d.label || L.mic;
      sel.appendChild(o);
    }
    const saved = store.get("studio.mic");
    if (saved && devs.some(d => d.deviceId === saved)) sel.value = saved;
  } catch (e) { /* no permission yet */ }
}

const WORKLET = `class Tap extends AudioWorkletProcessor {
  process(inputs) { const ch = inputs[0] && inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true; }
} registerProcessor("tap", Tap);`;

async function startMic() {
  if (S.stream) S.stream.getTracks().forEach(t => t.stop());
  const deviceId = $("micSelect").value;
  const audio = { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 };
  if (deviceId && deviceId.length > 5) audio.deviceId = { exact: deviceId };
  S.stream = await navigator.mediaDevices.getUserMedia({ audio });
  if (!S.ctx) {
    try { S.ctx = new AudioContext({ sampleRate: 48000 }); } catch (e) { S.ctx = new AudioContext(); }
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
    await S.ctx.audioWorklet.addModule(url);
  }
  if (S.ctx.state === "suspended") await S.ctx.resume();
  if (S.src) S.src.disconnect();
  S.src = S.ctx.createMediaStreamSource(S.stream);
  if (!S.node) {
    S.node = new AudioWorkletNode(S.ctx, "tap");
    S.node.port.onmessage = e => onSamples(e.data);
  }
  S.src.connect(S.node);
  await listMics();
  const track = S.stream.getAudioTracks()[0];
  if (track && track.getSettings().deviceId) { $("micSelect").value = track.getSettings().deviceId; store.set("studio.mic", track.getSettings().deviceId); }
  $("micBtn").textContent = L.micIsOn;
}

let levelPeak = 0, lastPitchAt = 0;
function onSamples(buf) {
  for (let i = 0; i < buf.length; i++) {
    S.ring[S.ringPos] = buf[i];
    S.ringPos = (S.ringPos + 1) % S.ring.length;
    const a = Math.abs(buf[i]); if (a > levelPeak) levelPeak = a;
  }
  if (S.recording) S.chunks.push(buf);
  const now = performance.now();
  if (now - lastPitchAt > 50) {
    lastPitchAt = now;
    const db = 20 * Math.log10(levelPeak + 1e-9);
    $("levelBar").style.width = Math.max(0, Math.min(100, (db + 60) / 60 * 100)) + "%";
    $("levelBar").style.background = db > -1 ? "var(--bad)" : db > -6 ? "var(--warn)" : "var(--good)";
    $("levelText").textContent = db > -1 ? L.overload : db < -45 ? L.quiet : `${db.toFixed(0)} dB`;
    levelPeak *= 0.5;
    if (S.recording) livePitch();
  }
}

function yin(x, sr) {
  const W = 1024, maxT = Math.min(W - 1, Math.floor(sr / 60)), minT = Math.floor(sr / 1000);
  let e = 0; for (let i = 0; i < x.length; i++) e += x[i] * x[i];
  if (e / x.length < 1e-6) return 0;
  const d = new Float32Array(maxT + 1);
  for (let t = 1; t <= maxT; t++) { let s = 0; for (let j = 0; j < W; j++) { const q = x[j] - x[j + t]; s += q * q; } d[t] = s; }
  let run = 0;
  for (let t = 1; t <= maxT; t++) {
    run += d[t]; const v = d[t] * t / (run || 1);
    if (t >= minT && v < 0.15) {
      while (t + 1 <= maxT && d[t + 1] * (t + 1) / (run + d[t + 1]) < v) { t++; run += d[t]; }
      return sr / t;
    }
  }
  return 0;
}

function livePitch() {
  const t = S.task;
  if (isSong(t)) {
    const el = (S.ctx.currentTime - S.recStart);
    const n = 2048, x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = S.ring[(S.ringPos - n + i + S.ring.length) % S.ring.length];
    const f = yin(x, S.ctx.sampleRate);
    const m = f ? 69 + 12 * Math.log2(f / 440) : null;
    $("liveText").textContent = L.singing(fmtTime(el)) + (m ? " · " + L.nowNote(noteLabel(m)) : "");
    S.livePts.push(m); drawLive();
    return;
  }
  if (t.target == null) return;
  const n = 2048, x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = S.ring[(S.ringPos - n + i + S.ring.length) % S.ring.length];
  const f = yin(x, S.ctx.sampleRate);
  const needle = $("needle");
  if (!f) { needle.style.opacity = .3; S.livePts.push(null); drawLive(); return; }
  const m = 69 + 12 * Math.log2(f / 440);
  const ref = Array.isArray(t.target) ? nearest(m, t.target) : t.target;
  let cents = (m - ref) * 100;
  if (Math.abs(cents) > 600) cents = ((cents + 600) % 1200 + 1200) % 1200 - 600; // octave slip: show the note error
  const pos = 50 + Math.max(-50, Math.min(50, cents / 2));
  needle.style.left = pos + "%";
  needle.style.opacity = 1;
  needle.style.background = Math.abs(cents) < 35 ? "var(--good)" : Math.abs(cents) < 80 ? "var(--warn)" : "var(--bad)";
  $("liveText").textContent = Math.abs(cents) < 35 ? L.inTune(noteLabel(m)) :
    (cents > 0 ? L.goLower(noteLabel(m)) : L.goHigher(noteLabel(m)));
  S.livePts.push(m);
  drawLive();
}
function nearest(m, arr) { const lo = Math.min(...arr), hi = Math.max(...arr); return Math.max(lo, Math.min(hi, m)); }

function drawLive() {
  const c = $("liveCanvas"), g = c.getContext("2d");
  c.width = c.clientWidth * devicePixelRatio; c.height = 120 * devicePixelRatio;
  g.clearRect(0, 0, c.width, c.height);
  const t = S.task; if (!t || (t.target == null && !isSong(t))) return;
  let lo, hi, tg = [];
  if (isSong(t)) {
    const r = S.plan.range; lo = r.lo[0] - 2; hi = r.hi[0] + 2;
  } else {
    tg = Array.isArray(t.target) ? t.target : [t.target];
    lo = Math.min(...tg) - 3; hi = Math.max(...tg) + 3;
  }
  const y = m => c.height - (m - lo) / (hi - lo) * c.height;
  const cs = getComputedStyle(document.body);
  g.fillStyle = cs.getPropertyValue("--band");
  for (const m of new Set(tg)) g.fillRect(0, y(m + 0.35), c.width, y(m - 0.35) - y(m + 0.35));
  g.strokeStyle = cs.getPropertyValue("--accent"); g.lineWidth = 2 * devicePixelRatio;
  g.beginPath(); let started = false;
  const pts = S.livePts.slice(-200);
  pts.forEach((m, i) => {
    if (m == null) { started = false; return; }
    const xx = i / 200 * c.width, yy = y(Math.max(lo, Math.min(hi, m)));
    if (!started) { g.moveTo(xx, yy); started = true; } else g.lineTo(xx, yy);
  });
  g.stroke();
}

// ---------------------------------------------------------------- reference tones
function tone(freq, start, dur, glideTo) {
  const ctx = S.ctx, o = ctx.createOscillator(), gn = ctx.createGain();
  const real = new Float32Array([0, 1, .5, .32, .2, .12, .08, .05]), imag = new Float32Array(real.length);
  o.setPeriodicWave(ctx.createPeriodicWave(real, imag));
  o.frequency.setValueAtTime(freq, start);
  if (glideTo) o.frequency.exponentialRampToValueAtTime(glideTo, start + dur);
  gn.gain.setValueAtTime(0, start);
  gn.gain.linearRampToValueAtTime(0.18, start + 0.03);
  gn.gain.setValueAtTime(0.18, start + Math.max(0.05, dur - 0.08));
  gn.gain.linearRampToValueAtTime(0, start + dur);
  o.connect(gn).connect(ctx.destination);
  o.start(start); o.stop(start + dur + 0.05);
}

function click(start, hi) {
  const ctx = S.ctx, o = ctx.createOscillator(), gn = ctx.createGain();
  o.frequency.value = hi ? 1500 : 1000;
  gn.gain.setValueAtTime(0.25, start); gn.gain.exponentialRampToValueAtTime(0.001, start + 0.06);
  o.connect(gn).connect(ctx.destination); o.start(start); o.stop(start + 0.07);
}

function playReference() {   // returns seconds
  const t = S.task, now = S.ctx.currentTime + 0.05;
  if (t.target == null) return 0;
  if (t.cat === "glide") { tone(hz(t.target[0]), now, 3.0, hz(t.target[1])); return 3.1; }
  if (Array.isArray(t.target)) { t.target.forEach((m, i) => tone(hz(m), now + i * 0.45, 0.42)); return t.target.length * 0.45 + 0.1; }
  if (t.cat === "staccato") { for (let i = 0; i < 4; i++) tone(hz(t.target), now + i * 0.5, 0.2); return 2.1; }
  tone(hz(t.target), now, 1.5); return 1.6;
}

// ---------------------------------------------------------------- record → grade
async function ensureMic() { if (!S.stream) await startMic(); }

async function record() {
  if (!S.task || !S.voice) return;
  if (S.recording) { stopRecording(); return; }
  try { await ensureMic(); } catch (e) { alert(L.noMic + e.message); return; }
  $("result").classList.add("hidden");
  $("recBtn").disabled = true; $("listenBtn").disabled = true;
  $("liveText").textContent = S.task.target == null ? L.getReady : L.listenNote;
  const ref = playReference();
  await sleep(ref * 1000 + 300);
  const t0 = S.ctx.currentTime + 0.1;
  for (let i = 0; i < 3; i++) click(t0 + i * 0.6, i === 2);
  for (const n of [3, 2, 1]) { $("liveText").textContent = `${n}…`; await sleep(600); }
  S.chunks = []; S.livePts = []; S.recording = true; S.recStart = S.ctx.currentTime;
  $("recBtn").disabled = false; $("recBtn").textContent = L.stop; $("recBtn").classList.add("recording");
  $("liveText").textContent = isSong(S.task) ? L.singing("0:00") : S.task.target == null ? L.breathe : L.sing;
  S.timer = setTimeout(stopRecording, (S.task.seconds + (isSong(S.task) ? 0 : 1.2)) * 1000);
}

function stopRecording() {
  if (!S.recording) return;
  clearTimeout(S.timer);
  S.recording = false;
  $("recBtn").textContent = L.record; $("recBtn").classList.remove("recording");
  $("listenBtn").disabled = false;
  $("liveText").textContent = L.grading;
  const n = S.chunks.reduce((a, c) => a + c.length, 0), x = new Float32Array(n);
  let p = 0; for (const c of S.chunks) { x.set(c, p); p += c.length; }
  S.chunks = [];
  upload(x, S.ctx.sampleRate);
}

function wavFile(x, sr, int16) {   // 32-bit float for short takes, 16-bit for whole songs (half the size)
  const bps = int16 ? 2 : 4, b = new ArrayBuffer(44 + x.length * bps), v = new DataView(b);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + x.length * bps, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, int16 ? 1 : 3, true); v.setUint16(22, 1, true); v.setUint32(24, sr, true);
  v.setUint32(28, sr * bps, true); v.setUint16(32, bps, true); v.setUint16(34, bps * 8, true); w(36, "data");
  v.setUint32(40, x.length * bps, true);
  if (int16) {
    const out = new Int16Array(b, 44);
    for (let i = 0; i < x.length; i++) out[i] = Math.max(-32768, Math.min(32767, Math.round(x[i] * 32767)));
  } else new Float32Array(b, 44).set(x);
  return new Blob([b], { type: "audio/wav" });
}

async function upload(x, sr) {
  const blob = wavFile(x, sr, isSong(S.task));
  $("playback").src = URL.createObjectURL(blob);
  const fd = form({ voice: S.voice, task_id: S.task.id, lang: LANG });
  fd.append("audio", blob, "take.wav");
  try {
    showResult(await api("api/analyze", { method: "POST", body: fd }));
  } catch (e) { $("liveText").textContent = L.gradeErr + e.message; }
}

function showResult(res) {
  S.result = res;
  $("result").classList.remove("hidden");
  const sc = $("score");
  sc.textContent = res.score;
  sc.style.background = res.score >= 80 ? "var(--good)" : res.score >= 60 ? "var(--warn)" : "var(--bad)";
  $("verdict").textContent = res.verdict;
  $("resultMeta").textContent = L.meta(res.voiced_s.toFixed(1), res.peak_db.toFixed(0));
  const ul = $("tips"); ul.innerHTML = "";
  for (const t of res.tips) { const li = document.createElement("li"); li.className = t.kind; li.textContent = t.text; ul.appendChild(li); }
  $("saveBtn").disabled = res.score < 60;
  $("liveText").textContent = res.score >= 60 ? L.canSave : L.tryAgain;
  $("curve").classList.toggle("hidden", !(res.curve.t || []).length);
  drawCurve(res);
  $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function drawCurve(res) {
  const c = $("curve"), g = c.getContext("2d");
  c.width = c.clientWidth * devicePixelRatio; c.height = 180 * devicePixelRatio;
  g.clearRect(0, 0, c.width, c.height);
  const cs = getComputedStyle(document.body), cv = res.curve, t = cv.t || [];
  if (!t.length) return;
  const tx = s => s / t[t.length - 1] * c.width;
  let vals, lo, hi, bands = [], label;
  if (cv.cents) { vals = cv.cents; lo = -300; hi = 300; bands = [[-35, 35]]; label = L.band; }
  else if (cv.free) {
    vals = cv.midi; const ok = vals.filter(v => v != null).sort((a, b) => a - b);
    lo = ok.length ? ok[Math.floor(ok.length * 0.02)] - 2 : 40; hi = ok.length ? ok[Math.floor(ok.length * 0.98)] + 2 : 70;
    label = L.melody;
  } else {
    vals = cv.midi; const tg = cv.notes || [cv.from, cv.to];
    lo = Math.min(...tg) - 3; hi = Math.max(...tg) + 3;
    bands = [...new Set(tg)].map(m => [m - 0.35, m + 0.35]); label = L.bands;
  }
  const y = v => c.height - (v - lo) / (hi - lo) * c.height;
  g.fillStyle = cs.getPropertyValue("--band");
  for (const [a, b] of bands) g.fillRect(0, y(b), c.width, y(a) - y(b));
  g.strokeStyle = cs.getPropertyValue("--accent"); g.lineWidth = 2 * devicePixelRatio; g.beginPath();
  let on = false;
  vals.forEach((v, i) => { if (v == null) { on = false; return; } const X = tx(t[i]), Y = y(Math.max(lo, Math.min(hi, v)));
    if (!on) { g.moveTo(X, Y); on = true; } else g.lineTo(X, Y); });
  g.stroke();
  g.fillStyle = cs.getPropertyValue("--muted"); g.font = `${12 * devicePixelRatio}px sans-serif`;
  g.fillText(label, 8 * devicePixelRatio, 16 * devicePixelRatio);
}

async function save() {
  if (!S.result || S.result.score < 60) return;
  try {
    const j = await api("api/save", { method: "POST", body: form({ voice: S.voice, task_id: S.task.id,
      take: S.result.take, score: S.result.score, label: isSong(S.task) ? $("songLabel").value : "" }) });
    S.progress = j.progress; S.result = null;
    $("recMin").textContent = j.recorded_min;
    renderCats();
    nextTask(S.task.id);
    refreshTrain();
  } catch (e) { alert(L.notSaved + e.message); }
}

async function again() {
  if (S.result) fetch("api/discard", { method: "POST", body: form({ voice: S.voice, take: S.result.take }) });
  S.result = null;
  record();
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

// ---------------------------------------------------------------- training
function showTrain(st) {
  S.train = st;
  const mine = st.voice === S.voice;
  let info = L.trainInfo($("recMin").textContent, st.min_minutes, st.good_minutes);
  if (S.plan && S.plan.model) info += L.trainHasModel;
  $("trainInfo").textContent = info;
  const idx = Math.max(0, st.steps.indexOf(st.step));
  const total = mine ? (idx + (st.frac || 0)) / st.steps.length : 0;
  $("trainBar").style.width = (st.running && mine ? 100 * total : (mine && st.done ? 100 : 0)) + "%";
  if (st.running && !mine) $("trainStep").textContent = L.trainOther(st.voice);
  else if (st.running) $("trainStep").textContent = L.trainRunning(st.voice, L.steps[st.step] || st.step || "",
    Math.round(100 * (st.frac || 0)), st.text);
  else if (mine && st.error) $("trainStep").textContent = L.trainErr(st.error);
  else if (mine && st.done) $("trainStep").textContent = L.trainDone;
  else $("trainStep").textContent = "";
  $("trainBtn").disabled = st.running || !S.voice;
  $("trainLog").textContent = mine ? (st.log || []).join("\n") : "";
  const box = $("trainLog"); box.scrollTop = box.scrollHeight;
  clearTimeout(S.trainPoll);
  if (st.running) S.trainPoll = setTimeout(refreshTrain, 3000);
}

async function refreshTrain() {
  try { showTrain(await api(`api/train?voice=${encodeURIComponent(S.voice || "")}`)); } catch (e) { /* server busy */ }
}

$("trainBtn").onclick = async () => {
  if (!S.voice || !confirm(L.trainAsk)) return;
  const st = S.train || {};
  const force = parseFloat($("recMin").textContent) < (st.min_minutes || 3) && confirm(L.trainForce);
  try { showTrain(await api("api/train", { method: "POST", body: form({ voice: S.voice, force }) })); }
  catch (e) { alert(e.message); }
};

// ---------------------------------------------------------------- wiring
$("micBtn").onclick = () => startMic().catch(e => alert(L.noMic + e.message));
$("micSelect").onchange = () => { store.set("studio.mic", $("micSelect").value); startMic().catch(() => {}); };
$("listenBtn").onclick = async () => { await ensureMic(); playReference(); };
$("recBtn").onclick = record;
$("skipBtn").onclick = () => nextTask(S.task && S.task.id);
$("saveBtn").onclick = save;
$("againBtn").onclick = again;
document.addEventListener("keydown", e => {
  if (["SELECT", "INPUT", "TEXTAREA"].includes(e.target.tagName) || $("newVoiceDlg").open) return;
  if (e.code === "Space") { e.preventDefault(); record(); }
  else if (e.code === "Enter") { e.preventDefault(); save(); }
  else if (e.code === "KeyR") { e.preventDefault(); again(); }
});
window.addEventListener("resize", () => { drawLive(); if (S.result) drawCurve(S.result); });
applyLang();
listMics();
loadVoices().catch(e => { $("taskTitle").textContent = e.message; });
