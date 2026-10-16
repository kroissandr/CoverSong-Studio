import json
import os
import time

from .cache import STAGES, Stage, params_key
from .stages import clean, convert, fx, input_stage, mix, separate
from .util import Timer, log


def song_dir_for(cfg, song_id, title=None):
    base = cfg["paths"]["output"]
    os.makedirs(base, exist_ok=True)
    # reuse an existing folder for this id, whatever its title
    for name in os.listdir(base):
        if name == song_id or name.endswith(f"_{song_id}"):
            return os.path.join(base, name)
    return os.path.join(base, f"{title}_{song_id}" if title else song_id)


def run_pipeline(cfg, song_input, voice_model=None, from_stage=None, until_stage="mix", excerpt=None, voice_plan=None):
    """voice_plan: several singers -> several models (multi-voice tab), see voices.plan()."""
    t0 = time.time()
    voice_model = voice_model or cfg.get("voice_model")
    kind, song_id, title = input_stage.identify(song_input)
    if excerpt:
        song_id = f"{song_id}_x{excerpt[0]:g}-{excerpt[1]:g}"
    song_dir = song_dir_for(cfg, song_id, title)
    os.makedirs(song_dir, exist_ok=True)
    log(f"Song: {song_input}")
    log(f"Output folder: {song_dir}")
    force = set(STAGES[STAGES.index(from_stage):]) if from_stage else set()
    last = STAGES.index(until_stage)

    def step(name, params, upstream, fn):
        st = Stage(song_dir, name)
        key = params_key(params, upstream)
        if name not in force and st.is_done(key):
            log(f"⏭ {name}: already done, taken from the cache")
            m = st.manifest()
            return st.outputs(), key, m.get("info", {})
        with Timer(f"Stage {name}"):
            st.begin()
            outputs, info = fn(st)
            st.finish(key, params, outputs, info)
        return outputs, key, info

    # 0. input
    inp, k_in, info_in = step("input", {"source": song_input.strip().strip('"'), "id": song_id,
                                        "excerpt": list(excerpt) if excerpt else None}, [],
                              lambda st: input_stage.run(song_input, kind, st, excerpt))
    title = info_in.get("title") or title or song_id
    if last < 1:
        return inp
    # 1. separation
    sep_params = {"chain": separate.CHAIN, "levels": "restored", "backing": "dry-v1"}
    sep, k_sep, _ = step("separate", sep_params, [k_in], lambda st: separate.run(cfg, inp["original"], st))
    if last < 2:
        return sep
    # 2. conversion
    conv_params = {"convert": cfg["convert"], "pitch_all": cfg["mix"].get("pitch_change_all", 0), "model": voice_model,
                   "model_sig": convert.model_signature(cfg, voice_model), "analysis": "words+consonants+autotune-v5-backing-wet", "f0fix": "edges-v1+range-v9-comfort+upper-v2+gate-v1+bk-rmvpe+fill-v1+airy-v1"}
    if voice_plan:
        conv_params["model_sig"] = {m: convert.model_signature(cfg, m) for m in
                                    set(voice_plan["assign"].values()) | {voice_plan.get("backing")}
                                    if m and m not in ("orig", "mute", "voices")}
        conv_params["voices"] = voice_plan
    conv, k_conv, conv_info = step("convert", conv_params, [k_sep],
                                   lambda st: convert.run(cfg, voice_model, sep, st, voice_plan))
    if last < 3:
        return conv
    # 3. clean-up: vocalizations, clicks, sibilants
    from . import breaths
    cln, k_cln, cln_info = step("clean", {"clean": cfg.get("clean", {}), "v": "consonants+repair+polish-v2+choir-v3-chords+breaths-v2",
                                          "breaths": breaths.bank_signature(cfg, voice_model)}, [k_conv],
                                lambda st: clean.run(cfg, conv, sep, st, voice_model, conv_info.get("semitones", 0),
                                                     conv_info.get("params"), conv_info.get("language"), voice_plan))
    if last < 4:
        return cln
    # 4. vocal fx
    fxo, k_fx, _ = step("fx", {"fx": "match-v14-hum-eq-wet-cap", "opt": cfg.get("fx") or {}}, [k_cln, k_sep], lambda st: fx.run(cfg, cln, sep, st, cln_info))
    if last < 5:
        return fxo
    # 5. mix
    name = "multi" if voice_plan else voice_model
    mixo, _, info_mix = step("mix", {"mix": cfg["mix"], "title": title, "model": name, "v": "one-sound-v5-sections"}, [k_fx, k_sep, k_cln],
                             lambda st: mix.run(cfg, fxo, sep, st, song_dir, title, name, inp["original"],
                                                                cln_info))
    intel = cln_info.get("intelligibility")
    if intel:
        log(f"Intelligibility: {intel['after']:.0%} of the original's words; weakest places: {intel['worst']}")
    log(f"Done in {time.time() - t0:.0f} s: {info_mix.get('final') or mixo.get('cover')}")
    with open(os.path.join(song_dir, "last_settings.json"), "w", encoding="utf-8") as f:
        json.dump({"voice_model": voice_model, "convert": cfg["convert"], "clean": cfg.get("clean"), "mix": cfg["mix"],
                   "separation": separate.CHAIN}, f, ensure_ascii=False, indent=2)
    return dict(mixo, song_dir=song_dir, title=title, final=info_mix.get("final"), final_wav=info_mix.get("final_wav"),
                stems=os.path.join(os.path.dirname(mixo["cover_wav"]), "stems"), octaves=conv_info.get("octaves"),
                lufs=info_mix.get("lufs"), seconds=time.time() - t0, intelligibility=cln_info.get("intelligibility"),
                closeness=info_mix.get("closeness"))
