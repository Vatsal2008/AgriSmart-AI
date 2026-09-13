"""Record every phrase the offline app speaks, in all 13 languages, with Meta's MMS-TTS voices (one VITS voice
per language, trained on native speakers). Output per language: offline/web/audio/<lang>/<n>.ogg and
offline/web/audio/<lang>/index.json ({"clips": {phrase_key: "audio/<lang>/<n>.ogg"}, "texts": {file: text}}),
which the page loads for the chosen language and plays offline.

    python offline/tools/build_audio.py [--langs hi,ta] [--limit 5] [--threads 4]

Reads offline/build/speech_inventory.json (made by build_offline_data.py). Identical texts share one clip, and a
clip whose text has not changed since the last run is kept, so re-runs only record what is new. Languages are
independent, so several processes can each take a few (--langs) and run side by side.
Voices: facebook/mms-tts-<code>, licence CC BY-NC 4.0 (non-commercial use).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel

ROOT = Path(__file__).resolve().parents[2]
INV = ROOT / "offline" / "build" / "speech_inventory.json"
AUDIO = ROOT / "offline" / "web" / "audio"
MMS = {"en": "eng", "hi": "hin", "bn": "ben", "mr": "mar", "te": "tel", "ta": "tam", "gu": "guj",
       "ur": "urd-script_arabic", "kn": "kan", "or": "ory", "ml": "mal", "pa": "pan", "as": "asm"}

ap = argparse.ArgumentParser()
ap.add_argument("--langs", default=",".join(MMS))
ap.add_argument("--limit", type=int, default=0, help="record only the first N phrases per language (for a test)")
ap.add_argument("--threads", type=int, default=0, help="CPU threads for this process (default: all)")
args = ap.parse_args()
if args.threads:
    torch.set_num_threads(args.threads)

inv = json.loads(INV.read_text(encoding="utf-8"))
roman = None

for lang in args.langs.split(","):
    t0 = time.time()
    repo = f"facebook/mms-tts-{MMS[lang]}"
    tok = AutoTokenizer.from_pretrained(repo)
    model = VitsModel.from_pretrained(repo).eval()
    uses_uroman = getattr(tok, "is_uroman", False)
    if uses_uroman and roman is None:
        import uroman
        roman = uroman.Uroman()
    out_dir = AUDIO / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    old = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"texts": {}}
    reuse = {text: f for f, text in old.get("texts", {}).items() if (AUDIO.parent / f).exists()}
    items = list(inv[lang].items())[: args.limit or None]
    by_text, clips, texts, silent, short, recorded = {}, {}, {}, [], [], 0
    used_names = set(Path(f).name for f in reuse.values())
    next_n = 0
    for key, text in items:
        if text in by_text:
            clips[key] = by_text[text]
            continue
        if text in reuse:
            by_text[text] = clips[key] = reuse[text]
            texts[reuse[text]] = text
            continue
        spoken = roman.romanize_string(text) if uses_uroman else text
        ids = tok(spoken, return_tensors="pt")
        if ids["input_ids"].shape[1] < 2:
            short.append(key)
            continue
        torch.manual_seed(0)  # the same voice every run
        with torch.no_grad():
            wav = model(**ids).waveform[0].numpy()
        peak = float(np.abs(wav).max())
        if peak < 0.01:
            silent.append(key)
            continue
        wav = (wav / peak * 0.9).astype(np.float32)  # even loudness across clips
        while f"{next_n:04d}.ogg" in used_names:
            next_n += 1
        name = f"{next_n:04d}.ogg"
        used_names.add(name)
        sf.write(out_dir / name, wav, model.config.sampling_rate, format="OGG", subtype="VORBIS")
        rel = f"audio/{lang}/{name}"
        by_text[text] = clips[key] = rel
        texts[rel] = text
        recorded += 1
        if recorded % 50 == 0:  # save progress, so an interrupted run resumes here
            index_path.write_text(json.dumps({"clips": clips, "texts": texts}, ensure_ascii=False), encoding="utf-8")
    index_path.write_text(json.dumps({"clips": clips, "texts": texts}, ensure_ascii=False), encoding="utf-8")
    for f in out_dir.glob("*.ogg"):  # clips of phrases that no longer exist
        if f"audio/{lang}/{f.name}" not in texts:
            f.unlink()
    size = sum(f.stat().st_size for f in out_dir.glob("*.ogg")) / 1e6
    print(f"{lang}: {len(clips)}/{len(items)} phrases, {len(texts)} clips ({recorded} new), {size:.1f} MB, "
          f"{time.time() - t0:.0f}s (uroman: {uses_uroman}); silent {len(silent)} {silent[:3]}, "
          f"untokenizable {len(short)} {short[:3]}", flush=True)
