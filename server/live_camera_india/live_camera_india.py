"""AgriSmart India: multi-crop live scan, in 12 Indian languages plus English.

Point the laptop camera at a leaf (or at a phone showing a leaf photo), or load a photo from this
machine. Choose the crop first for a sharper guess, or let the model guess it. Shows every class the
model is weighing, not just the top one, and speaks the result aloud in the chosen language.

Run from the project folder:
    .venv/Scripts/python server/live_camera_india/live_camera_india.py
then open the browser tab it starts and press "Start camera" (or use "Test a photo from your device").
Binds to every network interface by default, so a phone on the same Wi-Fi can reach it too, the same as
server/live_camera/live_camera.py; pass --host 127.0.0.1 to keep it laptop-only.

Model: looks for model/india/model.pt (trained by model/training/india/02_train). Until that exists,
the page runs in a clearly-labelled MOCK mode against the same 37-crop, 179-class taxonomy, so the whole
app can be built and checked before training finishes -- mock results are never shown as real.
"""
import argparse
import hashlib
import io
import json
import socket
import sys
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
APP = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model" / "india"
PAGE = (APP / "live_camera_india.html").read_bytes()
TRANSLATIONS_BYTES = (APP / "india_translations.json").read_bytes()
# what-to-do advice per disease, 13 languages (built by build_advice.py from the model's class list)
ADVICE_BYTES = (APP / "india_advice.json").read_bytes() if (APP / "india_advice.json").exists() else b"{}"
MAX_UPLOAD = 8 * 1024 * 1024
TOP_N = 8  # how many classes the statistics panel shows

INFER = None  # a single long-lived worker thread; see live_camera.py for why (avoids an OMP/memory crash
               # from every web request spinning up its own PyTorch thread pool)


# ---------------------------------------------------------------------------
# Real model, if it has been trained, else a clearly-marked mock over the same taxonomy
# ---------------------------------------------------------------------------
def load_backend():
    weights = MODEL_DIR / "model.pt"
    if weights.exists():
        sys.path.insert(0, str(MODEL_DIR))
        import predict as india_predict  # model/india/predict.py, written by the training notebook
        india_predict.load()  # fail fast if the checkpoint is broken
        crops = india_predict.crops_and_labels()
        print(f"Loaded the trained India model: {sum(len(v) for v in crops.values())} classes across "
              f"{len(crops)} crops.", flush=True)
        return {"mode": "real", "crops": crops, "predict_proba": india_predict.predict_proba}

    fallback = json.loads((APP / "india_crops_fallback.json").read_text(encoding="utf-8"))
    crops = {crop: info["labels"] for crop, info in fallback.items()}
    print(f"No trained model at {weights} yet -- running in MOCK mode "
          f"({sum(len(v) for v in crops.values())} classes across {len(crops)} crops from the "
          "data-prep taxonomy). Mock results are for building/testing the app only.", flush=True)

    def mock_predict_proba(image, crop=None):
        # Deterministic per-image "randomness" (same photo always gets the same mock answer), weighted
        # so "Healthy" usually wins -- a plausible-looking placeholder, never claimed to be a real guess.
        digest = hashlib.sha256(image.tobytes()[:65536]).digest()
        seed = int.from_bytes(digest[:8], "big")
        chosen_crop = crop if crop in crops else list(crops)[seed % len(crops)]
        labels = crops[chosen_crop]
        weights = []
        for i, label in enumerate(labels):
            base = 0.55 if label == "Healthy" else 0.45 / max(1, len(labels) - 1)
            jitter = ((seed >> (i * 3)) % 23) / 100.0
            weights.append(max(0.01, base + jitter - 0.1))
        total = sum(weights)
        probs = {f"{chosen_crop}::{label}": w / total for label, w in zip(labels, weights)}
        return probs

    return {"mode": "mock", "crops": crops, "predict_proba": mock_predict_proba}


BACKEND = None  # set in main()


def describe(gclass, p):
    crop, label = gclass.split("::", 1)
    return {"crop": crop, "label": label, "p": round(p, 4), "healthy": label == "Healthy"}


# Look-alike plants (squash/cucumber, bean/soybean, tomato/potato, rice/wheat, cabbage/cauliflower...) are
# where the model goes wrong most when no crop is chosen: on field photos it picks the right plant only
# ~87% of the time. So the answer is judged per PLANT first (each crop's classes' probabilities added up),
# and if the top plant is not clearly ahead, the page asks which crop it is instead of naming a disease.
CROP_SURE = 0.50    # the top plant needs at least this share of the probability...
CROP_MARGIN = 0.35  # ...and must lead the runner-up plant by at least this much (a 0.75 share rule asked
                    # about clear photos too, e.g. grape 0.70 vs apple 0.04; the margin is what matters)
MISMATCH = 0.15     # a chosen crop holding less than this share (and not on top) is flagged as a mismatch


def crop_scores(probs):
    agg = {}
    for g, p in probs.items():
        c = g.split("::", 1)[0]
        agg[c] = agg.get(c, 0.0) + p
    return sorted(agg.items(), key=lambda kv: -kv[1])


def within(probs, crop):
    """The distribution renormalised over one crop's classes (all other crops' classes dropped)."""
    mass = sum(p for g, p in probs.items() if g.startswith(crop + "::"))
    return {g: p / mass for g, p in probs.items() if g.startswith(crop + "::")} if mass > 0 else probs


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/translations.json":
            self._send(200, TRANSLATIONS_BYTES, "application/json; charset=utf-8")
        elif self.path == "/advice.json":
            self._send(200, ADVICE_BYTES, "application/json; charset=utf-8")
        elif self.path == "/health":
            self._json({"ok": True, "mode": BACKEND["mode"],
                       "crops": {c: len(v) for c, v in BACKEND["crops"].items()},
                       "total_classes": sum(len(v) for v in BACKEND["crops"].values())})
        elif self.path == "/crops":
            self._json(BACKEND["crops"])
        else:
            self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/predict":
            return self._send(404, b"Not found", "text/plain")
        crop = None
        if "?" in self.path:
            from urllib.parse import parse_qs
            q = parse_qs(self.path.split("?", 1)[1])
            crop = (q.get("crop") or [None])[0]
            if crop == "":
                crop = None
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_UPLOAD:
            return self._json({"error": "Send one JPEG or PNG image of up to 8 MB."}, 400)
        try:
            image = Image.open(io.BytesIO(self.rfile.read(length))).convert("RGB")
        except Exception:
            return self._json({"error": "That file could not be read as an image."}, 400)

        if crop not in BACKEND["crops"]:
            crop = None
        t = time.perf_counter()
        # The real model always scores all crops' classes in one pass; a chosen crop narrows the answer
        # afterwards. (Asking the model for the chosen crop only, as an earlier version did, made the
        # detected crop always equal the chosen one, so a wrong crop choice could never be noticed.)
        ask = None if BACKEND["mode"] == "real" else crop
        raw = INFER.submit(BACKEND["predict_proba"], image, ask).result()
        ranked = crop_scores(raw)
        detected_crop = ranked[0][0] if ranked else None
        share = dict(ranked)
        ambiguous = False
        if crop:
            probs = within(raw, crop)
            crop_matches = detected_crop == crop or share.get(crop, 0.0) >= MISMATCH
        else:
            crop_matches = True
            ambiguous = len(ranked) > 1 and (ranked[0][1] < CROP_SURE or ranked[0][1] - ranked[1][1] < CROP_MARGIN)
            # sure of the plant: rank diseases within it, so a stray class of a look-alike can't win
            probs = raw if ambiguous else within(raw, detected_crop)
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:TOP_N]
        self._json({
            "mode": BACKEND["mode"],
            "top": [describe(g, p) for g, p in top],
            "detected_crop": detected_crop,
            "crop_scores": [{"crop": c, "p": round(p, 4)} for c, p in ranked[:4]],
            "crop_ambiguous": ambiguous,
            "crop_given": crop,
            "crop_matches": crop_matches,
            "ms": round((time.perf_counter() - t) * 1000),
        })

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def get_wifi_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    global BACKEND, INFER
    ap = argparse.ArgumentParser(description="AgriSmart India live camera scanner")
    ap.add_argument("--host", type=str, default="0.0.0.0", help="host interface to bind (default 0.0.0.0 for Wi-Fi)")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    t = time.perf_counter()
    print("Loading the model...", flush=True)
    import torch
    torch.set_num_threads(args.threads)
    INFER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model",
                               initializer=torch.set_num_threads, initargs=(args.threads,))
    BACKEND = INFER.submit(load_backend).result()
    INFER.submit(BACKEND["predict_proba"], Image.new("RGB", (224, 224), (90, 140, 70)), None).result()
    print(f"Ready in {time.perf_counter() - t:.1f}s", flush=True)

    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError:
        sys.exit(f"Port {args.port} is already in use. Try: python live_camera_india.py --port {args.port + 1}")

    wifi_ip = get_wifi_ip()
    local_url = f"http://127.0.0.1:{args.port}/"
    print("\n=======================================================", flush=True)
    print(" AgriSmart India model server is LIVE" + (" on Wi-Fi!" if args.host != "127.0.0.1" else "!"), flush=True)
    print(f" Web browser (laptop): {local_url}", flush=True)
    if args.host != "127.0.0.1":
        print(f" Mobile app (Wi-Fi URL): http://{wifi_ip}:{args.port}/", flush=True)
    print("=======================================================\n", flush=True)
    if not args.no_browser:
        webbrowser.open(local_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
