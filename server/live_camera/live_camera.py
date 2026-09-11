"""AgriSmart live scan: point the laptop camera at a leaf, or at a phone showing a leaf photo,
and the page shows the disease and says it out loud.

Run from the project folder:
    .venv/Scripts/python server/live_camera/live_camera.py
then press "Start camera" in the browser tab that opens and allow camera access.
Camera frames are never sent anywhere else. Binds to every network interface by default so a phone on
the same Wi-Fi (the Android app) can reach it; pass --host 127.0.0.1 to keep it laptop-only.
"""
import argparse
import io
import json
import sys
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
import predict  # noqa: E402  (model/predict.py)

PAGE = Path(__file__).with_name("live_camera.html").read_bytes()
MAX_UPLOAD = 8 * 1024 * 1024

# PyTorch starts its own team of CPU worker threads for every thread that runs the model. Answering each web
# request on a fresh thread therefore piles up thread teams until Windows runs out of memory
# ("OMP: Cannot create thread"). So every prediction runs on this one long-lived thread instead.
INFER = None  # ThreadPoolExecutor with a single worker, created in main()

# label -> (crop, disease) the way a farmer would say it
DISPLAY = {
    "Apple___Apple_scab": ("Apple", "Apple scab"),
    "Apple___Cedar_apple_rust": ("Apple", "Cedar apple rust"),
    "Apple___healthy": ("Apple", "Healthy"),
    "Blueberry___healthy": ("Blueberry", "Healthy"),
    "Cherry_(including_sour)___healthy": ("Cherry", "Healthy"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("Corn (maize)", "Gray leaf spot"),
    "Corn_(maize)___Common_rust_": ("Corn (maize)", "Common rust"),
    "Corn_(maize)___Northern_Leaf_Blight": ("Corn (maize)", "Northern leaf blight"),
    "Grape___Black_rot": ("Grape", "Black rot"),
    "Grape___healthy": ("Grape", "Healthy"),
    "Peach___healthy": ("Peach", "Healthy"),
    "Pepper,_bell___Bacterial_spot": ("Bell pepper", "Bacterial spot"),
    "Pepper,_bell___healthy": ("Bell pepper", "Healthy"),
    "Potato___Early_blight": ("Potato", "Early blight"),
    "Potato___Late_blight": ("Potato", "Late blight"),
    "Raspberry___healthy": ("Raspberry", "Healthy"),
    "Soybean___healthy": ("Soybean", "Healthy"),
    "Squash___Powdery_mildew": ("Squash", "Powdery mildew"),
    "Strawberry___healthy": ("Strawberry", "Healthy"),
    "Tomato___Bacterial_spot": ("Tomato", "Bacterial spot"),
    "Tomato___Early_blight": ("Tomato", "Early blight"),
    "Tomato___Late_blight": ("Tomato", "Late blight"),
    "Tomato___Leaf_Mold": ("Tomato", "Leaf mold"),
    "Tomato___Septoria_leaf_spot": ("Tomato", "Septoria leaf spot"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("Tomato", "Spider mites"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("Tomato", "Yellow leaf curl virus"),
    "Tomato___Tomato_mosaic_virus": ("Tomato", "Mosaic virus"),
    "Tomato___healthy": ("Tomato", "Healthy"),
}


def describe(label, p):
    crop, disease = DISPLAY.get(label) or (label.split("___")[0].replace("_", " "),
                                           label.split("___")[-1].replace("_", " "))
    return {"label": label, "crop": crop, "disease": disease, "healthy": disease.lower() == "healthy", "p": round(p, 4)}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/health":
            self._json({"ok": True, "classes": len(predict.load()[2])})
        else:
            self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        if self.path != "/predict":
            return self._send(404, b"Not found", "text/plain")
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_UPLOAD:
            return self._json({"error": "Send one JPEG or PNG image of up to 8 MB."}, 400)
        try:
            image = Image.open(io.BytesIO(self.rfile.read(length))).convert("RGB")
        except Exception:
            return self._json({"error": "That file could not be read as an image."}, 400)
        t = time.perf_counter()
        # one pass (no mirror averaging) keeps the live view responsive
        probs = INFER.submit(predict.predict_proba, image, tta=False).result()
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        self._json({"top": [describe(label, p) for label, p in top], "ms": round((time.perf_counter() - t) * 1000)})

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # keep the console quiet: one line per scan would flood it
        pass


def get_wifi_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser(description="AgriSmart live camera scanner")
    ap.add_argument("--host", type=str, default="0.0.0.0", help="host interface to bind (default 0.0.0.0 for Wi-Fi)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads the model may use (default 4)")
    args = ap.parse_args()

    global INFER
    torch.set_num_threads(args.threads)
    INFER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model",
                               initializer=torch.set_num_threads, initargs=(args.threads,))
    t = time.perf_counter()
    print("Loading the model...", flush=True)
    INFER.submit(predict.load).result()
    INFER.submit(predict.predict_proba, Image.new("RGB", (224, 224), (90, 140, 70)), tta=False).result()  # warm-up
    print(f"Model ready in {time.perf_counter() - t:.1f}s", flush=True)

    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError:
        sys.exit(f"Port {args.port} is already in use. Try: python live_camera.py --port {args.port + 1}")
    
    wifi_ip = get_wifi_ip()
    local_url = f"http://127.0.0.1:{args.port}/"
    wifi_url = f"http://{wifi_ip}:{args.port}/"
    
    print("\n=======================================================", flush=True)
    print(" AgriSmart Model Server is LIVE on Wi-Fi!", flush=True)
    print(f" Web Browser (Laptop): {local_url}", flush=True)
    print(f" Mobile App (Wi-Fi URL): {wifi_url}", flush=True)
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
