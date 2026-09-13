# AgriSmart offline

The crop-disease checker as an app that needs **no internet and no server**. The model runs on the phone or
laptop itself. It comes in two forms with the same features:

| | What to download (GitHub release `offline-app-v1`) | How to start |
|---|---|---|
| **Android** | `AgriSmart-offline.apk` | Install it (allow "install unknown apps" once), open **AgriSmart** |
| **Windows** | `AgriSmart-offline-windows.zip` | Unzip, double-click **`run.bat`**. The app opens in the browser at `http://localhost:8777`. Close the black window to stop it |

Everything works with no internet: the models, the voice clips and the page are all inside the app. With internet,
the app adds a few extras by itself, with no sign-up and no keys (see **Online extras** below).

## What it does

- **Take a photo once.** Press **Start camera** and hold one leaf inside the box. When the picture is steady,
  the app takes the photo by itself (**Auto capture**), or you can press **Capture**. The photo then freezes on
  screen and the app checks it, so you don't need to keep holding the phone up. **Scan again** goes back to the
  camera.
- **Upload a photo** from the gallery or files instead.
- **Answer:** the crop, the disease (or "healthy") and how sure the model is.
- **Why it happens:** the cause of the disease, for example a fungus spread by rain splash, a virus carried by
  whiteflies, or a nutrient shortage.
- **What to do:** first-aid steps with the product and dose, when to repeat, and prevention, safety and when to
  call the Krishi Vigyan Kendra.
- **Voice in 13 languages:** English, Hindi, Bengali, Marathi, Telugu, Tamil, Gujarati, Urdu, Kannada, Odia,
  Malayalam, Punjabi and Assamese. The answer, the cause and every step are read out in the chosen language.
  The voice **finishes even if you move the phone away**, because nothing new is checked until you press
  Scan again. Use **Stop voice** to stop it, and **Listen** to hear it again.
- **Three models to choose from:**
  - **India v1** (recommended): 59 crops, 387 classes.
  - **India v2**: the 336 px re-train.
  - **Core model**: the 13-crop, 28-class model from the hackathon.
- **Pick your crop** (optional). This gives better answers.
  - If the photo doesn't look like the crop you picked, the app asks **"Is this sugarcane? Yes / No"**. Yes gives the
    answer for sugarcane, because the farmer knows their own field. No lets the model guess the plant.
  - If the model can't tell two similar plants apart, it asks which crop it is.
- **Online extras** (when there is internet; on by default, no sign-up and no keys; switch off with **Online check**):
  - **Learn more (Wikipedia):** a short article about the disease or plant, in the chosen language when Wikipedia
    has one (otherwise in English), with a link to the full article.
  - **Similar photos** of the same disease from Wikimedia Commons.
  - **Search with Google Lens:** hands the photo to Google Lens (the Lens app on Android; on Windows the photo is
    copied and Google Lens opens in the browser, press Ctrl+V). Lens can name any plant, including ones the offline
    model doesn't know, such as neem or tulsi.
  - Only the crop and disease names go to Wikipedia. The photo leaves the device only when you tap the Lens button.
  - There is no AI answer inside the app: every AI service that reads photos now needs an API key (the free no-key
    ones, such as Pollinations, have closed), and a key would have to live on a server run by the team.
- **Extras:**
  - A warning for dark or blurry photos.
  - **Careful mode**, which also checks the mirror image. It is slower and a little steadier.
  - **History** of the last 20 checks, kept on the device.
  - **Share**, which sends the answer and the steps to WhatsApp, SMS and similar apps.
  - Urdu is shown right-to-left.

## The models

| Model | Crops / classes | File | Photo size | Field macro-F1 |
|---|---|---|---|---|
| India v1 (`india_v1`) | 59 / 387 | 58 MB | 224 px | 0.74 |
| India v2 (`india_v2`) | 59 / 387 | 59 MB | 336 px | 0.74 |
| Core (`core`) | 13 / 28 | 26 MB | 224 px | see `model/README.md` |

- **Quantization.** The models are quantized from the PyTorch weights with weight-only quantization.
  `tools/export_onnx.py` compares each quantized model with the original on 280 real leaf photos:
  - India v1, 4-bit weights: same answer on 99.6% of the photos.
  - India v2, 4-bit weights: same answer on 100%.
  - Core, 8-bit weights: same answer on 98.6%. 4-bit weights changed about 13% of the smaller core model's answers.
  - Plain int8 quantization changed about 7% of the India model's answers and distorted its confidence, so it
    was not used.
- **Where they run.** The page runs the models with onnxruntime-web (WebAssembly, several CPU threads).
- **Speed.** A photo takes about 0.2–1 s on a laptop, and a few seconds on a mid-range phone with India v2.

The answers are guidance, not a diagnosis. On real field photos the India models name the right disease
roughly three times out of four. Always check the product label, and confirm with an agriculture officer
before spraying.

## About the voice

- **The voices** are Meta's MMS-TTS models, one per language, each trained on recordings of native speakers.
  Every sentence the app can say was recorded ahead of time with these voices, so the voice works offline on
  every phone. Most phones have no built-in Odia or Assamese voice, for example.
- **Numbers.** The voices cannot read digits, so spoken steps say "the amount is shown on the screen" and use
  the day words ("ten days").
- **Product names** are written in each language's script so they are pronounced locally, for example
  मैंकोजेब, மைம்கோஜேப், مینکوزیب.
- **Please have a native speaker listen to a few answers in each language.** The recordings were made
  automatically and nobody has listened to them yet.
- The MMS voices are licensed **CC BY-NC 4.0**, so they are for non-commercial use.

## Building it yourself

All commands run from the repository root. Python needs `torch timm onnx onnxruntime transformers soundfile uroman indic-transliteration`.

```bash
python offline/tools/export_onnx.py          # models -> offline/web/models/*.onnx + models.json
python offline/tools/build_offline_data.py   # translations, causes, advice -> offline/web/data/app.json
python offline/tools/build_audio.py          # voice clips -> offline/web/audio/<lang>/ (hours on a CPU; --langs to split)
cd offline/android && gradlew.bat assembleRelease   # APK (copies offline/web into the app)
powershell -File offline/tools/package_release.ps1  # both downloads -> offline/build/release/
```

The models, voice clips and build output are not stored in git. They are published as release files instead.
If you run `run.bat` from a fresh clone, `tools/fetch_assets.ps1` downloads them once.

## Files

- `web/`: the app itself (`index.html`, `app.js`, `style.css`, `data/app.json`, `ort/` with onnxruntime-web
  1.22, `models/`, `audio/`).
- `run.bat` and `serve.ps1`: a small local web server for Windows. It uses PowerShell only, with nothing to
  install. It sends the cross-origin-isolation headers the multi-threaded model needs.
- `android/`: the APK. It is a plain Java WebView that serves `web/` from inside the APK, with the same headers.
- `tools/`: the build scripts above.
